#!/usr/bin/env python3
"""
Re-solve extrinsics for already-recorded clips from their OWN recorded frames.

Clips recorded before the calibration fixes (joint multi-tag solve + planar-PnP
branch selection + depth levelling) can carry a pose solved on the wrong PnP
branch — the classic flip that puts a camera metres off. This re-runs the
CURRENT realsense_extrinsics.calibrate_from_samples against a clip's recorded
color+depth and writes the corrected pose back into its metadata, with the
camera never present: the recorded tag views are all the solver needs.

The physical tags haven't moved, so today's calibration/tags.json (+ room_level)
apply to every era. We solve in the tag-2 room frame (where both cameras can see
a mapped tag), then, when the clip's own reference tag differs, re-express the
result in that tag's frame via tags.json — a rigid transform that preserves the
solve. The camera's position comes from THIS clip's tag observations, so an era
where the camera sat differently is handled correctly (unlike a restamp).

Depth is recorded aligned to color, so color frame k pairs with depth frame k.

Usage:
  python detect/resolve_from_recording.py --days day_11 day_12 --tag-frame 1
  python detect/resolve_from_recording.py --days day_11 --tag-frame 1 --apply
Env:
  SMARTROOM_SAVE_DIR   recordings root
  CALIB_SRC            dir holding the authoritative tags.json + room_level.json
                       (default: the newest calib embedded near the clips is not
                       used — pass the Pi-synced calibration dir)
"""

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import realsense_extrinsics as rx  # noqa: E402  (scp'd alongside for analysis)

FLIP_SERIALS = {"801312070607"}
N_FRAMES = 8
FRAME_START = 30
TAG_HEIGHT_MM = {1: 1110.0, 2: 1590.0}


def save_dir():
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


class Intr:
    def __init__(self, cal):
        self.fx, self.fy, self.ppx, self.ppy = cal["fx"], cal["fy"], cal["ppx"], cal["ppy"]
        self.width, self.height = cal["width"], cal["height"]
        self.model, self.coeffs = cal.get("model", "none"), list(cal.get("coeffs", [0, 0, 0, 0, 0]))


def decode(path, w, h, pix, bpp, n=N_FRAMES, start=FRAME_START):
    out = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(path), "-vsync", "0",
                          "-frames:v", str(start + n), "-f", "rawvideo",
                          "-pix_fmt", pix, "-"], capture_output=True).stdout
    frame = w * h * bpp
    a = np.frombuffer(out, np.uint8)
    return a[:a.size // frame * frame].reshape(-1, h, w, bpp)[start:]


def transform_to_tag(ext, tags, target_tag):
    """Re-express a tag-2-frame extrinsic in `target_tag`'s frame (rigid)."""
    if target_tag == 2:
        return (np.array(ext["rotation_cam_to_room"], float),
                np.array(ext["camera_position_mm"], float))
    entry = tags["tags"][str(target_tag)]
    R_t2 = np.array(entry["rotation_tag_to_room"], float)   # target -> tag2
    p_t = np.array(entry["position_mm"], float)             # target origin in tag2
    Rc2 = np.array(ext["rotation_cam_to_room"], float)      # cam -> tag2
    cam2 = np.array(ext["camera_position_mm"], float)
    return R_t2.T @ Rc2, R_t2.T @ (cam2 - p_t)


def resolve_recording(meta_path, calib_src, target_tag, apply):
    meta = json.loads(meta_path.read_text())
    base = meta_path.parent
    tags = json.loads((calib_src / "tags.json").read_text())
    lines = []
    # All-or-nothing: only rewrite a recording when EVERY colour stream re-solves,
    # so a clip never ends up with one camera in the new frame and one in the old.
    color_streams = [n for n in meta.get("streams", {}) if n.endswith("_color")]
    solved_updates = {}   # stream name -> (new extrinsics dict)

    # temp calib dir so calibrate_from_samples reads the authoritative map/level
    # and its writes never touch the live calibration
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for fn in ("tags.json", "room_level.json"):
            if (calib_src / fn).exists():
                shutil.copy2(calib_src / fn, tmp / fn)
        os.environ["SMARTROOM_TAG_ID"] = "2"      # solve in the tag-2 frame

        for name, stream in meta.get("streams", {}).items():
            if not name.endswith("_color"):
                continue
            ext, cal = stream.get("extrinsics"), stream.get("calibration")
            if not ext or not cal:
                continue
            serial = ext.get("camera_id")
            base_key = name[:-len("_color")]
            color = base / stream["path"].split("/")[-1]
            depth_mkv = base / f"{base_key}_depth.mkv"
            if not color.exists() or not depth_mkv.exists():
                lines.append(f"    {name}: missing color/depth clip")
                continue

            intr = Intr(cal)
            if serial in FLIP_SERIALS:
                intr = rx.rotate180_intrinsics(intr)
            w, h = cal["width"], cal["height"]
            scale = float(stream_depth_scale(meta, base_key))
            colors = decode(color, w, h, "bgr24", 3)
            depths = decode(depth_mkv, w, h, "gray16le", 2)
            n = min(len(colors), len(depths))
            if n == 0:
                lines.append(f"    {name}: no frames decoded")
                continue
            depths = depths[:n].view("<u2").reshape(n, h, w).astype(np.float32) * scale
            samples = [(colors[i].copy(), depths[i]) for i in range(n)]

            ok, msg = rx.calibrate_from_samples(samples, intr, serial,
                                                camera_name=stream.get("camera", "RealSense"),
                                                tag_id=2, out_dir=tmp)
            if not ok:
                lines.append(f"    {name}: could not re-solve ({msg.split(' — ')[0]})")
                continue
            solved = json.loads((tmp / f"{serial}.extrinsics.json").read_text())
            Rc, cam = transform_to_tag(solved, tags, target_tag)
            old = [round(v) for v in ext.get("camera_position_mm", [])]
            new = [round(float(v)) for v in cam]
            lines.append(f"    {name}: {old} -> {new}  (tag{target_tag} frame; "
                         f"{msg.split(',')[0].replace('camera at ', '')})"
                         + ("" if apply else "  [dry run]"))
            rvec = cv2.Rodrigues(Rc.T)[0].flatten()
            new_ext = {
                **{k: solved[k] for k in ("camera_id", "camera", "source", "tag",
                                          "reprojection_error_px", "depth_agreement_mm",
                                          "levelled", "solved_from_tags") if k in solved},
                "frame": f"tag {target_tag}: origin=center, X=right, Y=DOWN "
                         f"(gravity-levelled: up is -Y), Z=out of tag; units mm",
                "tag": {"family": "36h11", "id": target_tag,
                        "size_mm": solved.get("tag", {}).get("size_mm", 138.4)},
                "rvec": rvec.tolist(),
                "tvec_mm": (-Rc.T @ cam).tolist(),
                "rotation_cam_to_room": Rc.tolist(),
                "camera_position_mm": [round(float(v), 1) for v in cam],
                "resolved_from_recording": {
                    "note": "extrinsics re-solved from this clip's own recorded frames "
                            "with the current algorithm; re-expressed in the tag-"
                            f"{target_tag} frame",
                    "at": dt.datetime.now().astimezone().isoformat(),
                },
            }
            solved_updates[name] = (base_key, new_ext)

    # all-or-nothing: every colour stream must have re-solved
    missing = [n for n in color_streams if n not in solved_updates]
    if missing:
        lines.append(f"    -> SKIPPED recording: {', '.join(missing)} did not re-solve "
                     f"(left in its original frame to avoid mixing)")
        return lines
    if apply and solved_updates:
        for name, (base_key, new_ext) in solved_updates.items():
            meta["streams"][name]["extrinsics"] = new_ext
            dep = meta["streams"].get(f"{base_key}_depth")
            if dep is not None:
                dep["extrinsics"] = json.loads(json.dumps(new_ext))
        meta.setdefault("room_frame", {})
        meta["room_frame"]["reference_tag"] = {"family": "36h11", "id": target_tag,
                                               "size_mm": 138.4}
        meta["room_frame"]["tag_center_above_floor_mm"] = TAG_HEIGHT_MM.get(target_tag, 1110.0)
        backup = meta_path.with_suffix(".json.preresolve")
        if not backup.exists():
            shutil.copy2(meta_path, backup)
        tmp_out = meta_path.with_suffix(".json.tmp")
        tmp_out.write_text(json.dumps(meta, indent=2))
        os.replace(tmp_out, meta_path)
    return lines


def stream_depth_scale(meta, base_key):
    dep = meta.get("streams", {}).get(f"{base_key}_depth") or {}
    return dep.get("depth_scale_m") or 0.001


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--tag-frame", type=int, default=1, help="reference tag for the output frame")
    ap.add_argument("--calib-src", default=None,
                    help="dir with the authoritative tags.json + room_level.json")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    root = save_dir()
    calib_src = Path(args.calib_src) if args.calib_src else (PROJECT_ROOT / "calibration")
    if not (calib_src / "tags.json").exists():
        print(f"no tags.json under {calib_src} — pass --calib-src", file=sys.stderr)
        return 1
    metas = sorted(m for day in args.days
                   for d in sorted(root.glob(f"{day}*"))
                   for m in d.glob("*/streams/*/metadata.json"))
    if not metas:
        print(f"no recordings under {root} matching {args.days}", file=sys.stderr)
        return 1
    print(f"{'Re-solving' if args.apply else 'Checking'} {len(metas)} recordings "
          f"into the tag-{args.tag_frame} frame\n")
    for meta_path in metas:
        rec = meta_path.parent.parent.parent
        lines = resolve_recording(meta_path, calib_src, args.tag_frame, args.apply)
        if lines:
            print(f"  {rec.parent.name}/{rec.name}")
            print("\n".join(lines))
    if not args.apply:
        print("\n(dry run — pass --apply, then rerun detect/localize.py --force)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
