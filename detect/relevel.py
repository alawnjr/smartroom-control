#!/usr/bin/env python3
"""
Re-level already-recorded clips against the vertical in their own depth.

A camera's extrinsics get their tilt from an AprilTag that is a few dozen pixels
across, so the room frame they define is only as level as that tag solve — and
measured across day_12..day_14, the D455's stored pose is a consistent 5.0-5.3
degrees off level. Every room map drawn from those clips is tilted by that much.

The fix needs no tag: a depth camera sees the room's true vertical directly, in
the shared normal of every horizontal surface in view (floor, ceiling, desks —
tens of thousands of samples against the tag's four corners). This rotates each
stored pose so its measured vertical IS the frame's Y axis, orientation only —
the camera stays exactly where PnP put it, and yaw is untouched, so the clip
keeps whichever room frame (tag 1 or tag 2) it was recorded in. Only the tilt
moves.

Clips whose depth shows too little horizontal surface to measure from are
SKIPPED, not guessed at (the D435 largely faces a wall, so most of its clips
fall here). Run detect/localize.py --force afterwards to rebuild the .geo.json
sidecars from the corrected poses.

The vertical measurement is ported from CityOSNode's realsense_extrinsics.py
(find_room_vertical), the same way calib_utils.py ports pixel_to_floor — the Pi
repo is not checked out here.

Usage:
  python detect/relevel.py --days day_12 day_13 day_14        # report only
  python detect/relevel.py --days day_12 --apply              # rewrite metadata
Env:
  SMARTROOM_SAVE_DIR   recordings root (default: <project>/recordings)
"""

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ported from realsense_extrinsics.py — keep these in step with the Pi's copy.
MIN_LEVEL_NORMALS = 15000
MAX_LEVEL_SCATTER_DEG = 6.0
# Cameras whose frames are stored rotated 180 (node.env SMARTROOM_DEPTH_FLIP);
# their sensor intrinsics need the same rotation before deprojecting.
FLIP_SERIALS = {"801312070607"}


def save_dir():
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


class Intrinsics:
    """The subset of rs.intrinsics the deprojection needs."""

    def __init__(self, cal, rotated=False):
        self.fx, self.fy = cal["fx"], cal["fy"]
        self.ppx, self.ppy = cal["ppx"], cal["ppy"]
        self.width, self.height = cal["width"], cal["height"]
        if rotated:                      # principal point mirrors with the image
            self.ppx = (self.width - 1) - self.ppx
            self.ppy = (self.height - 1) - self.ppy


def _surface_normals(depth_mm, intr, step=4):
    """Per-pixel surface normals (unit, camera frame), skipping depth edges."""
    h, w = depth_mm.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_mm
    points = np.stack([(us - intr.ppx) / intr.fx * z, (vs - intr.ppy) / intr.fy * z, z], axis=2)
    dx = points[step:-step, 2 * step:] - points[step:-step, :-2 * step]
    dy = points[2 * step:, step:-step] - points[:-2 * step, step:-step]
    zc = z[step:-step, step:-step]
    valid = ((zc > 300.0) & (zc < 8000.0)
             & (np.abs(dx[:, :, 2]) < 0.08 * zc) & (np.abs(dy[:, :, 2]) < 0.08 * zc)
             & (z[step:-step, 2 * step:] > 0) & (z[step:-step, :-2 * step] > 0)
             & (z[2 * step:, step:-step] > 0) & (z[:-2 * step, step:-step] > 0))
    normals = np.cross(dx[valid], dy[valid])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-6
    return normals[keep] / lengths[keep, None]


def find_room_vertical(depths_mm, intr, up_hint):
    """Dominant horizontal-surface normal — the room's vertical, in camera
    coordinates. Mean-shift from the hint so the hint's own error does not bias
    it. None when too little horizontal surface is in view."""
    up = np.asarray(up_hint, dtype=np.float64)
    up /= np.linalg.norm(up)
    sets = [n for n in (_surface_normals(d, intr) for d in depths_mm[:4]) if len(n)]
    if not sets:
        return None
    normals = np.concatenate(sets)
    normals = np.where((normals @ up)[:, None] < 0, -normals, normals)
    used = None
    for window_deg in (30.0, 15.0, 8.0, 5.0):
        near = normals[normals @ up > np.cos(np.radians(window_deg))]
        if len(near) < 2000:
            break
        up = near.mean(axis=0)
        up /= np.linalg.norm(up)
        used = near
    if used is None or len(used) < 2000:
        return None
    scatter = float(np.degrees(np.arccos(np.clip(used @ up, -1.0, 1.0))).mean())
    return {"up_cam": up, "normals_used": int(len(used)), "scatter_deg": round(scatter, 2)}


def minimal_rotation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def read_depth(path, width, height, count=6, start=30):
    """The first frames of a lossless-FFV1 depth clip, as raw z16."""
    out = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(path), "-vsync", "0",
                          "-frames:v", str(start + count), "-f", "rawvideo",
                          "-pix_fmt", "gray16le", "-"], capture_output=True).stdout
    frame = width * height
    a = np.frombuffer(out, "<u2")
    return a[:a.size // frame * frame].reshape(-1, height, width)[start:]


def relevel_recording(meta_path, apply):
    """-> list of per-camera report lines."""
    meta = json.loads(meta_path.read_text())
    base = meta_path.parent
    lines, changed = [], False

    for name, stream in list(meta.get("streams", {}).items()):
        if not name.endswith("_depth"):
            continue
        ext, cal = stream.get("extrinsics"), stream.get("calibration")
        if not ext or not cal or "rotation_cam_to_room" not in ext:
            continue
        serial = ext.get("camera_id")
        clip = base / stream["path"].split("/")[-1]
        if not clip.exists():
            lines.append(f"    {name}: depth clip missing")
            continue
        intr = Intrinsics(cal, rotated=serial in FLIP_SERIALS)
        scale = stream.get("depth_scale_m") or 0.001
        depths = [d.astype(np.float32) * scale * 1000.0
                  for d in read_depth(clip, cal["width"], cal["height"])]
        if not depths:
            lines.append(f"    {name}: no depth frames decoded")
            continue

        R = np.array(ext["rotation_cam_to_room"], dtype=np.float64).T   # room -> camera
        vertical = find_room_vertical(depths, intr, -R[:, 1])           # tag frame Y is DOWN
        if vertical is None or (vertical["normals_used"] < MIN_LEVEL_NORMALS
                                or vertical["scatter_deg"] > MAX_LEVEL_SCATTER_DEG):
            got = (f"{vertical['normals_used']} normals" if vertical else "no normals")
            lines.append(f"    {name}: SKIPPED — {got}, too little horizontal surface")
            continue

        up_room = R.T @ vertical["up_cam"]
        off_deg = float(np.degrees(np.arccos(np.clip(up_room @ [0.0, -1.0, 0.0], -1.0, 1.0))))
        # orientation only: rotate the frame's vertical onto the measured one and
        # keep the camera exactly where PnP placed it
        N = minimal_rotation(up_room, [0.0, -1.0, 0.0])
        cam_pos = np.array(ext["camera_position_mm"], dtype=np.float64)
        R_new = R @ N.T
        tvec = -R_new @ cam_pos
        lines.append(f"    {name}: {off_deg:5.2f} deg off level "
                     f"({vertical['normals_used']} normals, {vertical['scatter_deg']} scatter)"
                     + ("" if apply else "  [dry run]"))
        if not apply:
            continue

        patch = {
            "rvec": cv2.Rodrigues(R_new)[0].flatten().tolist(),
            "tvec_mm": tvec.tolist(),
            "rotation_cam_to_room": R_new.T.tolist(),
            "camera_position_mm": [round(float(v), 1) for v in cam_pos],
            "relevelled": {
                "corrected_deg": round(off_deg, 2),
                "normals_used": vertical["normals_used"],
                "scatter_deg": vertical["scatter_deg"],
                "note": "tilt re-measured from this clip's own depth; position and yaw unchanged",
                "at": dt.datetime.now().astimezone().isoformat(),
            },
        }
        # both the colour and depth entries of this camera carry the same block
        for other_name, other in meta["streams"].items():
            if (other.get("extrinsics") or {}).get("camera_id") == serial:
                other["extrinsics"].update(patch)
        changed = True

    if changed:
        backup = meta_path.with_suffix(".json.prelevel")
        if not backup.exists():
            shutil.copy2(meta_path, backup)
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, meta_path)
    return lines


def _mean_rotation(rotations):
    """Chordal mean of rotation matrices."""
    U, _, Vt = np.linalg.svd(np.mean(rotations, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def learn_corrections(metas):
    """The correction already applied to each calibration epoch, recovered by
    comparing a re-levelled pose against its .prelevel backup.

    -> {(camera_id, calibrated_at): rotation}"""
    found = {}
    for meta_path in metas:
        backup = meta_path.with_suffix(".json.prelevel")
        if not backup.exists():
            continue
        now, before = json.loads(meta_path.read_text()), json.loads(backup.read_text())
        for name, stream in now.get("streams", {}).items():
            ext = stream.get("extrinsics") or {}
            old = ((before.get("streams", {}).get(name) or {}).get("extrinsics")) or {}
            if "relevelled" not in ext or "rotation_cam_to_room" not in old:
                continue
            key = (ext.get("camera_id"), ext.get("calibrated_at"))
            N = (np.array(ext["rotation_cam_to_room"], float)
                 @ np.array(old["rotation_cam_to_room"], float).T)
            found.setdefault(key, []).append(N)
    return {k: _mean_rotation(v) for k, v in found.items()}


def propagate_recording(meta_path, corrections, apply):
    """Apply a correction learnt from a clip WITH depth to a clip without one.

    The live-inference segments carry no depth to measure a vertical from, but
    they carry the same camera's same calibration epoch — same physical pose,
    same room — so the tilt correction measured for that epoch is the correction
    for them too. Matched on (camera, calibrated_at) so a later recalibration
    never inherits an older epoch's fix."""
    meta = json.loads(meta_path.read_text())
    lines, changed = [], False
    for name, stream in meta.get("streams", {}).items():
        ext = stream.get("extrinsics")
        if not ext or "relevelled" in ext or "rotation_cam_to_room" not in ext:
            continue
        N = corrections.get((ext.get("camera_id"), ext.get("calibrated_at")))
        if N is None:
            lines.append(f"    {name}: no correction learnt for its calibration epoch")
            continue
        R_c2r = N @ np.array(ext["rotation_cam_to_room"], float)
        cam_pos = np.array(ext["camera_position_mm"], float)
        deg = float(np.degrees(np.arccos(np.clip((np.trace(N) - 1) / 2, -1.0, 1.0))))
        lines.append(f"    {name}: tilt corrected by {deg:.2f} deg from its epoch"
                     + ("" if apply else "  [dry run]"))
        if not apply:
            continue
        ext.update({
            "rvec": cv2.Rodrigues(R_c2r.T)[0].flatten().tolist(),
            "tvec_mm": (-R_c2r.T @ cam_pos).tolist(),
            "rotation_cam_to_room": R_c2r.tolist(),
            "relevelled": {
                "corrected_deg": round(deg, 2),
                "note": "no depth in this clip — correction carried over from a "
                        "depth clip of the same camera and calibration epoch",
                "at": dt.datetime.now().astimezone().isoformat(),
            },
        })
        changed = True
    if changed:
        backup = meta_path.with_suffix(".json.prelevel")
        if not backup.exists():
            shutil.copy2(meta_path, backup)
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, meta_path)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", nargs="+", required=True, help="day folder names or prefixes")
    ap.add_argument("--apply", action="store_true", help="write; otherwise report only")
    ap.add_argument("--propagate", action="store_true",
                    help="also fix depth-less clips, using the correction learnt for "
                         "their camera's calibration epoch")
    ap.add_argument("--learn-from", nargs="*", default=None,
                    help="days to learn epoch corrections from (default: --days)")
    args = ap.parse_args()

    root = save_dir()

    def collect(days):
        return sorted(m for day in days
                      for d in sorted(root.glob(f"{day}*"))
                      for m in d.glob("*/streams/*/metadata.json"))

    metas = collect(args.days)
    if not metas:
        print(f"no recordings under {root} matching {args.days}", file=sys.stderr)
        return 1
    print(f"{'Re-levelling' if args.apply else 'Checking'} {len(metas)} recordings under {root}\n")

    corrections = {}
    if args.propagate:
        corrections = learn_corrections(collect(args.learn_from or args.days))
        for (cam, at), N in sorted(corrections.items()):
            deg = np.degrees(np.arccos(np.clip((np.trace(N) - 1) / 2, -1.0, 1.0)))
            print(f"  learnt: camera {cam} calibrated {at} -> {deg:.2f} deg tilt correction")
        print()

    for meta_path in metas:
        rec = meta_path.parent.parent.parent
        lines = relevel_recording(meta_path, args.apply)
        if args.propagate:
            lines += propagate_recording(meta_path, corrections, args.apply)
        if lines:
            print(f"  {rec.parent.name}/{rec.name}")
            print("\n".join(lines))
    if not args.apply:
        print("\n(dry run — pass --apply to write, then rerun detect/localize.py --force)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
