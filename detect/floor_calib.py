#!/usr/bin/env python3
"""
Solve the AprilTag's height above the floor from a person of known stature.

Room positions (see calib_utils.pixel_to_floor) need one number the extrinsic
calibration can't provide: where the floor is below the wall-mounted tag. Rather
than a tape measure, this solves it from recordings of a person whose height is
known — for each standing pose sample, find the tag-center height H that makes
the ankle->nose projection equal the reference stature; the median over all
samples is the answer.

Uses only files that already exist: clips whose metadata.json embeds extrinsics
AND that have a camera_main.keypoints.yolo26n-pose.json sidecar (written by
detect.py's pose pass on the undistorted video). No video decoding, no models.

  python detect/floor_calib.py                 # scan all recordings
  python detect/floor_calib.py --height-cm 174 # reference person's stature

Writes recordings/.floor.json keyed by camera_id:
  {"usb-046d_...": {"tag_height_mm": 1202, "samples": 38, ...}}
Consumed by calib_utils.tag_height_mm (env SMARTROOM_TAG_HEIGHT_MM overrides).
"""

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from calib_utils import _ROLL_FIX, ANKLE_JOINT_HEIGHT_MM

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Anthropometric offset (mm): the pose model sees the nose, stature is to the crown.
NOSE_TO_CROWN_MM = 145.0
MIN_KP_CONF = 0.5
# Upright check: nose roughly above the ankles in the image (rejects sitting,
# bending, lying poses whose ankle->nose span isn't the stature).
MAX_LEAN_RATIO = 0.35
H_RANGE_MM = (300.0, 2900.0)
NOSE, L_ANKLE, R_ANKLE = 0, 15, 16


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


def clip_geometry(meta_path: Path):
    """(camera_id, K, R_up, cam_up, (w, h)) from an extrinsics-bearing metadata.json."""
    try:
        stream = json.loads(meta_path.read_text())["streams"]["camera_main"]
        cal, ext = stream["calibration"], stream["extrinsics"]
        K = np.array(cal["camera_matrix"], dtype=np.float64)
        cal_w, cal_h = (int(v) for v in cal["image_size"])
        res = stream["resolution"]  # [w, h] list (older recordings: "WxH")
        w, h = (int(v) for v in (res.split("x") if isinstance(res, str) else res))
        R = _ROLL_FIX @ np.array(ext["rotation_cam_to_room"], dtype=np.float64)
        cam = _ROLL_FIX @ np.array(ext["camera_position_mm"], dtype=np.float64)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if (cal_w, cal_h) != (w, h) and cal_w and cal_h:
        sx, sy = w / cal_w, h / cal_h
        K = K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
    return ext.get("camera_id", ""), K, R, cam, (w, h)


def solve_sample(nose_px, ankle_px, K, R, cam, stature_mm):
    """Tag-center height H (mm) that puts this standing person at stature_mm,
    or None. Frame here: origin = tag center, Y up, floor at Y = -H."""
    Kinv = np.linalg.inv(K)

    def ray(px):
        d = R @ (Kinv @ np.array([px[0], px[1], 1.0]))
        return d / np.linalg.norm(d)

    da, dn = ray(ankle_px), ray(nose_px)
    if da[1] >= -1e-6:  # ankle ray must point down
        return None

    def stature_at(H):
        t = ((-H + ANKLE_JOINT_HEIGHT_MM) - cam[1]) / da[1]
        if t <= 0:
            return None
        p = cam + t * da  # ankle joint in the room
        # Nose depth: the point on the nose ray closest (in plan view) to the
        # vertical line through the ankles.
        den = dn[0] * dn[0] + dn[2] * dn[2]
        if den < 1e-9:
            return None
        tn = ((p[0] - cam[0]) * dn[0] + (p[2] - cam[2]) * dn[2]) / den
        if tn <= 0:
            return None
        nose_y = cam[1] + tn * dn[1]
        return (nose_y + H) + NOSE_TO_CROWN_MM

    def err(H):
        s = stature_at(H)
        return (s - stature_mm) if s is not None else 1e9

    lo, hi = H_RANGE_MM
    if err(lo) * err(hi) > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if err(lo) * err(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def clip_samples(kp_path: Path, K, R, cam, size, stature_mm):
    """All solved H values for one clip's keypoints sidecar."""
    try:
        frames = json.loads(kp_path.read_text())["frames"]
    except (OSError, ValueError, KeyError):
        return []
    w, h = size
    out = []
    for fr in frames:
        for person in fr.get("persons", []):
            kpts = np.array(person["kpts"], dtype=np.float64) * [w, h]
            conf = np.array(person["conf"], dtype=np.float64)
            if len(kpts) < 17 or min(conf[NOSE], conf[L_ANKLE], conf[R_ANKLE]) < MIN_KP_CONF:
                continue
            ankle = kpts[[L_ANKLE, R_ANKLE]].mean(axis=0)
            nose = kpts[NOSE]
            dy = abs(nose[1] - ankle[1])
            if dy < 1 or abs(nose[0] - ankle[0]) > MAX_LEAN_RATIO * dy:
                continue
            H = solve_sample(nose, ankle, K, R, cam, stature_mm)
            if H is not None and H_RANGE_MM[0] + 1 < H < H_RANGE_MM[1] - 1:
                out.append(H)
    return out


def main():
    ap = argparse.ArgumentParser(description="Solve the tag's floor height from a known-height person.")
    ap.add_argument("--height-cm", type=float, default=174.0,
                    help="reference person's stature (default 174)")
    args = ap.parse_args()
    stature_mm = args.height_cm * 10.0

    root = saved_root()
    per_camera = defaultdict(list)
    for meta_path in sorted(root.rglob("metadata.json")):
        if "undistorted" in meta_path.parts:
            continue
        geo = clip_geometry(meta_path)
        if geo is None:
            continue
        camera_id, K, R, cam, size = geo
        kp_path = meta_path.parent / "camera_main.keypoints.yolo26n-pose.json"
        if not kp_path.exists():
            continue
        samples = clip_samples(kp_path, K, R, cam, size, stature_mm)
        rel = meta_path.parent.relative_to(root)
        if samples:
            print(f"  {rel}: {len(samples)} standing samples, "
                  f"median H {np.median(samples):.0f} mm", file=sys.stderr)
            per_camera[camera_id].extend(samples)
        else:
            print(f"  {rel}: no usable standing samples", file=sys.stderr)

    if not per_camera:
        print("ERROR: no clips with extrinsics + pose keypoints + a standing person found.",
              file=sys.stderr)
        return 1

    result = {}
    for camera_id, samples in per_camera.items():
        arr = np.array(samples)
        med = float(np.median(arr))
        q1, q3 = (float(np.percentile(arr, q)) for q in (25, 75))
        result[camera_id] = {
            "tag_height_mm": round(med, 1),
            "samples": len(arr),
            "iqr_mm": [round(q1, 1), round(q3, 1)],
            "ref_height_cm": args.height_cm,
            "estimatedAt": dt.datetime.now().astimezone().isoformat(),
        }
        print(f"{camera_id}: tag center {med:.0f} mm above floor "
              f"({len(arr)} samples, IQR {q1:.0f}-{q3:.0f})", file=sys.stderr)

    out_path = root / ".floor.json"
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except (OSError, ValueError):
            existing = {}
    existing.update(result)
    fd, tmp = tempfile.mkstemp(dir=str(root), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, out_path)
    print(f"saved -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
