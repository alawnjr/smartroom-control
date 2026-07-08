#!/usr/bin/env python3
"""
Undistorted copies of calibrated recordings, made on the laptop after Save All.

For every recordings/<day>/<rec>/streams/<cam>/camera_main.mp4 whose sibling
metadata.json carries streams.camera_main.calibration (embedded by the Pi's
capture.py once its camera is checkerboard-calibrated), this writes a
lens-corrected copy to a SEPARATE folder next to the raw clip:

    streams/<cam>/camera_main.mp4               (raw — untouched, ground truth)
    streams/<cam>/undistorted/camera_main.mp4   (lens-corrected copy)

detect.py / action.py automatically analyze the undistorted copy when it's
fresh, so detection and pose estimation run on corrected frames. Recordings
without a calibration are skipped (analysis falls back to the raw clip).

Idempotent: a copy is remade only when missing or older than its source video /
metadata. Safe against concurrent runs (global flock). Triggered automatically
by the dashboard's Save All; also runnable by hand:

  python undistort.py            # all calibrated clips
  python undistort.py --force    # remake everything
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2

from calib_utils import load_undistort_maps

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


def out_path(mp4: Path) -> Path:
    return mp4.parent / "undistorted" / mp4.name


def has_calibration(mp4: Path) -> bool:
    try:
        meta = json.loads((mp4.parent / "metadata.json").read_text())
        return bool(meta["streams"]["camera_main"].get("calibration"))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def needs_undistort(mp4: Path, force: bool) -> bool:
    if not has_calibration(mp4):
        return False
    if force:
        return True
    out = out_path(mp4)
    if not out.exists():
        return True
    # Remake when the source video OR the calibration (metadata.json) is newer.
    newest_src = max(mp4.stat().st_mtime,
                     (mp4.parent / "metadata.json").stat().st_mtime)
    return out.stat().st_mtime < newest_src


def true_fps(mp4: Path, fallback: float) -> float:
    # The USB cams are variable-rate; frame_count/duration is the honest average
    # (same approach as the annotated videos elsewhere in this pipeline).
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets:format=duration",
             "-of", "json", str(mp4)],
            capture_output=True, text=True, check=True, timeout=60).stdout
        data = json.loads(out)
        frames = int(data["streams"][0]["nb_read_packets"])
        duration = float(data["format"]["duration"])
        if frames and duration:
            return frames / duration
    except Exception:  # noqa: BLE001
        pass
    return fallback


def process_clip(mp4: Path, root: Path) -> bool:
    cap = cv2.VideoCapture(str(mp4))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    und = load_undistort_maps(mp4, width, height)
    if und is None:
        cap.release()
        return False
    fps = true_fps(mp4, cap.get(cv2.CAP_PROP_FPS) or 30.0)

    out = out_path(mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_raw = out.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.remap(frame, und[0], und[1], cv2.INTER_LINEAR))
        frames += 1
    cap.release()
    writer.release()

    # Browser-playable h264, same as the annotated videos.
    final_tmp = out.with_suffix(".enc.mp4")
    proc = subprocess.run(["ffmpeg", "-y", "-i", str(tmp_raw), "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_tmp)],
                          capture_output=True)
    tmp_raw.unlink(missing_ok=True)
    if proc.returncode != 0 or not final_tmp.exists():
        final_tmp.unlink(missing_ok=True)
        print(f"  ENCODE FAILED: {mp4.relative_to(root)}", file=sys.stderr)
        return False
    os.replace(final_tmp, out)
    print(f"  undistorted {mp4.relative_to(root)} ({frames} frames)", file=sys.stderr)
    return True


def main():
    ap = argparse.ArgumentParser(description="Undistorted copies of calibrated recordings.")
    ap.add_argument("--force", action="store_true", help="remake even if up to date")
    args = ap.parse_args()

    root = saved_root()
    if not root.exists():
        print(f"no recordings dir: {root}", file=sys.stderr)
        return 0

    lock_file = open(root / ".undistort.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another undistort run is in progress; exiting", file=sys.stderr)
        return 0

    clips = [p for p in sorted(root.rglob("camera_main.mp4"))
             if "undistorted" not in p.parts and needs_undistort(p, args.force)]
    done = sum(process_clip(c, root) for c in clips)
    print(f"undistorted {done}/{len(clips)} clip(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
