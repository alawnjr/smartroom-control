#!/usr/bin/env python3
"""
Person-detection over the saved recordings, for the smartroom-control dashboard.

Runs one or more pretrained YOLO26 models (OpenVINO, intel:cpu) on each
`recordings/<node>/.../streams/camera_main.mp4`, counting PEOPLE only (COCO
class 0). For each clip AND each model it writes two siblings:

  camera_main.detections.<model>.json   occupancy stats + per-sampled-frame timeline
  camera_main.annotated.<model>.mp4     boxes burned in, H.264 (browser-playable)

so the dashboard can toggle between models (nano / s / m). Idempotent per
(clip, model); safe against concurrent runs (a global flock); writes a
`status:"analyzing"` marker first so the dashboard can show progress.

Config (env):
  SMARTROOM_SAVE_DIR          recordings root (default: <project>/recordings)
  SMARTROOM_YOLO_MODELS       comma list of model keys (default yolo26l,yolo26n-pose)
  SMARTROOM_YOLO_DIR          dir holding <key>_openvino_model/ (default ~/Code/yolo-bench)
  SMARTROOM_DETECT_IMGSZ      inference size (default 640)
  SMARTROOM_DETECT_SAMPLE_FPS frames/sec to analyze (default 5)
  SMARTROOM_DETECT_ANNOTATE   1/0 produce annotated video (default 1)

Usage:
  python detect.py                 # all models over all unprocessed clips
  python detect.py --path <rel>    # one clip (recordings-relative), with --force
  python detect.py --force         # reprocess everything
"""

import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSON_CLASS = 0
SCHEMA_VERSION = 2


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


def pick_device():
    """Inference device: CUDA when a usable GPU is present, else OpenVINO CPU.
    Override with SMARTROOM_DETECT_DEVICE (e.g. "0", "cuda:0", "intel:cpu")."""
    dev = os.environ.get("SMARTROOM_DETECT_DEVICE", "auto")
    if dev != "auto":
        return dev
    try:
        import torch
        return "0" if torch.cuda.is_available() else "intel:cpu"
    except ImportError:
        return "intel:cpu"


DEVICE = pick_device()


def model_specs():
    """(key, model ref) for each configured model. On OpenVINO CPU the ref is
    the exported <key>_openvino_model/ dir; on CUDA it's plain .pt weights
    (the bench dir's copy when present, else the official name — ultralytics
    downloads it on first use)."""
    keys = [k.strip() for k in os.environ.get(
        # Only the large detector for object detection (nano/small/medium dropped);
        # the pose model stays (it's a separate skeleton task, not box detection).
        "SMARTROOM_YOLO_MODELS", "yolo26l,yolo26n-pose"
    ).split(",") if k.strip()]
    base = Path(os.environ.get("SMARTROOM_YOLO_DIR") or (Path.home() / "Code" / "yolo-bench"))
    if DEVICE != "intel:cpu":
        return [(k, p if (p := base / f"{k}.pt").exists() else Path(f"{k}.pt")) for k in keys]
    return [(k, base / f"{k}_openvino_model") for k in keys]


IMGSZ = int(os.environ.get("SMARTROOM_DETECT_IMGSZ", "640"))
SAMPLE_FPS = float(os.environ.get("SMARTROOM_DETECT_SAMPLE_FPS", "5"))
ANNOTATE = os.environ.get("SMARTROOM_DETECT_ANNOTATE", "1") != "0"
KPT_CONF = 0.3  # keypoint visibility threshold for drawing

# COCO-17 skeleton edges (joint index pairs), for drawing pose annotations.
COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
]


def _is_pose(key: str) -> bool:
    return "pose" in key


def _draw_skeleton(frame, persons):
    """persons: list of (kpts_xy [(x,y),...17], conf [c,...17])."""
    import cv2

    for kpts, conf in persons:
        for a, b in COCO_SKELETON:
            if conf[a] > KPT_CONF and conf[b] > KPT_CONF:
                cv2.line(frame, tuple(map(int, kpts[a])), tuple(map(int, kpts[b])), (0, 200, 0), 2)
        for (x, y), c in zip(kpts, conf):
            if c > KPT_CONF:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)


def sidecar_paths(mp4: Path, key: str):
    return (
        mp4.with_name(f"{mp4.stem}.detections.{key}.json"),
        mp4.with_name(f"{mp4.stem}.annotated.{key}.mp4"),
    )


def _atomic_write_json(path: Path, data: dict):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def needs_processing(mp4: Path, key: str, force: bool) -> bool:
    if force:
        return True
    json_path, annotated = sidecar_paths(mp4, key)
    if not json_path.exists():
        return True
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return True
    if data.get("status") != "done":
        return True
    # 2s tolerance: recordings aren't modified after saving, and this avoids
    # float-rounding flicker (matches lib/detections.ts).
    if data.get("sourceMtimeMs", 0) + 2000 < mp4.stat().st_mtime * 1000:
        return True
    if ANNOTATE and not annotated.exists():
        return True
    return False


def process_clip(model, key: str, mp4: Path):
    import cv2

    from calib_utils import analysis_source

    pose = _is_pose(key)
    json_path, annotated_path = sidecar_paths(mp4, key)
    source_mtime_ms = mp4.stat().st_mtime * 1000
    _atomic_write_json(json_path, {"schemaVersion": SCHEMA_VERSION, "status": "analyzing",
                                   "model": key, "source": mp4.name, "sourceMtimeMs": source_mtime_ms})

    # Decode the lens-corrected copy when the recording is calibrated (see
    # undistort.py); sidecar names/paths stay keyed to the raw clip.
    src = analysis_source(mp4)
    cap = cv2.VideoCapture(str(src))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = max(1, round(native_fps / SAMPLE_FPS))

    writer = None
    tmp_annotated = None
    if ANNOTATE and width and height:
        tmp_annotated = annotated_path.with_suffix(".raw.mp4")
        writer = cv2.VideoWriter(str(tmp_annotated), cv2.VideoWriter_fourcc(*"mp4v"),
                                 native_fps, (width, height))

    timeline = []
    keypoints_timeline = []  # pose only: normalized keypoints per sampled frame (for action models)
    last_boxes = []          # detection: [(x1,y1,x2,y2)]
    last_persons = []        # pose: [(kpts_xy, conf)] in pixels, for drawing
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            t = round(idx / native_fps, 3)
            if pose:
                res = model.predict(frame, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
                kp = res.keypoints
                xy = kp.xy.tolist() if kp is not None else []
                xyn = kp.xyn.tolist() if kp is not None else []
                conf = (kp.conf.tolist() if (kp is not None and kp.conf is not None) else
                        [[1.0] * len(p) for p in xy])
                last_persons = [(xy[i], conf[i]) for i in range(len(xy))]
                count = len(xy)
                timeline.append({"t": t, "count": count})
                keypoints_timeline.append({
                    "t": t,
                    "persons": [
                        {"kpts": [[round(x, 4), round(y, 4)] for x, y in xyn[i]],
                         "conf": [round(c, 3) for c in conf[i]]}
                        for i in range(len(xyn))
                    ],
                })
            else:
                res = model.predict(frame, imgsz=IMGSZ, classes=[PERSON_CLASS],
                                    device=DEVICE, verbose=False)[0]
                last_boxes = [tuple(map(int, b)) for b in res.boxes.xyxy.tolist()] if res.boxes else []
                timeline.append({"t": t, "count": len(last_boxes)})
        if writer is not None:
            count = len(last_persons) if pose else len(last_boxes)
            if pose:
                _draw_skeleton(frame, last_persons)
            else:
                for (x1, y1, x2, y2) in last_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            if count:
                cv2.putText(frame, f"people: {count}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            writer.write(frame)
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()

    if pose:
        # Bulky per-frame keypoints go in a separate sidecar (the dashboard
        # listing doesn't read this — it's input for the action classifier).
        _atomic_write_json(mp4.with_name(f"{mp4.stem}.keypoints.{key}.json"), {
            "schemaVersion": SCHEMA_VERSION, "model": key, "source": mp4.name,
            "sourceMtimeMs": source_mtime_ms, "nativeFps": round(native_fps, 3),
            "sampleFps": SAMPLE_FPS, "keypointFormat": "coco17_xyn",
            "frames": keypoints_timeline,
        })

    has_annotated = False
    if tmp_annotated is not None and tmp_annotated.exists():
        final_tmp = annotated_path.with_suffix(".enc.mp4")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_annotated), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_tmp)],
            capture_output=True,
        )
        tmp_annotated.unlink(missing_ok=True)
        if proc.returncode == 0 and final_tmp.exists():
            os.replace(final_tmp, annotated_path)
            has_annotated = True
        else:
            final_tmp.unlink(missing_ok=True)

    counts = [p["count"] for p in timeline]
    _atomic_write_json(json_path, {
        "schemaVersion": SCHEMA_VERSION,
        "status": "done",
        "error": None,
        "model": key,
        "source": mp4.name,
        "sourceMtimeMs": source_mtime_ms,
        "sourceVideo": "undistorted" if src != mp4 else "raw",
        "device": DEVICE,
        "class": "person",
        "analyzedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "durationSec": round(total / native_fps, 3) if total else None,
        "nativeFps": round(native_fps, 3),
        "sampleFps": SAMPLE_FPS,
        "framesAnalyzed": len(timeline),
        "maxPersons": max(counts) if counts else 0,
        "avgPersons": round(sum(counts) / len(counts), 2) if counts else 0,
        "timeline": timeline,
        "annotated": annotated_path.name if has_annotated else None,
        "hasAnnotated": has_annotated,
    })
    print(f"  [{key}] done: {mp4.relative_to(saved_root())}  max={max(counts) if counts else 0}", file=sys.stderr)


def mark_error(mp4: Path, key: str, message: str):
    json_path, _ = sidecar_paths(mp4, key)
    try:
        _atomic_write_json(json_path, {"schemaVersion": SCHEMA_VERSION, "status": "error",
                                       "model": key, "error": message, "source": mp4.name,
                                       "sourceMtimeMs": mp4.stat().st_mtime * 1000})
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="YOLO26 person-detection over saved recordings.")
    ap.add_argument("--path", action="append", metavar="REL",
                    help="clip to analyze, relative to the recordings root; repeatable for a subset")
    ap.add_argument("--force", action="store_true", help="reprocess even if results are current")
    args = ap.parse_args()

    root = saved_root()
    if not root.exists():
        print(f"no recordings dir: {root}", file=sys.stderr)
        return 0

    lock_file = open(root / ".detect.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another detection run is in progress; exiting", file=sys.stderr)
        return 0

    # Become a process-group leader and publish our PID so the dashboard's Cancel
    # button can kill this run (and its ffmpeg/forkserver children) regardless of
    # whether it was started by the API, the timer, or the path watcher.
    try:
        os.setpgrp()
    except OSError:
        pass
    pid_path = root / ".detect.pid"
    try:
        pid_path.write_text(str(os.getpid()))
    except OSError:
        pass

    try:
        return _run(root, args)
    finally:
        try:
            pid_path.unlink()
        except OSError:
            pass


def _run(root: Path, args) -> int:
    # Every RGB source gets detection + pose: legacy webcam clips plus both
    # RealSense color streams (new recordings are depth-cameras-only).
    RGB_SOURCES = ("camera_main.mp4", "camera_d455_color.mp4", "camera_d435_color.mp4")
    if args.path:
        clips = [root / p for p in args.path]
    else:
        clips = sorted((p for name in RGB_SOURCES for p in root.rglob(name)),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    # undistorted/ holds lens-corrected COPIES of clips, not additional clips.
    clips = [c for c in clips if c.exists() and "undistorted" not in c.parts]

    if shutil.which("ffmpeg") is None and ANNOTATE:
        print("warning: ffmpeg not found; annotated videos will be skipped", file=sys.stderr)

    specs = model_specs()
    from ultralytics import YOLO
    started = dt.datetime.now(dt.timezone.utc)
    processed = errors = 0
    for key, md in specs:
        if not md.exists():
            print(f"skip model {key}: OpenVINO dir missing ({md}) — export it first", file=sys.stderr)
            continue
        todo = [c for c in clips if needs_processing(c, key, args.force)]
        print(f"[{key}] {len(todo)}/{len(clips)} clip(s) to process", file=sys.stderr)
        if not todo:
            continue
        model = YOLO(str(md))
        for mp4 in todo:
            try:
                process_clip(model, key, mp4)
                processed += 1
            except Exception as error:  # noqa: BLE001
                errors += 1
                print(f"  [{key}] error: {error}", file=sys.stderr)
                mark_error(mp4, key, str(error))
    if processed or errors:
        write_run_stats("detect", "Object detection", started, processed, errors)
    return 0


def write_run_stats(kind: str, label: str, started, processed: int, errors: int):
    # Record the just-finished batch so the dashboard sidebar can show "last run"
    # stats (elapsed, count). processed counts (clip x model) inferences done.
    root = saved_root()
    finished = dt.datetime.now(dt.timezone.utc)
    elapsed = (finished - started).total_seconds()
    data = {
        "kind": kind, "label": label,
        "startedAt": started.isoformat(), "finishedAt": finished.isoformat(),
        "elapsedSec": round(elapsed, 1), "processed": processed, "errors": errors,
        "perClipSec": round(elapsed / processed, 1) if processed else None,
    }
    try:
        tmp = root / f".last_run.{kind}.json.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, root / f".last_run.{kind}.json")
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
