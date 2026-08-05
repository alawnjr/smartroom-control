#!/usr/bin/env python3
"""
Object detection over the saved recordings, for the smartroom-control dashboard.

Runs one or more pretrained YOLO26 models (OpenVINO, intel:cpu) on each
`recordings/<node>/.../streams/camera_main.mp4`, reporting EVERY COCO class the
model knows — people plus the room's contents (chairs, laptops, bottles,
monitors). Restrict it with SMARTROOM_DETECT_CLASSES if you only want some.

People are still counted separately and in the old place (`count` per frame,
`maxPersons`/`avgPersons` in the summary), so the file remains an occupancy record
as well as an object record. Pose models are person-only by nature and unaffected.

For each clip AND each model it writes two siblings:

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
  SMARTROOM_DETECT_CLASSES    "all" (default), "person", or a COCO name/id list
  SMARTROOM_DETECT_OBJ_CONF   drop detections below this confidence (default 0.35)

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


# Which COCO classes the DETECTION models report. Historically this was hardwired
# to person (class 0) and everything downstream counted occupancy; now the default
# is every class the model knows, because the room's contents — chairs, laptops,
# bottles, monitors — are the context an occupancy or activity claim sits in, and
# the detector was already paying to look at them.
#   "all" (default) | "person" | any comma list of COCO names or ids
# People remain reported separately and unchanged (`count`, maxPersons, avgPersons),
# so nothing that consumed this file as an occupancy record has to change.
DETECT_CLASSES = os.environ.get("SMARTROOM_DETECT_CLASSES", "all").strip()
# Objects below this confidence are dropped. YOLO's default 0.25 floor produces a
# long tail of one-frame guesses at room clutter; the per-frame lists are meant to
# be read by a person, so they are worth trimming.
OBJ_CONF = float(os.environ.get("SMARTROOM_DETECT_OBJ_CONF", "0.35"))

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


def class_filter(names):
    """The `classes=` argument for model.predict, given the model's id->name map.
    None means no filter (every COCO class). Unknown names are reported and
    skipped rather than silently narrowing the run to nothing."""
    if DETECT_CLASSES.lower() in ("all", "coco", "*", ""):
        return None
    by_name = {str(v).lower(): int(k) for k, v in (names or {}).items()}
    out, bad = [], []
    for tok in (t.strip() for t in DETECT_CLASSES.split(",")):
        if not tok:
            continue
        if tok.isdigit():
            out.append(int(tok))
        elif tok.lower() in by_name:
            out.append(by_name[tok.lower()])
        else:
            bad.append(tok)
    if bad:
        print(f"  unknown class name(s) ignored: {', '.join(bad)}", file=sys.stderr)
    return sorted(set(out)) or None


def _colour_for(cls: int):
    """A stable, distinguishable BGR per class id (annotated video only)."""
    h = (cls * 47) % 180
    import numpy as _np
    import cv2 as _cv2
    px = _np.uint8([[[h, 200, 255]]])
    b, g, r = _cv2.cvtColor(px, _cv2.COLOR_HSV2BGR)[0][0].tolist()
    return int(b), int(g), int(r)


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


def transcode_h264(src: Path, dst: Path) -> subprocess.CompletedProcess:
    """Re-encode to browser-friendly H.264. Prefer the GPU hardware encoder
    (NVENC) — the encode, not the YOLO inference, dominates wall-clock, so this
    is the difference between the GPU sitting idle and actually being used. Fall
    back to libx264 when NVENC is unavailable (no GPU / driver)."""
    src_a = ["ffmpeg", "-y", "-i", str(src)]
    tail = ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    if os.environ.get("SMARTROOM_DETECT_ENCODER", "nvenc") != "cpu":
        proc = subprocess.run(
            src_a + ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"] + tail,
            capture_output=True,
        )
        if proc.returncode == 0 and dst.exists():
            return proc
        dst.unlink(missing_ok=True)  # NVENC unavailable — fall back to CPU
    return subprocess.run(src_a + ["-c:v", "libx264"] + tail, capture_output=True)


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
    # Resolved per model, since the class ids are the model's own (they happen to be
    # COCO's for these weights, but nothing here assumes that).
    classes = None if pose else class_filter(getattr(model, "names", None))
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
    last_boxes = []          # detection: [(x1,y1,x2,y2,cls,name,conf)] for drawing
    seen_classes = {}        # class name -> {frames, maxPerFrame, total, peakConf}
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
                res = model.predict(frame, imgsz=IMGSZ, classes=classes,
                                    device=DEVICE, verbose=False)[0]
                names = res.names or {}
                dets = []
                if res.boxes is not None and len(res.boxes):
                    for box, cls, cf in zip(res.boxes.xyxy.tolist(),
                                            res.boxes.cls.tolist(),
                                            res.boxes.conf.tolist()):
                        if cf < OBJ_CONF:
                            continue
                        c = int(cls)
                        dets.append({"cls": c, "name": str(names.get(c, c)),
                                     "conf": round(float(cf), 3),
                                     "box": [int(v) for v in box]})
                last_boxes = [(d["box"][0], d["box"][1], d["box"][2], d["box"][3],
                               d["cls"], d["name"], d["conf"]) for d in dets]
                # `count` stays the PEOPLE count: it is what occupancy, the
                # dashboard's peak/avg and every existing consumer mean by it.
                people = sum(1 for d in dets if d["cls"] == PERSON_CLASS)
                row = {"t": t, "count": people}
                if dets:
                    row["objects"] = dets
                timeline.append(row)
                per_frame = {}
                for d in dets:
                    per_frame[d["name"]] = per_frame.get(d["name"], 0) + 1
                for name, n in per_frame.items():
                    e = seen_classes.setdefault(name, {"frames": 0, "maxPerFrame": 0,
                                                       "total": 0, "peakConf": 0.0})
                    e["frames"] += 1
                    e["total"] += n
                    e["maxPerFrame"] = max(e["maxPerFrame"], n)
                for d in dets:
                    e = seen_classes[d["name"]]
                    e["peakConf"] = max(e["peakConf"], d["conf"])
        if writer is not None:
            count = (len(last_persons) if pose else
                     sum(1 for b in last_boxes if b[4] == PERSON_CLASS))
            if pose:
                _draw_skeleton(frame, last_persons)
            else:
                for (x1, y1, x2, y2, cls, name, cf) in last_boxes:
                    col = (0, 200, 0) if cls == PERSON_CLASS else _colour_for(cls)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(frame, f"{name} {cf:.2f}", (x1 + 2, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
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
        proc = transcode_h264(tmp_annotated, final_tmp)
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
        # What was asked of the detector, and what it actually found. "class" is
        # kept for older readers; it says "person" only when the run really was
        # restricted to people.
        "class": "person" if classes == [PERSON_CLASS] else (DETECT_CLASSES.lower() or "all"),
        "classFilter": DETECT_CLASSES.lower() or "all",
        "objectConf": OBJ_CONF,
        "classesSeen": sorted(seen_classes),
        "objects": {k: {"frames": v["frames"], "maxPerFrame": v["maxPerFrame"],
                        "avgPerFrame": round(v["total"] / max(1, len(timeline)), 3),
                        "peakConf": round(v["peakConf"], 3)}
                    for k, v in sorted(seen_classes.items(),
                                       key=lambda kv: -kv[1]["frames"])},
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
    ap = argparse.ArgumentParser(
        description="YOLO26 object detection (all COCO classes) over saved recordings.")
    ap.add_argument("--path", action="append", metavar="REL",
                    help="clip to analyze, relative to the recordings root; repeatable for a subset")
    ap.add_argument("--force", action="store_true", help="reprocess even if results are current")
    args = ap.parse_args()

    root = saved_root()
    if not root.exists():
        print(f"no recordings dir: {root}", file=sys.stderr)
        return 0

    # Lock/pid names carry an optional suffix so multiple GPU-sharded workers can
    # run concurrently (one per GPU, disjoint clip sets) without blocking on one
    # global lock — the sharding orchestrator in run-analysis.sh sets it.
    sfx = os.environ.get("SMARTROOM_LOCK_SUFFIX", "")
    lock_file = open(root / f".detect.lock{sfx}", "w")
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
    pid_path = root / f".detect.pid{sfx}"
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
