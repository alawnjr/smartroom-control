#!/usr/bin/env python3
"""
Per-person action recognition over the saved recordings.

YOLO26-pose with tracking gives a stable id + bbox per person each frame; we
keep a per-track-id sliding window of body crops and feed each window to a
pretrained Kinetics-400 video classifier (torchvision r2plus1d_18, CPU), then
overlay the predicted action label on each tracked person.

Per clip it writes, next to camera_main.mp4:
  camera_main.annotated.action.mp4   per-person boxes + id + action label (H.264)
  camera_main.detections.action.json summary (dashboard: tracks + per-track action)
  camera_main.actions.action.json    per-track action timeline

Idempotent (skips current results), flock-guarded (.action.lock), cancellable
(writes .action.pid, becomes a process-group leader). Heavier than detection, so
it's on-demand (/api/action), not in the auto-run timer.

Config (env): SMARTROOM_SAVE_DIR, SMARTROOM_YOLO_DIR (yolo26n-pose.pt),
SMARTROOM_ACTION_WINDOW (16), SMARTROOM_ACTION_STRIDE (2).

Usage: python action.py [--path <rel>] [--force]
"""

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOW = int(os.environ.get("SMARTROOM_ACTION_WINDOW", "16"))
STRIDE = int(os.environ.get("SMARTROOM_ACTION_STRIDE", "2"))
CLASSIFY_EVERY = 8
SCHEMA_VERSION = 1


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


def pose_weights() -> Path:
    base = Path(os.environ.get("SMARTROOM_YOLO_DIR") or (Path.home() / "Code" / "yolo-bench"))
    return base / "yolo26n-pose.pt"


def sidecars(mp4: Path):
    s = mp4.stem
    return (mp4.with_name(f"{s}.detections.action.json"),
            mp4.with_name(f"{s}.actions.action.json"),
            mp4.with_name(f"{s}.annotated.action.mp4"))


def _atomic_write_json(path: Path, data: dict):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def needs_action(mp4: Path, force: bool) -> bool:
    if force:
        return True
    json_path, _, annotated = sidecars(mp4)
    if not json_path.exists() or not annotated.exists():
        return True
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return True
    if data.get("status") != "done":
        return True
    return data.get("sourceMtimeMs", 0) + 2000 < mp4.stat().st_mtime * 1000


def process_clip(net, preprocess, categories, pose, mp4: Path):
    import cv2
    import numpy as np
    import torch

    json_path, actions_path, annotated_path = sidecars(mp4)
    source_mtime_ms = mp4.stat().st_mtime * 1000
    _atomic_write_json(json_path, {"schemaVersion": SCHEMA_VERSION, "status": "analyzing",
                                   "model": "action", "source": mp4.name, "sourceMtimeMs": source_mtime_ms})

    @torch.inference_mode()
    def classify(crops):
        frames = []
        for c in crops:
            c = cv2.resize(c, (128, 128))  # common size so frames stack; transform crops to 112
            frames.append(torch.from_numpy(np.ascontiguousarray(c[:, :, ::-1])).permute(2, 0, 1))
        clip = preprocess(torch.stack(frames)).unsqueeze(0)  # 1,C,T,H,W
        probs = net(clip).softmax(-1)[0]
        conf, idx = probs.max(0)
        return categories[int(idx)], float(conf)

    cap = cv2.VideoCapture(str(mp4))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    tmp_raw = annotated_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"), native_fps, (width, height))

    tubes = defaultdict(lambda: deque(maxlen=WINDOW))
    labels = {}
    timeline = defaultdict(list)
    seen = []
    idx = 0
    for r in pose.track(str(mp4), stream=True, persist=True, classes=[0], device="cpu", verbose=False):
        frame = r.orig_img.copy()
        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            for tid, (x1, y1, x2, y2) in zip(boxes.id.int().tolist(), boxes.xyxy.int().tolist()):
                x1, y1 = max(0, x1), max(0, y1)
                if idx % STRIDE == 0:
                    crop = r.orig_img[y1:y2, x1:x2]
                    if crop.size:
                        tubes[tid].append(crop)
                if len(tubes[tid]) >= WINDOW and idx % CLASSIFY_EVERY == 0:
                    label, conf = classify(list(tubes[tid]))
                    labels[tid] = (label, conf)
                    timeline[tid].append({"t": round(idx / native_fps, 3), "action": label, "conf": round(conf, 3)})
                    if label not in seen:
                        seen.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                lab = labels.get(tid)
                text = f"#{tid} {lab[0]}" if lab else f"#{tid} ..."
                cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + 9 * len(text), y1), (0, 200, 0), -1)
                cv2.putText(frame, text, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        writer.write(frame)
        idx += 1
    writer.release()

    has_annotated = False
    if tmp_raw.exists():
        final_tmp = annotated_path.with_suffix(".enc.mp4")
        proc = subprocess.run(["ffmpeg", "-y", "-i", str(tmp_raw), "-c:v", "libx264",
                               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_tmp)],
                              capture_output=True)
        tmp_raw.unlink(missing_ok=True)
        if proc.returncode == 0 and final_tmp.exists():
            os.replace(final_tmp, annotated_path)
            has_annotated = True
        else:
            final_tmp.unlink(missing_ok=True)

    track_actions = {}
    for tid, tl in timeline.items():
        counts = {}
        for p in tl:
            counts[p["action"]] = counts.get(p["action"], 0) + 1
        if counts:
            track_actions[str(tid)] = max(counts, key=counts.get)

    _atomic_write_json(actions_path, {"schemaVersion": SCHEMA_VERSION, "model": "action",
                                      "source": mp4.name, "sourceMtimeMs": source_mtime_ms,
                                      "nativeFps": round(native_fps, 3), "window": WINDOW, "stride": STRIDE,
                                      "tracks": {str(t): timeline[t] for t in timeline}})
    _atomic_write_json(json_path, {
        "schemaVersion": SCHEMA_VERSION, "status": "done", "error": None,
        "model": "action", "source": mp4.name, "sourceMtimeMs": source_mtime_ms,
        "device": "cpu", "classifier": "r2plus1d_18_kinetics400",
        "analyzedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "durationSec": round(total / native_fps, 3) if total else None,
        "tracks": len(timeline), "trackActions": track_actions, "actions": seen,
        "annotated": annotated_path.name if has_annotated else None, "hasAnnotated": has_annotated,
    })
    print(f"  action done: {mp4.relative_to(saved_root())} tracks={len(timeline)} {seen}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Per-person action recognition over saved clips.")
    ap.add_argument("--path", help="single clip, relative to the recordings root")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = saved_root()
    if not root.exists():
        print(f"no recordings dir: {root}", file=sys.stderr)
        return 0

    lock_file = open(root / ".action.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another action run is in progress; exiting", file=sys.stderr)
        return 0
    try:
        os.setpgrp()
    except OSError:
        pass
    pid_path = root / ".action.pid"
    try:
        pid_path.write_text(str(os.getpid()))
    except OSError:
        pass

    try:
        clips = ([root / args.path] if args.path
                 else sorted(root.rglob("camera_main.mp4"), key=lambda p: p.stat().st_mtime, reverse=True))
        clips = [c for c in clips if c.exists()]
        todo = [c for c in clips if needs_action(c, args.force)]
        print(f"action: {len(todo)}/{len(clips)} clip(s) to process", file=sys.stderr)
        if not todo:
            return 0

        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
        from ultralytics import YOLO
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        categories = weights.meta["categories"]
        preprocess = weights.transforms()
        net = r2plus1d_18(weights=weights).eval()
        pose = YOLO(str(pose_weights()))

        for mp4 in todo:
            try:
                print(f"action: processing {mp4.relative_to(root)}", file=sys.stderr)
                process_clip(net, preprocess, categories, pose, mp4)
            except Exception as error:  # noqa: BLE001
                print(f"  action error: {error}", file=sys.stderr)
                jp, _, _ = sidecars(mp4)
                try:
                    _atomic_write_json(jp, {"schemaVersion": SCHEMA_VERSION, "status": "error",
                                            "model": "action", "error": str(error), "source": mp4.name,
                                            "sourceMtimeMs": mp4.stat().st_mtime * 1000})
                except Exception:
                    pass
        return 0
    finally:
        try:
            pid_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
