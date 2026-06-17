#!/usr/bin/env python3
"""
Per-person action recognition over the saved recordings (NTU-RGB+D actions).

YOLO26-pose with tracking gives a stable id + COCO-17 skeleton per person each
frame; we keep a per-track-id sliding window of keypoints and feed each window
to a pretrained 2D ST-GCN++ trained on NTU-RGB+D 60 (mmaction2, CPU), then
overlay the predicted NTU action label on each tracked person.

Runs in the dedicated Python 3.10 venv (.venv-action) which has the
mmcv/mmaction2 stack. Per clip it writes, next to camera_main.mp4:
  camera_main.annotated.action.mp4   per-person skeleton + id + NTU action (H.264)
  camera_main.detections.action.json summary (dashboard: tracks + per-track action)
  camera_main.actions.action.json    per-track action timeline

Idempotent, flock-guarded (.action.lock), cancellable (.action.pid).

Config (env): SMARTROOM_SAVE_DIR, SMARTROOM_YOLO_DIR (yolo26n-pose.pt),
SMARTROOM_STGCN_CONFIG, SMARTROOM_STGCN_CKPT, SMARTROOM_ACTION_WINDOW (48),
SMARTROOM_ACTION_STRIDE (2).
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
WINDOW = int(os.environ.get("SMARTROOM_ACTION_WINDOW", "48"))
STRIDE = int(os.environ.get("SMARTROOM_ACTION_STRIDE", "2"))
CLASSIFY_EVERY = 12
SCHEMA_VERSION = 2

# NTU-RGB+D 60 action labels, index-aligned with the model's classes (A1..A60).
NTU60 = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw",
    "sit down", "stand up", "clapping", "reading", "writing", "tear up paper", "put on jacket",
    "take off jacket", "put on a shoe", "take off a shoe", "put on glasses", "take off glasses",
    "put on a hat", "take off a hat", "cheer up", "hand waving", "kick something",
    "reach into pocket", "hopping", "jump up", "phone call", "play with phone", "type on keyboard",
    "point to something", "take a selfie", "check time", "rub two hands", "nod head/bow",
    "shake head", "wipe face", "salute", "put palms together", "cross hands in front",
    "sneeze/cough", "staggering", "falling down", "headache", "chest pain", "back pain",
    "neck pain", "nausea/vomiting", "fan self", "punch/slap", "kicking", "pushing",
    "pat on back", "point finger", "hugging", "give object", "touch pocket", "handshake",
    "walk towards", "walk apart",
]

COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


def pose_weights() -> Path:
    base = Path(os.environ.get("SMARTROOM_YOLO_DIR") or (Path.home() / "Code" / "yolo-bench"))
    return base / "yolo26n-pose.pt"


def stgcn_config() -> str:
    if os.environ.get("SMARTROOM_STGCN_CONFIG"):
        return os.environ["SMARTROOM_STGCN_CONFIG"]
    import mmaction
    return os.path.join(os.path.dirname(mmaction.__file__), ".mim", "configs", "skeleton",
                        "stgcnpp", "stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py")


def stgcn_ckpt() -> str:
    return os.environ.get("SMARTROOM_STGCN_CKPT") or str(
        Path.home() / "Code" / "yolo-bench" / "stgcnpp_ntu60_2d.pth")


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


def process_clip(model, infer, pose, mp4: Path):
    import cv2
    import numpy as np

    json_path, actions_path, annotated_path = sidecars(mp4)
    source_mtime_ms = mp4.stat().st_mtime * 1000
    _atomic_write_json(json_path, {"schemaVersion": SCHEMA_VERSION, "status": "analyzing",
                                   "model": "action", "source": mp4.name, "sourceMtimeMs": source_mtime_ms})

    cap = cv2.VideoCapture(str(mp4))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    def classify(window):
        # window: list of (kpts (17,2) float, conf (17,) float) -> NTU label, conf
        pose_results = [{"keypoints": kp[None].astype("float32"),
                         "keypoint_scores": sc[None].astype("float32")} for kp, sc in window]
        res = infer(model, pose_results, (height, width))
        score = res.pred_score
        probs = score.softmax(-1) if hasattr(score, "softmax") else score
        c, i = float(probs.max()), int(probs.argmax())
        return (NTU60[i] if i < len(NTU60) else str(i)), c

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
        kpts = r.keypoints
        if boxes is not None and boxes.id is not None and kpts is not None:
            ids = boxes.id.int().tolist()
            xyxy = boxes.xyxy.int().tolist()
            xy = kpts.xy.cpu().numpy()
            conf = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones(xy.shape[:2], "float32")
            for n, tid in enumerate(ids):
                if idx % STRIDE == 0:
                    tubes[tid].append((xy[n], conf[n]))
                if len(tubes[tid]) >= WINDOW and idx % CLASSIFY_EVERY == 0:
                    label, cf = classify(list(tubes[tid]))
                    labels[tid] = (label, cf)
                    timeline[tid].append({"t": round(idx / native_fps, 3), "action": label, "conf": round(cf, 3)})
                    if label not in seen:
                        seen.append(label)
                # draw skeleton + label
                pts, cs = xy[n], conf[n]
                for a, b in COCO_SKELETON:
                    if cs[a] > 0.3 and cs[b] > 0.3:
                        cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), (0, 200, 0), 2)
                x1, y1 = max(0, xyxy[n][0]), max(0, xyxy[n][1])
                lab = labels.get(tid)
                text = f"#{tid} {lab[0]}" if lab else f"#{tid} ..."
                cv2.rectangle(frame, (x1, max(0, y1 - 18)), (x1 + 8 * len(text), y1), (0, 200, 0), -1)
                cv2.putText(frame, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
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
        "device": "cpu", "classifier": "stgcnpp_ntu60_2d",
        "analyzedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "durationSec": round(total / native_fps, 3) if total else None,
        "tracks": len(timeline), "trackActions": track_actions, "actions": seen,
        "annotated": annotated_path.name if has_annotated else None, "hasAnnotated": has_annotated,
    })
    print(f"  action done: {mp4.relative_to(saved_root())} tracks={len(timeline)} {seen}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Per-person NTU action recognition over saved clips.")
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

        from mmaction.apis import inference_skeleton, init_recognizer
        from ultralytics import YOLO
        model = init_recognizer(stgcn_config(), stgcn_ckpt(), device="cpu")
        pose = YOLO(str(pose_weights()))

        for mp4 in todo:
            try:
                print(f"action: processing {mp4.relative_to(root)}", file=sys.stderr)
                process_clip(model, inference_skeleton, pose, mp4)
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
