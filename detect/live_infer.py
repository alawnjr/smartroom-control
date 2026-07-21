#!/usr/bin/env python3
"""
Live inference service (server side of the live-stream feature).

The Pi cannot be reached from the quad server (one-way network), so the Pi's
`live_forward.py` PUSHES JPEG frames here over a single persistent connection,
length-prefixed:  [4-byte big-endian uint32 length][JPEG bytes] repeated.

This process runs the same YOLO26 pose model the batch pipeline uses, localizes
every person to the shared AprilTag room frame with the monocular floor-ray
(exactly `localize.py`'s `camera_main` / depth-missing path), and serves:

  POST /ingest?cam=<stream-key>   frame sink (from the Pi forwarder)
  GET  /live.mjpg                 annotated MJPEG (skeletons + foot markers)
  GET  /positions                 latest room positions JSON + roomFrame
  GET  /                          a viewer page (video + top-down room map)

Calibration is NOT sent from the Pi. Extrinsics are static, so `geom` is built
once from the newest UPLOADED recording that contains this camera (its
metadata.json already embeds calibration + extrinsics) via
`calib_utils.load_room_geometry` — the same function the batch localizer uses.

Env:
  SMARTROOM_SAVE_DIR        recordings root (to find a clip for calibration)
  SMARTROOM_DETECT_DEVICE   torch device ("0" for GPU, "cpu"); default auto
  SMARTROOM_LIVE_WEIGHTS    pose weights (default ~/Code/yolo-bench/yolo26n-pose.pt)

Usage:
  python detect/live_infer.py --cam camera_d455_color --port 8010
"""

import argparse
import json
import os
import struct
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calib_utils import load_room_geometry  # noqa: E402
from localize import backproject_room, hip_point, joint_px  # noqa: E402

# COCO-17 shoulders (fallback anchor when the hips are occluded, e.g. seated at
# a desk). Both anchors are ranged by real depth — no floor-ray, never the feet.
L_SHOULDER, R_SHOULDER = 5, 6

BOUNDARY = "frame"
JPEG_QUALITY = 75
KP_CONF = float(os.environ.get("SMARTROOM_ROOM_KP_CONF", "0.5"))
DEPTH_MATCH_FRAC = 0.06   # a depth sample within this (fraction of frame) counts as "this hip"
DEPTH_STALE_S = 1.0       # ignore depth samples older than this
ACTION_WINDOW = 48        # skeleton-window length (mirrors action.WINDOW); deque cap
ACTION_TRACK_TTL_S = 2.0  # drop a track's window/label if unseen this long
ACTION_SWEEP_S = 0.35     # how often the action thread re-classifies live tracks
AVA_SHORT = 256           # short-side the SlowFast-AVA clip is resized to
AVA_BUF = 128             # rolling RGB frame buffer (a few seconds at any fps)
AVA_PERIOD_S = 0.4        # how often to run the (heavier) AVA forward
AVA_THR = 0.4             # per-class action score threshold
# SlowFast-AVA was trained on ~30fps clips where its 32x2 window ≈ 2.1s. Our live
# feed is ~10fps, so taking the last 64 frames would span ~6.4s — too much motion
# integrated per label (inertia) and 3x-stretched so dynamic actions look static.
# Instead pick frames from the last AVA_SPAN_S seconds (wall clock) and resample
# to clip_len, matching the training time-span regardless of the live fps.
AVA_SPAN_S = float(os.environ.get("SMARTROOM_AVA_SPAN_S", "2.1"))
AVA_MIN_FRAMES = 8        # need at least this many frames in the span to classify
# Geometric jump detector (ports action.py detect_jumps to a live streaming form,
# independent of the ML classifier). A jump = the hip center-of-mass rising above
# its rolling "standing" baseline by > JUMP_FRAC of body height. Distance-invariant.
JUMP_FRAC = float(os.environ.get("SMARTROOM_JUMP_FRAC", "0.20"))
JUMP_WINDOW_S = 1.5       # rolling baseline window
JUMP_MIN_STREAK = 2       # consecutive airborne frames before firing (anti-jitter)
JUMP_HOLD_S = 0.5         # keep showing "jump" this long after the last airborne frame

# COCO-17 skeleton edges (for drawing) + a color per limb group.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (5, 6), (11, 12), (5, 11), (6, 12),       # torso
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),  # head
]


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


def find_calib_clip(cam_key: str) -> Path | None:
    """Newest uploaded <cam_key>.mp4 whose sibling metadata.json has extrinsics."""
    root = saved_root()
    if not root.exists():
        return None
    clips = sorted(root.rglob(f"{cam_key}.mp4"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for mp4 in clips:
        if "undistorted" in mp4.parts:
            continue
        md = mp4.parent / "metadata.json"
        if not md.exists():
            continue
        try:
            streams = json.loads(md.read_text()).get("streams", {})
            entry = streams.get(mp4.stem, {})
            if entry.get("calibration") and entry.get("extrinsics"):
                return mp4
        except (OSError, ValueError):
            continue
    return None


def _resize_short(w, h, short):
    """New (w, h) with the short side scaled to `short`, aspect preserved."""
    scale = short / min(w, h)
    return int(round(w * scale)), int(round(h * scale))


def load_label_map(path):
    """AVA label map: 'id: name' per line -> {int id: name} (same as the demo)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if ": " in line:
            i, name = line.split(": ", 1)
            out[int(i)] = name.strip()
    return out


class Shared:
    """Newest-frame-wins slots shared across the ingest, inference and HTTP
    threads (mirrors realsense_depth_page.py's ViewCache pattern)."""

    def __init__(self):
        self.cond = threading.Condition()
        self.in_jpeg = None          # latest raw JPEG bytes from the Pi
        self.in_id = 0
        self.out_jpeg = None         # latest annotated JPEG
        self.out_id = 0
        self.positions = []          # [{id,x,z,src}]
        self.updated_ms = 0
        self.fps = 0.0
        # depth back-channel: the server publishes the latest hip pixels it wants
        # ranged, the Pi forwarder samples its own /value there and posts metres
        # back (D455 depth aligned to color). Both keyed in frame-fraction coords.
        self.hips = []               # [[u_frac, v_frac], ...] latest frame's hips
        self.depths = []             # [(u_frac, v_frac, metres, monotonic_t), ...]
        # temporal action classification: a rolling skeleton window per track id
        # (fed by the pose loop) and the latest label the action thread produced.
        self.windows = defaultdict(lambda: deque(maxlen=ACTION_WINDOW))
        self.win_seen = {}           # tid -> monotonic_t of last skeleton append
        self.labels = {}             # tid -> {"action", "conf", "top", "t"}
        # SlowFast-AVA: a rolling buffer of (resized BGR frame, [(tid, box_resized)])
        # — whole-frame clips + per-person proposals, classified together.
        self.ava_buf = deque(maxlen=AVA_BUF)

    def put_in(self, jpeg):
        with self.cond:
            self.in_jpeg = jpeg
            self.in_id += 1
            self.cond.notify_all()

    def set_hips(self, hips):
        with self.cond:
            self.hips = hips

    def get_hips(self):
        with self.cond:
            return list(self.hips)

    def put_depths(self, samples):
        now = time.monotonic()
        with self.cond:
            self.depths = [(s[0], s[1], s[2], now) for s in samples]

    def push_skeleton(self, tid, kpts, conf):
        """Append one (kpts(17,2), conf(17)) sample to a track's rolling window."""
        now = time.monotonic()
        with self.cond:
            self.windows[tid].append((kpts, conf))
            self.win_seen[tid] = now

    def snapshot_windows(self):
        """{tid: list-of-samples} for tracks seen recently; prunes stale ones."""
        now = time.monotonic()
        out = {}
        with self.cond:
            stale = [t for t, s in self.win_seen.items() if now - s > ACTION_TRACK_TTL_S]
            for t in stale:
                self.windows.pop(t, None)
                self.win_seen.pop(t, None)
                self.labels.pop(t, None)
            for t, dq in self.windows.items():
                out[t] = list(dq)
        return out

    def set_label(self, tid, action, conf, top):
        with self.cond:
            self.labels[tid] = {"action": action, "conf": round(float(conf), 3),
                                "top": top, "t": time.monotonic()}

    def get_label(self, tid):
        with self.cond:
            return self.labels.get(tid)

    def push_ava(self, frame_bgr, boxes, w, h):
        """Resize the frame to the AVA short side and scale each (tid, box) into
        those coords, then buffer it. boxes: [(tid, [x1,y1,x2,y2])] in full res."""
        nw, nh = _resize_short(w, h, AVA_SHORT)
        small = cv2.resize(frame_bgr, (nw, nh))
        rx, ry = nw / w, nh / h
        scaled = [(tid, [b[0] * rx, b[1] * ry, b[2] * rx, b[3] * ry])
                  for tid, b in boxes]
        with self.cond:
            self.ava_buf.append((small, scaled, time.monotonic()))

    def snapshot_ava(self):
        with self.cond:
            return list(self.ava_buf)

    def depth_near(self, u_frac, v_frac):
        """Freshest metric depth (mm) sampled near this hip, or None."""
        now = time.monotonic()
        best, best_d = None, DEPTH_MATCH_FRAC
        with self.cond:
            samples = list(self.depths)
        for su, sv, m, t in samples:
            if now - t > DEPTH_STALE_S or not m or m <= 0:
                continue
            d = ((su - u_frac) ** 2 + (sv - v_frac) ** 2) ** 0.5
            if d < best_d:
                best, best_d = m * 1000.0, d
        return best

    def put_out(self, jpeg, positions, fps):
        with self.cond:
            self.out_jpeg = jpeg
            self.out_id += 1
            self.positions = positions
            self.fps = fps
            self.updated_ms = int(time.time() * 1000)
            self.cond.notify_all()


def _make_bytetrack():
    """ByteTracker with ultralytics' default bytetrack.yaml params — the same
    tracker model.track() builds, so ids are stable across frames (mirrors
    action.py's _make_bytetrack). Image-space, so it dedupes overlapping person
    boxes that fragmented the old greedy room-space assigner."""
    from types import SimpleNamespace

    from ultralytics.trackers.byte_tracker import BYTETracker
    args = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1,
                           new_track_thresh=0.25, track_buffer=30,
                           match_thresh=0.8, fuse_score=True,
                           gmc_method="sparseOptFlow")
    return BYTETracker(args)


def _shoulder_point(person, w, h):
    """Mid-shoulder pixel (conf-gated), or None — fallback anchor when the hips
    are occluded. Mirrors localize.hip_point but on the shoulder joints."""
    pts = [pt for pt in (joint_px(person, L_SHOULDER, w, h),
                         joint_px(person, R_SHOULDER, w, h)) if pt]
    if not pts:
        return None
    return sum(pt[0] for pt in pts) / len(pts), sum(pt[1] for pt in pts) / len(pts)


def _hip_com(person):
    """(hip-midpoint y, body pixel height) from a person's pixel keypoints, for
    jump detection. Either may be None if too few joints are confident."""
    px, cf = person["px"], person["conf"]
    ys = [px[j][1] for j in range(len(cf)) if cf[j] >= KP_CONF]
    hips = [px[j][1] for j in (11, 12) if j < len(cf) and cf[j] >= KP_CONF]
    comy = sum(hips) / len(hips) if hips else None
    body_h = (max(ys) - min(ys)) if len(ys) >= 2 else None
    return comy, body_h


class JumpDetector:
    """Per-track streaming jump detector. Image y grows downward, so airborne =
    the hip CoM sitting ABOVE (smaller y than) its rolling-median standing
    baseline by more than JUMP_FRAC of the person's pixel height."""

    def __init__(self):
        self.hist = defaultdict(deque)   # tid -> deque of (t, comy, body_h)
        self.streak = defaultdict(int)
        self.until = {}                  # tid -> monotonic t to keep showing "jump"

    def update(self, tid, comy, body_h, t):
        dq = self.hist[tid]
        dq.append((t, comy, body_h))
        while dq and t - dq[0][0] > JUMP_WINDOW_S:
            dq.popleft()
        comys = sorted(c for _, c, _ in dq if c is not None)
        bhs = sorted(b for _, _, b in dq if b)
        if comy is not None and len(comys) >= 4 and bhs:
            baseline = comys[len(comys) // 2]          # median standing CoM
            body = bhs[len(bhs) // 2] or 1.0
            if (baseline - comy) / body >= JUMP_FRAC:   # CoM risen above baseline
                self.streak[tid] += 1
                if self.streak[tid] >= JUMP_MIN_STREAK:
                    self.until[tid] = t + JUMP_HOLD_S
            else:
                self.streak[tid] = 0
        return self.until.get(tid, 0.0) > t

    def prune(self, live, t):
        for tid in [k for k, dq in self.hist.items()
                    if k not in live and (not dq or t - dq[-1][0] > 3)]:
            self.hist.pop(tid, None)
            self.streak.pop(tid, None)
            self.until.pop(tid, None)


def infer_loop(shared: Shared, geom: dict, weights: str, device: str, flip: bool,
               mode: str):
    from ultralytics import YOLO
    model = YOLO(weights)
    tracker = _make_bytetrack()
    jumps = JumpDetector()
    use_half = device not in ("cpu", "intel:cpu")
    last_id = 0
    ema_fps = 0.0
    print(f"[live] model loaded ({weights}) device={device} half={use_half}", flush=True)
    while True:
        with shared.cond:
            while shared.in_id == last_id or shared.in_jpeg is None:
                shared.cond.wait(timeout=5.0)
                if shared.in_jpeg is None:
                    continue
            last_id = shared.in_id
            jpeg = shared.in_jpeg
        t0 = time.time()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        h, w = frame.shape[:2]
        try:
            res = model.predict(frame, imgsz=640, device=device, half=use_half,
                                classes=[0], verbose=False)[0].cpu()
        except Exception as exc:  # noqa: BLE001
            print(f"[live] predict error: {exc}", flush=True)
            continue

        # Image-space ByteTrack for STABLE ids (the tracker model.track() uses),
        # shared by localization and action — fixes greedy room-space
        # fragmentation and dedupes overlapping person boxes. update() rows are
        # [x1,y1,x2,y2,id,conf,cls,det_idx]; det_idx maps back to the keypoints.
        persons = []        # (tid, person dict)
        kp = res.keypoints
        if kp is not None and kp.xy is not None and res.boxes is not None:
            xy = kp.xy.numpy()
            xyn = kp.xyn.numpy()
            conf = (kp.conf.numpy() if kp.conf is not None
                    else np.ones(xy.shape[:2], "float32"))
            for row in tracker.update(res.boxes, res.orig_img):
                di = int(row[7])
                if di < 0 or di >= len(xy):
                    continue
                persons.append((int(row[4]), {
                    "kpts": xyn[di].tolist(), "conf": conf[di].tolist(),
                    "px": xy[di].tolist(),
                    "box": [float(row[0]), float(row[1]), float(row[2]), float(row[3])]}))

        # Localize each person by the D455's real depth at an upper-body anchor:
        # the mid-hip if visible, else the mid-shoulder (both survive occluded
        # feet / a desk). backproject_room needs no height assumption. Publish
        # the anchor pixels the depth back-channel should range.
        found = []          # (tid, pos_xz, marker_px, person, src)
        anchors_frac = []
        for tid, p in persons:
            anchor = hip_point(p, w, h)   # mid-hip pixels, or None if low-conf
            src = "depth-hip"
            if anchor is None:
                anchor = _shoulder_point(p, w, h)
                src = "depth-shoulder"
            if anchor is None:
                continue
            anchors_frac.append([anchor[0] / w, anchor[1] / h])
            z_mm = shared.depth_near(anchor[0] / w, anchor[1] / h)
            if not z_mm:
                continue
            p_room = backproject_room(anchor[0], anchor[1], z_mm, geom)
            if p_room is None:
                continue
            found.append((tid, (float(p_room[0]), float(p_room[2])), anchor, p, src))
        shared.set_hips(anchors_frac)

        skeleton = mode in ("ntu", "hmdb")
        ava = mode == "ava"
        positions = []
        ava_boxes = []
        for tid, pos, marker, p, src in found:
            if skeleton:
                shared.push_skeleton(tid,
                                     np.asarray(p["px"], dtype="float32"),
                                     np.asarray(p["conf"], dtype="float32"))
            if ava and p.get("box") is not None:
                ava_boxes.append((tid, p["box"]))
            lab = shared.get_label(tid) if (skeleton or ava) else None
            action = lab.get("action") if lab else None
            aconf = lab["conf"] if (lab and action) else None
            # geometric jump detector — independent of the classifier, wins when
            # the person is airborne (the ML models are weak on brief jumps).
            comy, body_h = _hip_com(p)
            if jumps.update(tid, comy, body_h, t0):
                action, aconf = "jump", 1.0
            entry = {"id": tid, "x": round(pos[0], 1), "z": round(pos[1], 1), "src": src}
            if action:
                entry["action"] = action
                entry["actionConf"] = aconf
            positions.append(entry)
            _draw_person(frame, p["px"], p["conf"], marker, tid, src, action)
        jumps.prune({tid for tid, *_ in found}, t0)
        if ava:
            shared.push_ava(frame, ava_boxes, w, h)

        dt = time.time() - t0
        ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
        cv2.putText(frame, f"{len(positions)} person(s)  {ema_fps:4.1f} fps",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ok, enc = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            shared.put_out(enc.tobytes(), positions, round(ema_fps, 1))


def action_loop(shared: Shared, width: int, height: int, variant_key: str):
    """Temporal action classification. Reuses action.py's mmaction recognizer +
    label maps + thresholds, run on each live track's trailing skeleton window
    (front-padded until full). Runs in its own thread so it never slows pose."""
    import torch
    import action as A
    from mmaction.apis import inference_skeleton, init_recognizer

    variant = A.VARIANTS[variant_key]
    class_names = variant["labels"]
    temp = A.variant_temp(variant)
    min_conf = A.variant_min_conf(variant, len(class_names))
    disabled = A.load_disabled(variant["key"], class_names)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_recognizer(A.variant_config(variant), A.variant_ckpt(variant),
                            device=device)
    print(f"[live] action '{variant_key}' loaded: {len(class_names)} classes, "
          f"WINDOW={A.WINDOW} MIN={A.MIN_WINDOW} min_conf={min_conf:.3f} device={device}",
          flush=True)

    while True:
        for tid, win in shared.snapshot_windows().items():
            if len(win) < A.MIN_WINDOW:
                continue
            win = ([win[0]] * (A.WINDOW - len(win)) + win) if len(win) < A.WINDOW \
                else win[-A.WINDOW:]
            pose_results = [{"keypoints": kp[None].astype("float32"),
                             "keypoint_scores": sc[None].astype("float32")}
                            for kp, sc in win]
            try:
                res = inference_skeleton(model, pose_results, (height, width))
            except Exception as exc:  # noqa: BLE001
                print(f"[live] action infer error: {exc}", flush=True)
                continue
            probs = (res.pred_score.clamp_min(1e-8).log() / temp).softmax(-1)
            if disabled:
                probs = probs.clone()
                probs[disabled] = 0.0
            k = min(A.TOPK, int(probs.numel()))
            vals, idxs = probs.topk(k)
            vals = [float(v) for v in vals.tolist()]
            idxs = [int(i) for i in idxs.tolist()]
            nm = lambda i: class_names[i] if i < len(class_names) else str(i)  # noqa: E731
            top = [[nm(i), round(v, 3)] for v, i in zip(vals, idxs)]
            c, i = vals[0], idxs[0]
            shared.set_label(tid, nm(i) if c >= min_conf else None, c, top)
        time.sleep(ACTION_SWEEP_S)


def ava_loop(shared: Shared, config_path: str, ckpt: str, label_map_path: str,
             device: str, action_thr: float):
    """SlowFast-AVA spatiotemporal detection. Reuses mmaction2's official demo
    inference recipe: build the detection model, then per prediction step take
    the trailing RGB clip + current person boxes as proposals and read per-box
    multi-label action scores. One forward classifies everyone in the frame."""
    import mmcv
    import mmengine
    import numpy as np
    import torch
    from mmengine.runner import load_checkpoint
    from mmengine.structures import InstanceData
    from mmaction.registry import MODELS
    from mmaction.structures import ActionDataSample
    try:
        from mmaction.utils import register_all_modules
        register_all_modules(True)
    except Exception:  # noqa: BLE001
        pass

    cfg = mmengine.Config.fromfile(config_path)
    # equal bbox count across classes (demo does this); handle test_cfg.rcnn None
    tc = cfg.model.get("test_cfg") or {}
    tc["rcnn"] = dict(action_thr=0)
    cfg.model["test_cfg"] = tc
    cfg.model.backbone.pretrained = None
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt, map_location="cpu")
    model.to(device).eval()

    sampler = [x for x in cfg.val_pipeline
               if str(x["type"]).endswith("SampleAVAFrames")][0]
    clip_len, interval = sampler["clip_len"], sampler["frame_interval"]
    mean = np.array(cfg.model.data_preprocessor["mean"])
    std = np.array(cfg.model.data_preprocessor["std"])
    label_map = load_label_map(label_map_path)
    print(f"[live] AVA model loaded: {len(label_map)} classes, clip_len={clip_len} "
          f"span={AVA_SPAN_S}s thr={action_thr} device={device}", flush=True)

    while True:
        time.sleep(AVA_PERIOD_S)
        buf = shared.snapshot_ava()
        if len(buf) < AVA_MIN_FRAMES:
            continue
        # frames from the last AVA_SPAN_S seconds, resampled to clip_len so the
        # clip covers the training time-span regardless of the live fps.
        t_now = buf[-1][2]
        seg = [e for e in buf if t_now - e[2] <= AVA_SPAN_S]
        if len(seg) < AVA_MIN_FRAMES:
            continue
        _, boxes, _ = seg[-1]              # proposals = the newest frame's people
        if not boxes:
            continue
        nh, nw = seg[-1][0].shape[:2]
        idx = np.linspace(0, len(seg) - 1, clip_len).round().astype(int)
        imgs = [seg[i][0].astype(np.float32) for i in idx]
        for im in imgs:
            mmcv.imnormalize_(im, mean, std, to_rgb=False)
        arr = np.stack(imgs).transpose(3, 0, 1, 2)[np.newaxis]   # 1,C,T,H,W
        inp = torch.from_numpy(arr).to(device)
        tids = [t for t, _ in boxes]
        prop = torch.tensor([b for _, b in boxes], dtype=torch.float32, device=device)
        ds = ActionDataSample()
        ds.proposals = InstanceData(bboxes=prop)
        ds.set_metainfo(dict(img_shape=(nh, nw)))
        try:
            with torch.no_grad():
                res = model(inp, [ds], mode="predict")
            scores = res[0].pred_instances.scores    # (num_props, num_classes)
        except Exception as exc:  # noqa: BLE001
            print(f"[live] AVA infer error: {exc}", flush=True)
            continue
        for j, tid in enumerate(tids):
            labs = [(label_map[i], float(scores[j, i]))
                    for i in range(scores.shape[1])
                    if i in label_map and float(scores[j, i]) > action_thr]
            labs.sort(key=lambda x: -x[1])
            if labs:
                shared.set_label(tid, labs[0][0], labs[0][1],
                                 [[a, round(s, 3)] for a, s in labs[:6]])
            else:
                shared.set_label(tid, None, 0.0, [])


def _draw_person(frame, px, conf, marker, tid, src, action=None):
    color = _track_color(tid)
    for a, b in SKELETON:
        if a < len(conf) and b < len(conf) and conf[a] > KP_CONF and conf[b] > KP_CONF:
            pa = (int(px[a][0]), int(px[a][1]))
            pb = (int(px[b][0]), int(px[b][1]))
            cv2.line(frame, pa, pb, color, 2)
    for j in range(len(conf)):
        if conf[j] > KP_CONF:
            cv2.circle(frame, (int(px[j][0]), int(px[j][1])), 3, color, -1)
    # cyan ring at the hip when depth-ranged, orange at the shoulder fallback
    mcol = (255, 255, 0) if src == "depth-hip" else (0, 165, 255)
    cv2.circle(frame, (int(marker[0]), int(marker[1])), 6, mcol, 2)
    tag = f"#{tid}" + (f" {action}" if action else "")
    cv2.putText(frame, tag, (int(marker[0]) + 8, int(marker[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _track_color(tid):
    rng = (37 * (tid + 1)) % 180
    hsv = np.uint8([[[rng, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(b), int(g), int(r))


def make_handler(shared: Shared, room_frame: dict):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/depths":
                self._recv_depths()
                return
            if path != "/ingest":
                self.send_error(404)
                return
            # length-prefixed JPEG stream over one persistent connection
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            n = 0
            try:
                while True:
                    hdr = self._readn(4)
                    if not hdr:
                        break
                    (length,) = struct.unpack(">I", hdr)
                    if length == 0 or length > 20_000_000:
                        break
                    jpeg = self._readn(length)
                    if jpeg is None:
                        break
                    shared.put_in(jpeg)
                    n += 1
            except (ConnectionError, OSError):
                pass
            print(f"[live] ingest connection closed after {n} frames", flush=True)

        def _readn(self, n):
            buf = b""
            while len(buf) < n:
                chunk = self.rfile.read(n - len(buf))
                if not chunk:
                    return None if not buf else None
                buf += chunk
            return buf

        def _recv_depths(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                samples = json.loads(self.rfile.read(length) or b"[]")
                shared.put_depths([(float(s["u"]), float(s["v"]), float(s["m"]))
                                   for s in samples])
            except (ValueError, KeyError, TypeError):
                pass
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._page()
            elif path == "/positions":
                self._positions()
            elif path == "/hips":
                self._hips()
            elif path == "/live.mjpg":
                self._stream()
            else:
                self.send_error(404)

        def _hips(self):
            body = json.dumps({"hips": shared.get_hips()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _positions(self):
            with shared.cond:
                body = json.dumps({
                    "positions": shared.positions,
                    "updatedMs": shared.updated_ms,
                    "fps": shared.fps,
                    "roomFrame": room_frame,
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-cache, no-store")
            self._cors()
            self.end_headers()
            last = 0
            try:
                while True:
                    with shared.cond:
                        while shared.out_id == last or shared.out_jpeg is None:
                            shared.cond.wait(timeout=5.0)
                        last = shared.out_id
                        frame = shared.out_jpeg
                    self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (ConnectionError, OSError):
                pass

        def _page(self):
            body = PAGE_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


PAGE_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>smartroom · live inference</title>
<style>
 body{margin:0;background:#0c0a09;color:#e7e5e4;font:14px system-ui;padding:16px}
 h1{font-size:16px;margin:0 0 12px}
 .wrap{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
 .card{background:#1c1917;border-radius:12px;padding:10px}
 img{display:block;border-radius:8px;max-width:640px;width:100%}
 canvas{background:#0a0a0a;border-radius:8px}
 .meta{font-size:12px;color:#a8a29e;margin-top:6px}
</style></head><body>
<h1>smartroom — live pose + room localization</h1>
<div class=wrap>
 <div class=card><div>Camera</div><img id=v src="/live.mjpg"><div class=meta id=fps></div></div>
 <div class=card><div>Top-down room map (mm)</div>
   <canvas id=map width=420 height=420></canvas>
   <div class=meta id=cnt></div></div>
</div>
<script>
const cv=document.getElementById('map'),ctx=cv.getContext('2d');
let room=null;
function draw(pos){
  const W=cv.width,H=cv.height,pad=30;
  ctx.clearRect(0,0,W,H);
  // room frame: X (right) horizontal, Z (out of wall) vertical (0 at wall, grows toward viewer)
  const R=4500; // mm half-extent shown
  function tx(x){return pad+(x+R)/(2*R)*(W-2*pad);}
  function tz(z){return H-pad-(z)/(R)*(H-2*pad);} // z 0..R from top wall down
  ctx.strokeStyle='#44403c';ctx.strokeRect(pad,pad,W-2*pad,H-2*pad);
  ctx.fillStyle='#57534e';ctx.font='11px system-ui';
  ctx.fillText('wall / tag (z=0)',pad,pad-8);
  // camera marker
  if(room&&room.cameraPositionMm){const c=room.cameraPositionMm;
    ctx.fillStyle='#0ea5e9';ctx.beginPath();ctx.arc(tx(c[0]),tz(c[2]),5,0,7);ctx.fill();
    ctx.fillText('cam',tx(c[0])+7,tz(c[2]));}
  for(const p of pos){
    // both anchors are depth-ranged: amber = hip, sky = shoulder fallback
    const col=(p.src==='depth-shoulder')?'#38bdf8':'#f59e0b';
    ctx.fillStyle=col;ctx.beginPath();ctx.arc(tx(p.x),tz(p.z),8,0,7);ctx.fill();
    ctx.fillStyle='#0c0a09';ctx.fillText('#'+p.id,tx(p.x)-6,tz(p.z)+4);
    if(p.action){ctx.fillStyle='#fde68a';ctx.fillText(p.action,tx(p.x)+11,tz(p.z)+4);}
  }
  ctx.fillStyle='#f59e0b';ctx.fillText('● depth-hip',pad,H-8);
  ctx.fillStyle='#38bdf8';ctx.fillText('● depth-shoulder',pad+90,H-8);
}
async function poll(){
  try{const r=await fetch('/positions');const d=await r.json();
    const pos=d.positions||[];room=d.roomFrame;draw(pos);
    document.getElementById('fps').textContent='inference '+(d.fps||0)+' fps';
    const acts=pos.map(p=>'#'+p.id+': '+(p.action?p.action+' ('+(p.actionConf||0).toFixed(2)+')':'…'));
    document.getElementById('cnt').innerHTML=pos.length+' person(s) · '+
      (acts.length?acts.join(' · '):'—');
  }catch(e){}
  setTimeout(poll,200);
}
poll();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cam", default="camera_d455_color",
                    help="stream key to localize (finds calibration in recordings)")
    ap.add_argument("--clip", help="explicit recording mp4 for calibration (optional)")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--flip", action="store_true",
                    help="rotate incoming frames 180 (if the Pi serves unrotated)")
    ap.add_argument("--action", default="ava",
                    help="action model: ava (SlowFast-AVA, per-person RGB) | "
                         "ntu | hmdb (skeleton) | off")
    args = ap.parse_args()
    mode = args.action.lower()
    if mode in ("none", "no", "0", ""):
        mode = "off"

    weights = os.environ.get("SMARTROOM_LIVE_WEIGHTS") or str(
        Path.home() / "Code/yolo-bench/yolo26n-pose.pt")
    device = os.environ.get("SMARTROOM_DETECT_DEVICE")
    if not device:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    clip = Path(args.clip) if args.clip else find_calib_clip(args.cam)
    if clip is None or not clip.exists():
        print(f"[live] FATAL: no recording with calibration for '{args.cam}' "
              f"under {saved_root()}", file=sys.stderr)
        return 2
    geom = load_room_geometry(clip, args.width, args.height, undistorted=False)
    if geom is None:
        print(f"[live] FATAL: {clip} has no room geometry (extrinsics)",
              file=sys.stderr)
        return 2
    room_frame = {
        "cameraPositionMm": [round(float(v), 1) for v in geom["cam_pos_mm"]],
        "tagId": geom.get("tag_id"),
        "tagHeightMm": geom.get("tag_height_mm"),
        "cameraId": geom.get("camera_id"),
        "calibClip": str(clip.relative_to(saved_root())),
    }
    print(f"[live] geom from {clip}  cam_pos_mm={room_frame['cameraPositionMm']}",
          flush=True)

    shared = Shared()
    threading.Thread(target=infer_loop,
                     args=(shared, geom, weights, device, args.flip, mode),
                     daemon=True).start()
    if mode == "ava":
        import mmaction
        cfg = os.environ.get("SMARTROOM_AVA_CONFIG") or os.path.join(
            os.path.dirname(mmaction.__file__), ".mim", "configs", "detection",
            "slowfast", "slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.py")
        ckpt = os.environ.get("SMARTROOM_AVA_CKPT") or str(
            Path.home() / "Code/yolo-bench/slowfast_ava.pth")
        lm = os.environ.get("SMARTROOM_AVA_LABELS") or str(
            Path(__file__).resolve().parent / "ava_label_map.txt")
        adev = "cuda:0" if device not in ("cpu", "intel:cpu") else "cpu"
        threading.Thread(target=ava_loop,
                         args=(shared, cfg, ckpt, lm, adev, AVA_THR),
                         daemon=True).start()
    elif mode in ("ntu", "hmdb"):
        threading.Thread(target=action_loop,
                         args=(shared, args.width, args.height, mode),
                         daemon=True).start()

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port),
                                make_handler(shared, room_frame))
    print(f"[live] serving on :{args.port}  (action={mode})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
