#!/usr/bin/env python3
"""
Per-person action recognition over the saved recordings.

YOLO26-pose with tracking gives a stable id + COCO-17 skeleton per person each
frame; we keep a per-track-id sliding window of keypoints and feed each window
to a pretrained skeleton recognizer (mmaction2, CPU), then overlay the predicted
action label on each tracked person. Selectable via --variant:
  ntu  (default) — 2D ST-GCN++ on NTU-RGB+D 60  -> sidecar key "action"
  hmdb           — PoseC3D on HMDB51 (adds walk/run) -> sidecar key "action-hmdb"
  ava            — SlowFast-AVA 2.1 -> sidecar key "action-ava"
The first two are skeleton models driven one person at a time. `ava` is the same
model the live service runs (see ava_model.py): it classifies RGB PIXELS plus
each person's box rather than keypoints, so it sees carried objects, posture and
scene context the skeleton models are blind to, and it is multi-label — several
simultaneous actions per person. Poses are still tracked, stored and drawn in
every variant; under `ava` they simply don't drive the label. All are
multi-person.

Runs in the dedicated Python 3.10 venv (.venv-action) which has the
mmcv/mmaction2 stack. Per clip it writes, next to camera_main.mp4:
  camera_main.annotated.action.mp4   per-person skeleton + id + NTU action (H.264)
  camera_main.detections.action.json summary (dashboard: tracks + per-track action)
  camera_main.actions.action.json    per-track action timeline

Idempotent, flock-guarded (.action.lock), cancellable (.action.pid).

Config (env): SMARTROOM_SAVE_DIR, SMARTROOM_YOLO_DIR (yolo26n-pose.pt),
SMARTROOM_STGCN_CONFIG, SMARTROOM_STGCN_CKPT, SMARTROOM_ACTION_WINDOW (48),
SMARTROOM_ACTION_STRIDE (2), SMARTROOM_ACTION_OFFSET_FRAC (0.35),
SMARTROOM_ACTION_TEMP (softmax temperature; >1 flattens),
SMARTROOM_ACTION_MIN_CONF (absolute abstention threshold; overrides the
per-variant k×chance default — below it a window is labelled "idle").
"""

import argparse
import bisect
import datetime as dt
import fcntl
import gc
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fast pose pass (default on): the classifier just needs, per frame, each
# person's tracked id + COCO-17 keypoints. The original path used
# ultralytics `.track()`, which runs YOLO pose ONE FRAME AT A TIME (poor GPU
# batching — the Quadro sat ~idle) and is the dominant analysis cost. Instead run
# BATCHED YOLO pose (multi-frame forward passes) and drive the SAME ByteTrack
# tracker on the results, so track ids are identical to `.track()` — no accuracy
# change — at ~2x the speed. Measured on the quad server: 7.5s -> ~4s per 30s
# clip, single track preserved on the calibration clip. SMARTROOM_ACTION_FAST=0
# restores the streaming path (also used automatically for RTMPose, whose
# keypoints are swapped in per frame off the tracked boxes).
FAST_ACTION = os.environ.get("SMARTROOM_ACTION_FAST", "1") != "0"
ACTION_BATCH = int(os.environ.get("SMARTROOM_ACTION_BATCH", "16"))
IMGSZ_ACTION = int(os.environ.get("SMARTROOM_ACTION_IMGSZ", "640"))
# Produce the burned-in annotated action video? Off skips pass 2 entirely (a full
# re-decode + draw + encode per clip). Shares the detect flag so the whole
# pipeline is consistent; the 3D view / API don't need it.
ANNOTATE = os.environ.get("SMARTROOM_DETECT_ANNOTATE", "1") != "0"
# Half precision (fp16) for the YOLO pose forward pass — ~1.5-2x on GPU with
# negligible keypoint change; ignored on CPU. Off via SMARTROOM_ACTION_HALF=0.
ACTION_HALF = os.environ.get("SMARTROOM_ACTION_HALF", "1") != "0"


def _make_bytetrack():
    """A ByteTracker with ultralytics' default bytetrack.yaml parameters — the
    same tracker `model.track()` builds, so ids match the streaming path."""
    from types import SimpleNamespace

    from ultralytics.trackers.byte_tracker import BYTETracker
    args = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1,
                           new_track_thresh=0.25, track_buffer=30,
                           match_thresh=0.8, fuse_score=True,
                           gmc_method="sparseOptFlow")
    return BYTETracker(args)


def _tracked_frames_fast(pose, src, device):
    """Yield, per video frame, a list of (tid, xyxy, kpts_xy(17,2), conf(17,)) via
    batched YOLO pose + ByteTrack. `out` rows are [x1,y1,x2,y2,id,conf,cls,det_idx];
    det_idx maps a track back to its detection's keypoints in that frame."""
    import cv2
    import numpy as np

    tracker = _make_bytetrack()
    cap = cv2.VideoCapture(str(src))
    frames_out = []
    batch = []

    def flush():
        if not batch:
            return
        results = pose.predict(batch, imgsz=IMGSZ_ACTION, device=device,
                               half=(ACTION_HALF and str(device) != "cpu"),
                               classes=[0], verbose=False)
        for r in results:
            r = r.cpu()
            kp = r.keypoints
            xy = kp.xy.numpy() if kp is not None else np.empty((0, 17, 2), "float32")
            conf = (kp.conf.numpy() if (kp is not None and kp.conf is not None)
                    else np.ones(xy.shape[:2], "float32"))
            out = tracker.update(r.boxes, r.orig_img)
            dets = []
            for row in out:
                di = int(row[7])
                if di < 0 or di >= len(xy):
                    continue
                dets.append((int(row[4]), [int(row[0]), int(row[1]), int(row[2]), int(row[3])],
                             xy[di], conf[di]))
            frames_out.append(dets)
        batch.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        batch.append(frame)
        if len(batch) >= ACTION_BATCH:
            flush()
    flush()
    cap.release()
    return frames_out


def _transcode_h264(src: Path, dst: Path) -> subprocess.CompletedProcess:
    """Re-encode to browser-friendly H.264 on the GPU (NVENC) — the annotated-
    video encode dominates wall-clock, so keep it off the CPU. Fall back to
    libx264 when NVENC is unavailable (no GPU / driver)."""
    src_a = ["ffmpeg", "-y", "-i", str(src)]
    tail = ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    if os.environ.get("SMARTROOM_DETECT_ENCODER", "nvenc") != "cpu":
        proc = subprocess.run(
            src_a + ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"] + tail,
            capture_output=True,
        )
        if proc.returncode == 0 and dst.exists():
            return proc
        dst.unlink(missing_ok=True)
    return subprocess.run(src_a + ["-c:v", "libx264"] + tail, capture_output=True)


# WINDOW samples (taken every STRIDE native frames) form the trailing skeleton
# window fed to the classifier; it spans WINDOW*STRIDE/fps ≈ 3.2s. A label from a
# trailing window describes the motion at the window's *center*, so the annotator
# shifts each label back by half a window to align it with the motion that
# produced it (safe because this is offline — see process_clip pass 2).
WINDOW = int(os.environ.get("SMARTROOM_ACTION_WINDOW", "48"))
# Stride = how many native frames between the samples that fill the window. It is
# NOT a compute knob (pose runs every frame anyway); it stretches the fixed-size
# WINDOW to cover a target real-time span. A hardcoded stride only works at one
# fps: the old default of 2 was tuned for 30fps (48*2/30 ≈ 3.2s), but these
# cameras actually run ~13.5fps, where stride 2 gives a ~7s window sampled at only
# ~6.75fps — too sparse to resolve running cadence / the flight phase, and too
# long (brief actions get diluted). So derive stride from the clip's true fps to
# hit TARGET_WINDOW_SEC, clamped to >=1. Pin it with SMARTROOM_ACTION_STRIDE.
STRIDE_ENV = int(os.environ["SMARTROOM_ACTION_STRIDE"]) if os.environ.get("SMARTROOM_ACTION_STRIDE") else None
TARGET_WINDOW_SEC = float(os.environ.get("SMARTROOM_ACTION_WINDOW_SEC", "3.2"))


def pick_stride(native_fps: float) -> int:
    # Priority: env override > the dashboard's stride setting > fps-adaptive auto.
    # Auto: 48 samples spanning TARGET_WINDOW_SEC -> stride = fps*sec/WINDOW
    # (13.5fps,3.2s -> 1 ; 30fps,3.2s -> 2). load_stride_setting() reads the same
    # action-classes.json the whitelist uses; 0/absent there means "auto".
    if STRIDE_ENV:
        return max(1, STRIDE_ENV)
    cfg_stride = load_stride_setting()
    if cfg_stride:
        return max(1, cfg_stride)
    return max(1, round(native_fps * TARGET_WINDOW_SEC / WINDOW))
# Minimum samples before a track is classified. Waiting for the full WINDOW means
# no prediction for the first WINDOW*STRIDE frames (~7s at these cameras' real
# ~13.5fps), so predictions only appear halfway through a short clip. Instead we
# start classifying once a track has MIN_WINDOW samples and front-pad (repeat the
# earliest pose) up to WINDOW, so predictions begin early and just sharpen as real
# history fills in. Defaults to a quarter window.
MIN_WINDOW = int(os.environ.get("SMARTROOM_ACTION_MIN_WINDOW", str(max(2, WINDOW // 4))))
# How far back to shift each label, as a fraction of the window span. 0 = no
# shift (label sits where it was computed); a small value pulls labels slightly
# earlier toward the window's center so they line up better with the motion. 0.5
# = exact center (theoretically correct but feels early). Tunable via
# SMARTROOM_ACTION_OFFSET_FRAC.
OFFSET_FRAC = float(os.environ.get("SMARTROOM_ACTION_OFFSET_FRAC", "0.15"))
# Open-set handling. These models always argmax over all 60/51 classes, so on
# idle / out-of-distribution motion they emit a confident-looking *rare* class
# (NTU's medical labels, HMDB's sports). Two post-hoc fixes, no retraining:
#   TEMP (temperature scaling) — divides the recovered logits before softmax so
#     the confidence number is calibrated (the recognizer is overconfident).
#     T=1 is a no-op; T>1 flattens. See Guo et al. temperature scaling.
#   MIN_CONF (abstention) — below this calibrated confidence, emit no label
#     ("idle") instead of guessing, à la the "background class when all scores
#     are weak" rule in streaming action-detection work. Expressed as a MULTIPLE
#     OF CHANCE (1/num_classes), because a 60-way softmax max is naturally small
#     (and our webcam poses are out-of-distribution for these models, so output
#     sits near uniform); "k× chance" auto-adapts to 60 vs 51 classes. A raw
#     absolute override is also accepted (SMARTROOM_ACTION_MIN_CONF).
# Both are per-variant; the env vars, when set, override both variants.
IDLE = "idle"
# A person whose bounding box is flush against the frame edge is cut off, so their
# skeleton is partial and the classifier would just guess. Label such windows
# "not fully in frame" (not kept — no vote, no chip, no classifier call) instead.
# A window is partial if the majority of its frames are truncated; a frame is
# truncated if the box comes within EDGE_MARGIN px of any image border.
PARTIAL = "not fully in frame"
# Detected via keypoint visibility, NOT the bounding box: a box touching the frame
# edge is too camera-dependent (this footage frames heads near the top, so 78% of
# boxes touch a border even when the person is fully visible). When a person is
# truly cut off, the off-frame joints come back with low confidence. So a frame is
# "partial" if fewer than MIN_KEYPOINTS of the 17 COCO joints clear KP_CONF — which
# leaves desk-occluded sitting people (legs missing but torso present) alone.
KP_CONF = float(os.environ.get("SMARTROOM_ACTION_KP_CONF", "0.3"))
KP_CONF_SET = os.environ.get("SMARTROOM_ACTION_KP_CONF") is not None  # explicit override?
MIN_KEYPOINTS = int(os.environ.get("SMARTROOM_ACTION_MIN_KEYPOINTS", "10"))
# Ankle-confidence floor for metric room positions (independent of the relaxed
# RTMPose kp_conf — see the room-position block in process_clip).
ROOM_KP_CONF = float(os.environ.get("SMARTROOM_ROOM_KP_CONF", "0.5"))
# How many top classes to record per window for the live per-person bar graph.
TOPK = int(os.environ.get("SMARTROOM_ACTION_TOPK", "12"))
SCHEMA_VERSION = 2
# Per-variant class whitelist, written by the dashboard's Classes tab. A JSON map
# of variant key -> {"disabled": [class name, ...]}; disabled classes are masked
# (probability zeroed) before argmax, so the model can never emit them — it picks
# the best *allowed* class or abstains to idle. Absent file = all classes enabled.
CLASSES_CONFIG = Path(os.environ.get("SMARTROOM_ACTION_CLASSES_FILE")
                      or (PROJECT_ROOT / "action-classes.json"))


def load_disabled(key: str, class_names) -> list:
    # Indices of classes turned off for this variant (empty if no config / none off).
    try:
        cfg = json.loads(CLASSES_CONFIG.read_text())
        names = set(cfg.get(key, {}).get("disabled", []))
        return [i for i, n in enumerate(class_names) if n in names]
    except Exception:
        return []


def load_stride_setting():
    # Dashboard-set stride override under settings.stride (0/absent = auto). None if
    # no config. Read from the same file as the whitelist (see pick_stride).
    try:
        cfg = json.loads(CLASSES_CONFIG.read_text())
        s = cfg.get("settings", {}).get("stride")
        return int(s) if s else None
    except Exception:
        return None


def load_samples_per_classify():
    # Dashboard-set settings.samplesPerClassify (0/absent = use the variant default).
    try:
        cfg = json.loads(CLASSES_CONFIG.read_text())
        s = cfg.get("settings", {}).get("samplesPerClassify")
        return int(s) if s else None
    except Exception:
        return None


def load_pose_source_setting():
    # Dashboard-set settings.poseSource ("yolo"/"rtmpose"); None if absent/invalid.
    try:
        cfg = json.loads(CLASSES_CONFIG.read_text())
        s = cfg.get("settings", {}).get("poseSource")
        return s if s in ("yolo", "rtmpose") else None
    except Exception:
        return None


def resolve_pose_source() -> str:
    # Which skeleton source feeds the classifier. Priority: env > dashboard config >
    # "yolo" (the original, unchanged behaviour).
    env = os.environ.get("SMARTROOM_POSE_SOURCE")
    if env in ("yolo", "rtmpose"):
        return env
    return load_pose_source_setting() or "yolo"


def pick_classify_every(stride: int, default_ce: int) -> int:
    # How often to classify, in *frames*. We expose the knob as "new samples per
    # classify" (more intuitive than a frame count): with N samples/classify and a
    # given stride, classify_every = N * stride. N=1 -> classify on every new sample
    # (maximum overlap, heaviest). 0/absent -> the variant's frame-based default.
    env = os.environ.get("SMARTROOM_ACTION_SAMPLES_PER_CLASSIFY")
    spc = int(env) if env else load_samples_per_classify()
    if spc and spc > 0:
        return max(1, spc * stride)
    return default_ce


def detect_jumps(traj, fps, kp_conf=KP_CONF):
    # Geometric jump detector — independent of the skeleton classifier (which is
    # near-chance on brief, OOD jumps and may even skip the window via the
    # partial-frame gate). traj: list of (frame_idx, kpts(17,2), conf(17,)) for one
    # track, every frame. A jump = the center of mass (hip midpoint) rises briefly
    # ABOVE its rolling "standing" baseline by > JUMP_FRAC of the person's body
    # height, then returns. Rising above the standing baseline only happens when
    # airborne (crouch/stand stays at-or-below it), and normalizing by body height
    # in pixels makes it distance/zoom invariant. Returns frame-index intervals.
    import numpy as np
    if len(traj) < 4:
        return []
    frac = _env_float("SMARTROOM_ACTION_JUMP_FRAC") or 0.20
    max_sec = _env_float("SMARTROOM_ACTION_JUMP_MAX_SEC") or 1.2
    # A real jump's airborne phase lasts at least a beat; requiring a minimum
    # duration rejects single-frame hip-keypoint jitter spikes that briefly clear
    # the threshold (the main false-positive source).
    min_sec = _env_float("SMARTROOM_ACTION_JUMP_MIN_SEC") or 0.13
    n = len(traj)
    idxs = np.array([t[0] for t in traj])
    comy = np.full(n, np.nan)        # hip-midpoint y per frame (image coords)
    heights = []                     # body pixel height samples, for normalization
    for i, (_, kp, cf) in enumerate(traj):
        vis = cf >= kp_conf
        if vis[11] and vis[12]:
            comy[i] = (kp[11][1] + kp[12][1]) / 2.0
        elif vis[11] or vis[12]:
            comy[i] = kp[11][1] if vis[11] else kp[12][1]
        if vis.any():
            ys = kp[vis][:, 1]
            if ys.size >= 2:
                heights.append(float(ys.max() - ys.min()))
    if not heights or np.isnan(comy).all():
        return []
    body_h = float(np.median(heights)) or 1.0
    # Rolling-median baseline (~1.5s) tracks the standing CoM as the person moves
    # around; a brief jump can't shift a median, so it stands out as an excursion.
    bw = max(3, int(1.5 * fps))
    baseline = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - bw // 2), min(n, i + bw // 2 + 1)
        seg = comy[lo:hi]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            baseline[i] = np.median(seg)
    rise = (baseline - comy) / body_h         # >0 = CoM higher than its baseline
    above = np.nan_to_num(rise, nan=-1.0) >= frac
    events, i = [], 0
    maxlen, minlen = int(max_sec * fps), max(2, int(min_sec * fps))
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if minlen <= (j - i) <= maxlen:       # sustained but brief = ballistic = a jump
                events.append({"start": int(idxs[i]), "end": int(idxs[j - 1]),
                               "peak": round(float(np.nanmax(rise[i:j])), 3)})
            i = j
        else:
            i += 1
    return events


def _env_float(name):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None

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

# HMDB51 labels, index-aligned with mmaction2's hmdb51 label map (alphabetical).
HMDB51 = [
    "brush hair", "cartwheel", "catch", "chew", "clap", "climb", "climb stairs",
    "dive", "draw sword", "dribble", "drink", "eat", "fall floor", "fencing",
    "flic flac", "golf", "handstand", "hit", "hug", "jump", "kick", "kick ball",
    "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup",
    "ride bike", "ride horse", "run", "shake hands", "shoot ball", "shoot bow",
    "shoot gun", "sit", "situp", "smile", "smoke", "somersault", "stand",
    "swing baseball", "sword", "sword exercise", "talk", "throw", "turn",
    "walk", "wave",
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


# RTMPose (rtmlib) — an optional alternative skeleton source for the action
# classifiers, selectable per analysis (settings.poseSource / SMARTROOM_POSE_SOURCE).
# It's top-down: given YOLO's tracked person boxes it returns COCO-17 keypoints, so
# tracking stays YOLO's job and only the keypoint values change. Pure onnxruntime —
# no mmpose/mmcv, so it doesn't disturb the pinned action env. Models auto-download
# to ~/.cache/rtmlib on first use.
RTMPOSE_ONNX = os.environ.get(
    "SMARTROOM_RTMPOSE_ONNX",
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip")
RTMPOSE_BACKEND = os.environ.get("SMARTROOM_RTMPOSE_BACKEND", "onnxruntime")
RTMPOSE_DEVICE = os.environ.get("SMARTROOM_RTMPOSE_DEVICE", "cpu")


def _pick_device():
    """Torch device for YOLO tracking + the mmaction recognizer: CUDA when
    available, else CPU. Override with SMARTROOM_ACTION_DEVICE. (RTMPose runs
    through onnxruntime and keeps its own RTMPOSE_DEVICE knob.)"""
    dev = os.environ.get("SMARTROOM_ACTION_DEVICE", "auto")
    if dev != "auto":
        return dev
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _pick_device()


def _rtmpose_input_size():
    try:
        w, h = (int(x) for x in os.environ.get("SMARTROOM_RTMPOSE_INPUT", "192,256").split(","))
        return (w, h)
    except Exception:
        return (192, 256)


def load_rtmpose():
    from rtmlib import RTMPose  # lazy — only imported when rtmpose is selected
    return RTMPose(onnx_model=RTMPOSE_ONNX, model_input_size=_rtmpose_input_size(),
                   backend=RTMPOSE_BACKEND, device=RTMPOSE_DEVICE)


def _mm_skeleton_config(*parts) -> str:
    import mmaction
    return os.path.join(os.path.dirname(mmaction.__file__), ".mim", "configs", "skeleton", *parts)


# Selectable action models. Each is a skeleton recognizer fed one person's
# keypoint window at a time (the per-track loop below), so all support multiple
# people. `key` is the sidecar/dashboard model id; the NTU one stays "action"
# for backward compatibility.
#   ntu  — 2D ST-GCN++ on NTU-RGB+D 60 (light, office gestures, no plain "walk")
#   hmdb — PoseC3D on HMDB51 (heavier 3D-CNN; adds walk/run, more sports junk)
VARIANTS = {
    "ntu": {
        "key": "action", "labels": NTU60, "classifier": "stgcnpp_ntu60_2d",
        "classify_every": 12, "temp": 1.0, "conf_mult": 9.0,
        "config_env": "SMARTROOM_STGCN_CONFIG", "ckpt_env": "SMARTROOM_STGCN_CKPT",
        "config": lambda: _mm_skeleton_config(
            "stgcnpp", "stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"),
        "ckpt_default": str(Path.home() / "Code" / "yolo-bench" / "stgcnpp_ntu60_2d.pth"),
    },
    "hmdb": {
        "key": "action-hmdb", "labels": HMDB51, "classifier": "posec3d_hmdb51",
        "classify_every": 24,  # PoseC3D is much slower per inference; classify less often
        "temp": 1.0, "conf_mult": 9.0,
        "config_env": "SMARTROOM_HMDB_CONFIG", "ckpt_env": "SMARTROOM_HMDB_CKPT",
        "config": lambda: _mm_skeleton_config(
            "posec3d", "slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint.py"),
        "ckpt_default": str(Path.home() / "Code" / "yolo-bench" / "posec3d_hmdb51.pth"),
    },
    # ava — SlowFast-AVA spatiotemporal detection. The odd one out: it reads RGB
    # PIXELS plus each person's box, not keypoints, so it sees carried objects,
    # posture and scene context the skeleton models are blind to. Poses are still
    # tracked, stored and drawn exactly as before — they just don't drive the
    # label. Multi-label (sigmoid): every class over the threshold is reported,
    # so `top` can hold several simultaneous actions rather than a softmax rank.
    "ava": {
        "key": "action-ava", "labels": None, "classifier": "slowfast_ava21",
        "kind": "ava", "classify_every": 12, "temp": 1.0, "conf_mult": 0.0,
        "config_env": "SMARTROOM_AVA_CONFIG", "ckpt_env": "SMARTROOM_AVA_CKPT",
        "config": lambda: None, "ckpt_default": None,
    },
}

# The batch clip is a real video at its true fps, so unlike the live path we can
# take a genuine trailing window: AVA's 32x2 sampling spans ~2.1s at 30fps.
AVA_SPAN_S = float(os.environ.get("SMARTROOM_AVA_SPAN_S", "2.1"))
AVA_MIN_FRAMES = 8        # need at least this many buffered frames to classify


def variant_config(v: dict) -> str:
    return os.environ.get(v["config_env"]) or v["config"]()


def variant_ckpt(v: dict) -> str:
    return os.environ.get(v["ckpt_env"]) or v["ckpt_default"]


def variant_temp(v: dict) -> float:
    e = _env_float("SMARTROOM_ACTION_TEMP")
    return e if e is not None else v["temp"]


def variant_min_conf(v: dict, num_classes: int) -> float:
    # Absolute override wins; otherwise k× chance (1/num_classes).
    e = _env_float("SMARTROOM_ACTION_MIN_CONF")
    return e if e is not None else v["conf_mult"] / num_classes


def _true_fps(mp4: Path) -> float | None:
    # The real average frame rate, since CAP_PROP_FPS reports the nominal rate the
    # camera advertises (often ~2x the frames actually delivered) and cv2's
    # CAP_PROP_FRAME_COUNT is unreliable here. ffprobe's avg_frame_rate is exactly
    # frames/duration; parse the "num/den" fraction. None if it can't be derived.
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate", "-of",
             "default=noprint_wrappers=1:nokey=1", str(mp4)],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        num, _, den = out.partition("/")
        fps = float(num) / float(den) if den else float(num)
        if fps > 0:
            return fps
    except Exception:
        pass
    return None


def sidecars(mp4: Path, key: str):
    # All results live in-place next to the clip.
    s = mp4.stem
    return (mp4.with_name(f"{s}.detections.{key}.json"),
            mp4.with_name(f"{s}.actions.{key}.json"),
            mp4.with_name(f"{s}.annotated.{key}.mp4"))


def _atomic_write_json(path: Path, data: dict):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def write_run_stats(root: Path, kind: str, label: str, started, processed: int, errors: int, skipped: int):
    # Record the just-finished batch so the dashboard can show "last run" stats
    # (elapsed, clip count) in the sidebar. One file per kind (.last_run.<kind>.json).
    finished = dt.datetime.now(dt.timezone.utc)
    elapsed = (finished - started).total_seconds()
    try:
        _atomic_write_json(root / f".last_run.{kind}.json", {
            "kind": kind, "label": label,
            "startedAt": started.isoformat(), "finishedAt": finished.isoformat(),
            "elapsedSec": round(elapsed, 1), "processed": processed,
            "skipped": skipped, "errors": errors,
            "perClipSec": round(elapsed / processed, 1) if processed else None,
        })
    except Exception:
        pass


def needs_action(mp4: Path, force: bool, key: str) -> bool:
    if force:
        return True
    json_path, _, annotated = sidecars(mp4, key)
    if not json_path.exists() or not annotated.exists():
        return True
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return True
    if data.get("status") != "done":
        return True
    return data.get("sourceMtimeMs", 0) + 2000 < mp4.stat().st_mtime * 1000


def _ava_pass(src, framedata, boxes_by_frame, width, height, native_fps,
              classify_every, offset_frames, ev_idx, ev_lab, timeline,
              win_pose, seen):
    """Second decode pass for --variant ava: slide a trailing RGB window over
    the clip and classify every tracked person in one forward per step.

    Memory-bounded by design — only the last AVA_SPAN_S seconds are held, at the
    AVA short side — so a 3-minute segment costs tens of MB rather than the
    gigabytes a whole-clip buffer would. Fills the same ev_idx/ev_lab/timeline/
    win_pose structures the skeleton path fills, so every downstream sidecar,
    overlay and dashboard view is unchanged."""
    import cv2

    from ava_model import AvaDetector, resize_short

    det = AvaDetector(device=DEVICE)
    nw, nh = resize_short(width, height)
    rx, ry = nw / width, nh / height
    span = max(AVA_MIN_FRAMES, int(round(AVA_SPAN_S * native_fps)))
    buf = deque(maxlen=span)
    cap = cv2.VideoCapture(str(src))
    g = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(cv2.resize(frame, (nw, nh)))
        boxes = boxes_by_frame[g] if g < len(boxes_by_frame) else []
        if boxes and len(buf) >= AVA_MIN_FRAMES and g % classify_every == 0:
            scaled = [(tid, [b[0] * rx, b[1] * ry, b[2] * rx, b[3] * ry])
                      for tid, b in boxes]
            try:
                labels = det.infer(list(buf), scaled)
            except Exception as error:  # noqa: BLE001
                print(f"  ava infer error at frame {g}: {error}", file=sys.stderr)
                labels = {}
            kp_by_tid = {tid: (pts, cs) for tid, _, pts, cs in framedata[g]}
            # the label describes the middle of the trailing window, so it is
            # shifted back by the same offset the skeleton path uses
            t = max(0.0, (g - offset_frames) / native_fps)
            for tid, labs in labels.items():
                label = labs[0][0] if labs else None
                conf = labs[0][1] if labs else 0.0
                ev_idx[tid].append(g)
                ev_lab[tid].append(label or IDLE)
                timeline[tid].append({"t": round(t, 3), "action": label or IDLE,
                                      "conf": round(conf, 3),
                                      "kept": label is not None,
                                      # multi-label: EVERY class above threshold
                                      "top": labs[:TOPK]})
                pts, cs = kp_by_tid.get(tid, (None, None))
                win_pose[tid].append(
                    [[round(float(pts[j][0]), 1), round(float(pts[j][1]), 1),
                      round(float(cs[j]), 3)] for j in range(len(pts))]
                    if pts is not None else [])
                if label is not None and label not in seen:
                    seen.append(label)
        g += 1
    cap.release()


def process_clip(model, infer, pose, mp4: Path, variant: dict,
                 rtm=None, pose_source: str = "yolo"):
    import cv2
    import numpy as np

    # RTMPose confidences aren't distributed like YOLO's, so the 0.3 visibility gate
    # (used for the "not fully in frame" check, jump detection, and skeleton drawing)
    # can over-fire on RTM keypoints. Relax it for RTM runs unless the user set an
    # explicit SMARTROOM_ACTION_KP_CONF.
    kp_conf = KP_CONF
    if pose_source == "rtmpose" and not KP_CONF_SET:
        kp_conf = float(os.environ.get("SMARTROOM_RTMPOSE_KP_CONF", "0.2"))

    # AVA classifies RGB pixels, not keypoints: no skeleton window, no softmax
    # abstention, and its class list comes from the label map file.
    is_ava = variant.get("kind") == "ava"
    key, class_names = variant["key"], variant["labels"] or []
    temp = variant_temp(variant)
    min_conf = 0.0 if is_ava else variant_min_conf(variant, len(class_names))
    disabled_idx = [] if is_ava else load_disabled(key, class_names)
    json_path, actions_path, annotated_path = sidecars(mp4, key)
    out_dir = mp4.parent
    source_mtime_ms = mp4.stat().st_mtime * 1000
    _atomic_write_json(json_path, {"schemaVersion": SCHEMA_VERSION, "status": "analyzing",
                                   "model": key, "source": mp4.name, "sourceMtimeMs": source_mtime_ms})

    # Decode the lens-corrected copy when the recording is calibrated (see
    # undistort.py) — pose estimation then runs on undistorted frames. Sidecar
    # names/paths and staleness stay keyed to the raw clip.
    from calib_utils import (ANKLE_JOINT_HEIGHT_MM, analysis_source,
                             load_room_geometry, pixel_to_floor)
    src = analysis_source(mp4)
    cap = cv2.VideoCapture(str(src))
    reported_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    # These USB cameras report a nominal CAP_PROP_FPS (e.g. 30) but actually deliver
    # far fewer frames (variable rate, ~13.5fps). Using the reported value squashes
    # the timeline (t = idx/fps) and makes the sliding-window warm-up span ~2x the
    # real seconds. Prefer the *true* average fps = frame_count / duration so the
    # timeline lines up with the playing video and warm-up reflects real time.
    native_fps = _true_fps(src) or reported_fps
    stride = pick_stride(native_fps)  # samples per the clip's true fps (see pick_stride)

    # Metric room positions (mm, relative to the AprilTag's floor point) for
    # clips with embedded extrinsics + a known tag height; None otherwise and
    # the centroids sidecar stays exactly as before.
    room_geom = load_room_geometry(mp4, width, height, undistorted=(src != mp4))
    classify_every = pick_classify_every(stride, variant["classify_every"])  # frames between classifications

    def classify(window):
        # window: list of (kpts (17,2), conf (17,)) -> (label|None, conf, top)
        # where top is the K most-confident [label, prob] pairs (for the live
        # per-person bar graph); label is None when below the abstention threshold.
        pose_results = [{"keypoints": kp[None].astype("float32"),
                         "keypoint_scores": sc[None].astype("float32")} for kp, sc in window]
        res = infer(model, pose_results, (height, width))
        # mmaction's pred_score is already softmax probabilities. Recover logits
        # (log p, up to a constant softmax ignores), temperature-scale, re-softmax.
        # At temp=1 this returns p exactly — and fixes the prior double-softmax
        # that made the old confidence numbers meaningless.
        score = res.pred_score
        probs = (score.clamp_min(1e-8).log() / temp).softmax(-1)
        if disabled_idx:
            # Whitelist mask: zero out disallowed classes (no renormalize, so the
            # abstention check below still measures real probability mass — if the
            # model wanted a disabled class, the best allowed one stays weak -> idle).
            probs = probs.clone()
            probs[disabled_idx] = 0.0
        k = min(TOPK, int(probs.numel()))
        vals, idxs = probs.topk(k)
        vals, idxs = [float(v) for v in vals.tolist()], [int(i) for i in idxs.tolist()]
        name = lambda i: class_names[i] if i < len(class_names) else str(i)
        top = [[name(i), round(v, 3)] for v, i in zip(vals, idxs)]
        c, i = vals[0], idxs[0]
        if c < min_conf:
            return None, c, top  # abstain — too uncertain to name a class
        return name(i), c, top

    # The label from a trailing window best describes motion at the window's
    # center, ~half a window before the frame where it's computed. We have the
    # whole clip, so: pass 1 tracks + classifies (no drawing) and records each
    # frame's skeletons + each track's classification events; pass 2 re-decodes
    # and draws, shifting every label back by this offset to line it up with the
    # motion that produced it.
    offset_frames = int(WINDOW * stride * OFFSET_FRAC)

    # Pass 1 — track + classify.
    framedata = []                  # framedata[g] -> list of (tid, (x1,y1), pts, cs)
    ev_idx = defaultdict(list)      # per track: classify frame indices (ascending)
    ev_lab = defaultdict(list)      # per track: predicted label at each event
    tubes = defaultdict(lambda: deque(maxlen=WINDOW))
    trunc = defaultdict(lambda: deque(maxlen=WINDOW))  # per sample: too few keypoints visible?
    traj = defaultdict(list)        # per track: (idx, kpts, conf) every frame, for jump detection
    timeline = defaultdict(list)
    win_pose = defaultdict(list)    # per track: one [[x,y,c]*17] pose per classified window (aligned with timeline)
    centroids = defaultdict(list)   # per track: per-frame bbox centroid {t,x,y} (location tracking sidecar)
    boxes_by_frame = []             # per frame: [(tid, [x1,y1,x2,y2])] — AVA proposals
    seen = []

    # Per-frame detections as (ids, xyxy, xy, conf), from either the fast batched
    # ByteTrack path (default, YOLO) or the original streaming .track() (RTMPose,
    # or SMARTROOM_ACTION_FAST=0). The classify/window body below is identical.
    def _frame_source():
        if FAST_ACTION and pose_source != "rtmpose":
            for dets in _tracked_frames_fast(pose, src, DEVICE):
                if dets:
                    ids = [d[0] for d in dets]
                    xyxy = [d[1] for d in dets]
                    xy = np.asarray([d[2] for d in dets], dtype="float32")
                    conf = np.asarray([d[3] for d in dets], dtype="float32")
                    yield ids, xyxy, xy, conf
                else:
                    yield [], [], np.empty((0, 17, 2), "float32"), np.empty((0, 17), "float32")
            return
        for r in pose.track(str(src), stream=True, persist=True, classes=[0], device=DEVICE, verbose=False):
            boxes, kpts = r.boxes, r.keypoints
            if boxes is not None and boxes.id is not None and kpts is not None:
                ids = boxes.id.int().tolist()
                xyxy = boxes.xyxy.int().tolist()
                if pose_source == "rtmpose" and rtm is not None and len(xyxy):
                    # Keep YOLO's tracked boxes + IDs; take keypoints from RTMPose instead.
                    # rtmlib returns (N,17,2) pixel keypoints + (N,17) scores in the SAME
                    # order as the input boxes, so row n still maps to ids[n].
                    xy, conf = rtm(r.orig_img, bboxes=np.asarray(xyxy, dtype="float32"))
                    xy = np.asarray(xy, dtype="float32")
                    conf = np.asarray(conf, dtype="float32")
                else:
                    xy = kpts.xy.cpu().numpy()
                    conf = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones(xy.shape[:2], "float32")
                yield ids, xyxy, xy, conf
            else:
                yield [], [], np.empty((0, 17, 2), "float32"), np.empty((0, 17), "float32")

    idx = 0
    for ids, xyxy, xy, conf in _frame_source():
        perframe = []
        boxes_by_frame.append([(tid, list(xyxy[n])) for n, tid in enumerate(ids)])
        if len(ids):
            for n, tid in enumerate(ids):
                traj[tid].append((idx, xy[n], conf[n]))  # every frame (full vertical resolution)
                entry = {"t": round(idx / native_fps, 3),
                         "x": round((xyxy[n][0] + xyxy[n][2]) / 2.0, 1),
                         "y": round((xyxy[n][1] + xyxy[n][3]) / 2.0, 1)}
                if room_geom is not None:
                    # Ground contact pixel: the ankles when visible, else the box
                    # bottom (feet occluded -> position biased toward the camera,
                    # marked "bbox" so consumers can weight it accordingly).
                    # Stricter gate than kp_conf: a hallucinated ankle merely
                    # mis-draws a skeleton, but it projects meters of position error.
                    vis = [j for j in (15, 16) if conf[n][j] >= max(kp_conf, ROOM_KP_CONF)]
                    if vis:
                        gu = float(sum(xy[n][j][0] for j in vis)) / len(vis)
                        gv = float(sum(xy[n][j][1] for j in vis)) / len(vis)
                        ground_src, plane = "ankles", ANKLE_JOINT_HEIGHT_MM
                    else:
                        gu, gv = (xyxy[n][0] + xyxy[n][2]) / 2.0, float(xyxy[n][3])
                        ground_src, plane = "bbox", 0.0
                    hit = pixel_to_floor(gu, gv, room_geom, plane)
                    entry["room"] = ([round(hit[0], 1), round(hit[1], 1)]
                                     if hit is not None else None)
                    entry["src"] = ground_src
                centroids[tid].append(entry)
                if idx % stride == 0:
                    tubes[tid].append((xy[n], conf[n]))
                    n_vis = int((conf[n] >= kp_conf).sum())
                    trunc[tid].append(n_vis < MIN_KEYPOINTS)
                if (not is_ava) and len(tubes[tid]) >= MIN_WINDOW and idx % classify_every == 0:
                    # Judge truncation on the recent samples (around the moment the
                    # label describes), not the whole trailing window — scattered
                    # single-frame keypoint dropouts shouldn't count, but a person
                    # who's currently cut off should.
                    recent = list(trunc[tid])[-MIN_WINDOW:]
                    if recent and sum(recent) * 2 > len(recent):
                        # Currently cut off — skeleton is partial, so don't classify;
                        # mark the window "not fully in frame".
                        label, cf, top, action_lab = None, 0.0, [], PARTIAL
                    else:
                        # Front-pad partial windows (repeat the earliest pose) so the
                        # classifier always gets WINDOW frames and predictions can begin
                        # before the window is naturally full.
                        win = list(tubes[tid])
                        if len(win) < WINDOW:
                            win = [win[0]] * (WINDOW - len(win)) + win
                        label, cf, top = classify(win)
                        action_lab = label or IDLE
                    ev_idx[tid].append(idx)
                    # overlay shows "idle"/"not fully in frame" rather than a stale or
                    # guessed label; the summary vote + chips only count confident windows.
                    ev_lab[tid].append(action_lab)
                    # Every window goes in the timeline (with its top-K for the live
                    # bar graph); `kept` marks the confident ones that vote + chip.
                    t = max(0.0, (idx - offset_frames) / native_fps)
                    timeline[tid].append({"t": round(t, 3), "action": action_lab,
                                          "conf": round(cf, 3), "kept": label is not None, "top": top})
                    # One pose per classified window (current frame's keypoints), aligned 1:1
                    # with the timeline entry above -> feeds the per-person sidecar.
                    win_pose[tid].append([[round(float(xy[n][j][0]), 1), round(float(xy[n][j][1]), 1),
                                           round(float(conf[n][j]), 3)] for j in range(len(xy[n]))])
                    if label is not None and label not in seen:
                        seen.append(label)
                x1, y1 = max(0, xyxy[n][0]), max(0, xyxy[n][1])
                perframe.append((tid, (x1, y1), xy[n], conf[n]))
        framedata.append(perframe)
        idx += 1

    # AVA classifies from RGB, so it needs its own decode pass over the clip
    # (pass 1 kept only keypoints and boxes). Everything it fills is the same
    # structures the skeleton path fills above.
    if is_ava:
        _ava_pass(src, framedata, boxes_by_frame, width, height, native_fps,
                  classify_every, offset_frames, ev_idx, ev_lab, timeline,
                  win_pose, seen)

    # Geometric jump detection over each track's full trajectory (classifier-
    # independent). Surface "jumping" as a chip when any track jumps.
    jumps = {tid: ev for tid in traj if (ev := detect_jumps(traj[tid], native_fps, kp_conf))}
    if jumps and "jumping" not in seen:
        seen.append("jumping")

    # Pass 2 — re-decode and draw the annotated video. Skipped entirely when
    # annotated videos are disabled (SMARTROOM_DETECT_ANNOTATE=0): it re-decodes
    # the whole clip + draws + re-encodes, and the 3D view / API run on the JSON
    # sidecars, not this overlay. This is the single biggest action-stage cost
    # after the pose pass, so skipping it ~halves the stage when overlays aren't
    # needed. Frame g shows the label whose window was centered on g (the latest
    # classification computed at or before g+offset).
    has_annotated = False
    tmp_raw = annotated_path.with_suffix(".raw.mp4")
    if ANNOTATE:
        writer = cv2.VideoWriter(str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"), native_fps, (width, height))
        cap = cv2.VideoCapture(str(src))  # same frames the keypoints were computed on
        for g, perframe in enumerate(framedata):
            ret, frame = cap.read()
            if not ret:
                break
            fh, fw = frame.shape[:2]
            for tid, (x1, y1), pts, cs in perframe:
                for a, b in COCO_SKELETON:
                    if cs[a] > kp_conf and cs[b] > kp_conf:
                        cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), (0, 200, 0), 2)
                # Only the classifier's labels go on this video; geometric jump events
                # are kept separate (sidecar + the Analytics "Geometric" sub-view).
                pos = bisect.bisect_right(ev_idx[tid], g + offset_frames) - 1
                text = f"#{tid} {ev_lab[tid][pos]}" if pos >= 0 else f"#{tid} ..."
                font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                (tw, th), base = cv2.getTextSize(text, font, scale, thick)
                box_w, box_h = tw + 6, th + base + 4
                # Keep the label fully on-screen: sit above the box normally, but flip
                # below the box top when the person is near the top edge, and clamp
                # horizontally so a wide label never runs off either side.
                bx = max(0, min(x1, fw - box_w)) if fw >= box_w else 0
                by = y1 - box_h if y1 - box_h >= 0 else y1
                by = max(0, min(by, fh - box_h))
                cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (0, 200, 0), -1)
                cv2.putText(frame, text, (bx + 3, by + th + 2), font, scale, (0, 0, 0), thick)
            writer.write(frame)
        cap.release()
        writer.release()

    if tmp_raw.exists():
        final_tmp = annotated_path.with_suffix(".enc.mp4")
        proc = _transcode_h264(tmp_raw, final_tmp)
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
            if p.get("kept"):  # idle/abstained windows don't vote
                counts[p["action"]] = counts.get(p["action"], 0) + 1
        if counts:
            track_actions[str(tid)] = max(counts, key=counts.get)

    jumps_sec = {str(t): [{"start": round(e["start"] / native_fps, 3),
                           "end": round(e["end"] / native_fps, 3), "peak": e["peak"]}
                          for e in ev] for t, ev in jumps.items()}
    _atomic_write_json(actions_path, {"schemaVersion": SCHEMA_VERSION, "model": key,
                                      "source": mp4.name, "sourceMtimeMs": source_mtime_ms,
                                      "nativeFps": round(native_fps, 3), "window": WINDOW, "stride": stride,
                                      "poseSource": pose_source,
                                      "tracks": {str(t): timeline[t] for t in timeline},
                                      "jumps": jumps_sec})

    # Per-person sidecar (real_dataset-style): for each track, the classification as
    # both per-window rows (pose + label, one per classified window) and merged
    # from->to segments, plus the geometric jump events. Centroid/location tracking
    # is kept in a *separate* file (camera_main.centroids.<model>.json) below.
    def _segments(tl):
        # Collapse consecutive equal-label windows into {action,start,end,conf} ranges.
        segs = []
        for p in tl:
            if segs and segs[-1]["action"] == p["action"]:
                segs[-1]["end"] = p["t"]
                segs[-1]["_c"].append(p["conf"])
            else:
                segs.append({"action": p["action"], "start": p["t"], "end": p["t"], "_c": [p["conf"]]})
        for s in segs:
            c = s.pop("_c")
            s["conf"] = round(sum(c) / len(c), 3)
        return segs

    persons = {}
    for tid in timeline:
        windows = [{"t": p["t"], "action": p["action"], "conf": p["conf"],
                    "kept": p["kept"], "keypoints": kp}
                   for p, kp in zip(timeline[tid], win_pose[tid])]
        persons[str(tid)] = {"segments": _segments(timeline[tid]),
                             "jumps": jumps_sec.get(str(tid), []),
                             "windows": windows}
    persons_path = out_dir / f"{mp4.stem}.persons.{key}.json"
    _atomic_write_json(persons_path, {
        "schemaVersion": SCHEMA_VERSION, "model": key, "source": mp4.name,
        "sourceMtimeMs": source_mtime_ms, "nativeFps": round(native_fps, 3),
        "window": WINDOW, "stride": stride, "poseSource": pose_source, "persons": persons})

    # Location/centroid tracking — its own file (per-frame bbox centroid per
    # person, plus metric room positions when the clip has extrinsics).
    centroids_doc = {
        "schemaVersion": SCHEMA_VERSION, "model": key, "source": mp4.name,
        "sourceMtimeMs": source_mtime_ms, "nativeFps": round(native_fps, 3),
        "persons": {str(t): centroids[t] for t in centroids}}
    if room_geom is not None:
        cam_pos = room_geom["cam_pos_mm"]
        centroids_doc["roomFrame"] = {
            "origin": "floor point directly under the AprilTag's center",
            "axes": "X = tag's right (viewed facing the tag), Z = out of the wall; mm",
            "tagId": room_geom["tag_id"],
            "tagHeightMm": round(room_geom["tag_height_mm"], 1),
            "cameraPositionMm": [round(float(v), 1) for v in cam_pos],
            "cameraId": room_geom["camera_id"],
        }
    centroids_path = out_dir / f"{mp4.stem}.centroids.{key}.json"
    _atomic_write_json(centroids_path, centroids_doc)
    _atomic_write_json(json_path, {
        "schemaVersion": SCHEMA_VERSION, "status": "done", "error": None,
        "model": key, "source": mp4.name, "sourceMtimeMs": source_mtime_ms,
        "sourceVideo": "undistorted" if src != mp4 else "raw",
        "device": DEVICE, "classifier": variant["classifier"], "poseSource": pose_source,
        # Settings that produced this analysis (shown in the dashboard header).
        "stride": stride, "classifyEvery": classify_every,
        "samplesPerClassify": max(1, round(classify_every / stride)),
        "analyzedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "durationSec": round(total / native_fps, 3) if total else None,
        "temp": temp, "minConf": min_conf,
        "tracks": len(ev_idx), "trackActions": track_actions, "actions": seen,
        "jumps": sum(len(ev) for ev in jumps.values()),
        "annotated": annotated_path.name if has_annotated else None, "hasAnnotated": has_annotated,
    })
    print(f"  action done: {mp4.relative_to(saved_root())} tracks={len(timeline)} {seen}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Per-person skeleton action recognition over saved clips.")
    ap.add_argument("--path", action="append", metavar="REL",
                    help="clip to analyze, relative to the recordings root; repeatable for a subset")
    ap.add_argument("--variant", default="ntu",
                    help="action model(s), comma-separated: ntu (ST-GCN++/NTU-60 skeleton), "
                         "hmdb (PoseC3D/HMDB51 skeleton), ava (SlowFast-AVA, RGB pixels)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    variant_keys = [v.strip() for v in args.variant.split(",") if v.strip() in VARIANTS]
    if not variant_keys:
        print(f"no valid --variant in {args.variant!r}", file=sys.stderr)
        return 2
    root = saved_root()
    if not root.exists():
        print(f"no recordings dir: {root}", file=sys.stderr)
        return 0

    # Suffix lets GPU-sharded workers hold separate locks (see run-analysis.sh).
    sfx = os.environ.get("SMARTROOM_LOCK_SUFFIX", "")
    lock_file = open(root / f".action.lock{sfx}", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another action run is in progress; exiting", file=sys.stderr)
        return 0
    try:
        os.setpgrp()
    except OSError:
        pass
    pid_path = root / f".action.pid{sfx}"
    try:
        pid_path.write_text(str(os.getpid()))
    except OSError:
        pass

    try:
        # Action classification runs only on the PRIMARY RGB per recording:
        # legacy webcam clips and the D455 color stream (30fps). The D435
        # (15fps secondary) gets detection + pose only — see detect.py.
        ACTION_SOURCES = ("camera_main.mp4", "camera_d455_color.mp4")
        if args.path:
            clips = [root / p for p in args.path]
        else:
            clips = sorted((p for name in ACTION_SOURCES for p in root.rglob(name)),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        # undistorted/ holds lens-corrected COPIES of clips, not additional clips.
        clips = [c for c in clips if c.exists() and "undistorted" not in c.parts]

        from mmaction.apis import inference_skeleton, init_recognizer
        from ultralytics import YOLO
        pose = YOLO(str(pose_weights()))  # tracked boxes + IDs (and keypoints, for yolo source)

        # Optional alternative skeleton source (see resolve_pose_source). Built once,
        # shared across variants/clips; the import stays lazy so yolo-source runs never
        # touch rtmlib/onnxruntime.
        pose_source = resolve_pose_source()
        rtm = load_rtmpose() if pose_source == "rtmpose" else None
        print(f"pose source: {pose_source}", file=sys.stderr)

        # One process handles every requested variant (the global lock serializes
        # action runs, so all variants run here rather than spawning one process each).
        for vkey in variant_keys:
            variant = VARIANTS[vkey]
            key = variant["key"]
            todo = [c for c in clips if needs_action(c, args.force, key)]
            tag = f"action[{vkey}]"
            print(f"{tag}: {len(todo)}/{len(clips)} clip(s) to process", file=sys.stderr)
            if not todo:
                continue
            # AVA builds its own detector inside the pass (it needs mmdet's ROI
            # head, not a skeleton recognizer), so there is nothing to build here.
            model = (None if variant.get("kind") == "ava" else
                     init_recognizer(variant_config(variant), variant_ckpt(variant), device=DEVICE))
            label = ({"hmdb": "Actions (HMDB)", "ava": "Actions (AVA)"}
                     .get(vkey, "Actions (NTU)")
                     + (" · RTMPose" if pose_source == "rtmpose" else ""))
            started = dt.datetime.now(dt.timezone.utc)
            processed = errors = 0
            for mp4 in todo:
                try:
                    print(f"{tag}: processing {mp4.relative_to(root)}", file=sys.stderr)
                    process_clip(model, inference_skeleton, pose, mp4, variant,
                                 rtm=rtm, pose_source=pose_source)
                    processed += 1
                except Exception as error:  # noqa: BLE001
                    errors += 1
                    print(f"  action error: {error}", file=sys.stderr)
                    jp, _, _ = sidecars(mp4, key)
                    try:
                        _atomic_write_json(jp, {"schemaVersion": SCHEMA_VERSION, "status": "error",
                                                "model": key, "error": str(error), "source": mp4.name,
                                                "sourceMtimeMs": mp4.stat().st_mtime * 1000})
                    except Exception:
                        pass
                finally:
                    # Release this clip's frame buffers before the next one. CPU torch
                    # holds onto allocations, so without this a long batch grows until
                    # the OOM killer stops it a couple of clips in.
                    gc.collect()
            write_run_stats(root, key, label, started, processed, errors, len(clips) - len(todo))
        return 0
    finally:
        try:
            pid_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
