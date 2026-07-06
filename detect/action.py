#!/usr/bin/env python3
"""
Per-person action recognition over the saved recordings.

YOLO26-pose with tracking gives a stable id + COCO-17 skeleton per person each
frame; we keep a per-track-id sliding window of keypoints and feed each window
to a pretrained skeleton recognizer (mmaction2, CPU), then overlay the predicted
action label on each tracked person. Selectable via --variant:
  ntu  (default) — 2D ST-GCN++ on NTU-RGB+D 60  -> sidecar key "action"
  hmdb           — PoseC3D on HMDB51 (adds walk/run) -> sidecar key "action-hmdb"
Both are skeleton models driven one person at a time, so both are multi-person.

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
MIN_KEYPOINTS = int(os.environ.get("SMARTROOM_ACTION_MIN_KEYPOINTS", "10"))
# How many top classes to record per window for the live per-person bar graph.
TOPK = int(os.environ.get("SMARTROOM_ACTION_TOPK", "5"))
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


def detect_jumps(traj, fps):
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
        vis = cf >= KP_CONF
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
}


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
    s = mp4.stem
    return (mp4.with_name(f"{s}.detections.{key}.json"),
            mp4.with_name(f"{s}.actions.{key}.json"),
            mp4.with_name(f"{s}.annotated.{key}.mp4"))


def _atomic_write_json(path: Path, data: dict):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


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


def process_clip(model, infer, pose, mp4: Path, variant: dict):
    import cv2
    import numpy as np

    key, class_names = variant["key"], variant["labels"]
    temp, min_conf = variant_temp(variant), variant_min_conf(variant, len(class_names))
    disabled_idx = load_disabled(key, class_names)
    json_path, actions_path, annotated_path = sidecars(mp4, key)
    source_mtime_ms = mp4.stat().st_mtime * 1000
    _atomic_write_json(json_path, {"schemaVersion": SCHEMA_VERSION, "status": "analyzing",
                                   "model": key, "source": mp4.name, "sourceMtimeMs": source_mtime_ms})

    cap = cv2.VideoCapture(str(mp4))
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
    native_fps = _true_fps(mp4) or reported_fps
    stride = pick_stride(native_fps)  # samples per the clip's true fps (see pick_stride)
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
    seen = []
    idx = 0
    for r in pose.track(str(mp4), stream=True, persist=True, classes=[0], device="cpu", verbose=False):
        boxes = r.boxes
        kpts = r.keypoints
        perframe = []
        if boxes is not None and boxes.id is not None and kpts is not None:
            ids = boxes.id.int().tolist()
            xyxy = boxes.xyxy.int().tolist()
            xy = kpts.xy.cpu().numpy()
            conf = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones(xy.shape[:2], "float32")
            for n, tid in enumerate(ids):
                traj[tid].append((idx, xy[n], conf[n]))  # every frame (full vertical resolution)
                centroids[tid].append({"t": round(idx / native_fps, 3),
                                       "x": round((xyxy[n][0] + xyxy[n][2]) / 2.0, 1),
                                       "y": round((xyxy[n][1] + xyxy[n][3]) / 2.0, 1)})
                if idx % stride == 0:
                    tubes[tid].append((xy[n], conf[n]))
                    n_vis = int((conf[n] >= KP_CONF).sum())
                    trunc[tid].append(n_vis < MIN_KEYPOINTS)
                if len(tubes[tid]) >= MIN_WINDOW and idx % classify_every == 0:
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

    # Geometric jump detection over each track's full trajectory (classifier-
    # independent). Surface "jumping" as a chip when any track jumps.
    jumps = {tid: ev for tid in traj if (ev := detect_jumps(traj[tid], native_fps))}
    if jumps and "jumping" not in seen:
        seen.append("jumping")

    # Pass 2 — re-decode and draw; frame g shows the label whose window was
    # centered on g, i.e. the latest classification computed at or before g+offset.
    tmp_raw = annotated_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"), native_fps, (width, height))
    cap = cv2.VideoCapture(str(mp4))
    for g, perframe in enumerate(framedata):
        ret, frame = cap.read()
        if not ret:
            break
        fh, fw = frame.shape[:2]
        for tid, (x1, y1), pts, cs in perframe:
            for a, b in COCO_SKELETON:
                if cs[a] > 0.3 and cs[b] > 0.3:
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
    persons_path = mp4.with_name(f"{mp4.stem}.persons.{key}.json")
    _atomic_write_json(persons_path, {
        "schemaVersion": SCHEMA_VERSION, "model": key, "source": mp4.name,
        "sourceMtimeMs": source_mtime_ms, "nativeFps": round(native_fps, 3),
        "window": WINDOW, "stride": stride, "persons": persons})

    # Location/centroid tracking — its own file (per-frame bbox centroid per person).
    centroids_path = mp4.with_name(f"{mp4.stem}.centroids.{key}.json")
    _atomic_write_json(centroids_path, {
        "schemaVersion": SCHEMA_VERSION, "model": key, "source": mp4.name,
        "sourceMtimeMs": source_mtime_ms, "nativeFps": round(native_fps, 3),
        "persons": {str(t): centroids[t] for t in centroids}})
    _atomic_write_json(json_path, {
        "schemaVersion": SCHEMA_VERSION, "status": "done", "error": None,
        "model": key, "source": mp4.name, "sourceMtimeMs": source_mtime_ms,
        "device": "cpu", "classifier": variant["classifier"],
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
    ap.add_argument("--path", help="single clip, relative to the recordings root")
    ap.add_argument("--variant", choices=list(VARIANTS), default="ntu",
                    help="action model: ntu (ST-GCN++/NTU-60, default) or hmdb (PoseC3D/HMDB51)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    variant = VARIANTS[args.variant]
    key = variant["key"]
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
        todo = [c for c in clips if needs_action(c, args.force, key)]
        print(f"action[{args.variant}]: {len(todo)}/{len(clips)} clip(s) to process", file=sys.stderr)
        if not todo:
            return 0

        from mmaction.apis import inference_skeleton, init_recognizer
        from ultralytics import YOLO
        model = init_recognizer(variant_config(variant), variant_ckpt(variant), device="cpu")
        pose = YOLO(str(pose_weights()))

        for mp4 in todo:
            try:
                print(f"action[{args.variant}]: processing {mp4.relative_to(root)}", file=sys.stderr)
                process_clip(model, inference_skeleton, pose, mp4, variant)
            except Exception as error:  # noqa: BLE001
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
        return 0
    finally:
        try:
            pid_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
