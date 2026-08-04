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
  POST /timing/start?seconds=25   arm the lights on/off timing calibration
  GET  /timing/status             its progress and the measured per-camera offsets

Cameras do not all reach this server at the same speed — a Reolink frame comes
through the NVR's buffer and a second host, a RealSense frame over one LAN hop —
so a per-camera offset is measured (see timing_sync.py) and subtracted before any
two cameras' detections are compared. Uncalibrated, every camera is 0 and the
behaviour is exactly what it was before offsets existed.

Calibration is NOT sent from the Pi. Extrinsics are static, so `geom` is built
once from the newest UPLOADED recording that contains this camera (its
metadata.json already embeds calibration + extrinsics) via
`calib_utils.load_room_geometry` — the same function the batch localizer uses.

Env:
  SMARTROOM_SAVE_DIR        recordings root (to find a clip for calibration)
  SMARTROOM_DETECT_DEVICE   torch device ("0" for GPU, "cpu"); default auto
  SMARTROOM_LIVE_WEIGHTS    pose weights (default ~/Code/yolo-bench/yolo26n-pose.pt)
  SMARTROOM_LIVE_TIMING     measured camera offsets (default calibration/live_timing.json)
  SMARTROOM_TIME_OFFSET_<CAM>  override one camera's offset in ms

Usage:
  python detect/live_infer.py --cam camera_d455_color --port 8010
"""

import argparse
import datetime as dt
import faulthandler
import json
import multiprocessing as mp
import os
import re
import shutil
import signal
import struct
import subprocess
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
from localize import (  # noqa: E402
    MAX_RAY_REACH_MM,
    MIN_RAY_PITCH_DEG,
    backproject_room,
    ground_point,
    hip_point,
    joint_px,
)
from calib_utils import ANKLE_JOINT_HEIGHT_MM, pixel_to_floor  # noqa: E402
import timing_sync  # noqa: E402

# COCO-17 shoulders (fallback anchor when the hips are occluded, e.g. seated at
# a desk). Both anchors are ranged by real depth — no floor-ray, never the feet.
L_SHOULDER, R_SHOULDER = 5, 6

BOUNDARY = "frame"
JPEG_QUALITY = 75
KP_CONF = float(os.environ.get("SMARTROOM_ROOM_KP_CONF", "0.5"))
# A person is only real if the POSE is real, not just the box. YOLO's person box
# clears the tracker's 0.55 gate on dark clutter (the office chairs bottom-right),
# but the keypoints on such a phantom are garbage: ~2 joints above KP_CONF and
# ~0.18 mean confidence, vs 8-15 confident joints on an actual human. Without a
# pose-quality gate a phantom that happens to land a shoulder anchor gets a depth
# range, persists past SEGMENT_MIN_PEOPLE_FRAMES, and preserves an empty segment
# (and paints a footprint in the room map). Require this many joints above
# KP_CONF before a detection counts as a person to localize or to keep a segment.
MIN_POSE_KP = int(os.environ.get("SMARTROOM_MIN_POSE_KP", "5"))
DEPTH_MATCH_FRAC = 0.06   # a depth sample within this (fraction of frame) counts as "this hip"
# A 1s-old depth sample applied to a moving person puts them metres away and was
# a source of bad room positions (and hence false cross-camera merges). The
# back-channel polls ~8Hz, so 0.35s keeps roughly the freshest sample and drops
# the rest — better to skip a person this frame than to localize them wrongly.
DEPTH_STALE_S = float(os.environ.get("SMARTROOM_DEPTH_STALE_S", "0.35"))
# A depth camera measures the body SURFACE FACING IT, so each camera places a
# person half a torso-depth toward itself. Viewed from different sides the two
# cameras therefore disagree by roughly a whole torso depth — measured here as a
# ~300mm systematic offset, almost entirely along the camera-separation axis.
# Pushing each sample this far further along the viewing ray approximates the
# body CENTRE. TESTED AND REJECTED (default 0): at 150mm the cross-camera
# disagreement got WORSE, 300mm -> 480mm. The premise does not hold for this
# layout — the D455 (x=-1952) and D435 (x=+296) sit on the SAME side of a person
# at x~700, so they see the same-facing surface and there is no opposing bias to
# cancel; pushing both along their differing rays just separates them. Kept as a
# knob in case the cameras are ever repositioned to face each other.
BODY_HALF_DEPTH_MM = float(os.environ.get("SMARTROOM_BODY_HALF_DEPTH_MM", "0"))
# Cameras whose ROOM POSITIONS are not trusted (comma-separated cam keys). They
# keep streaming, keep their pose and action labels, and keep being recorded —
# they just publish no spatial measurement, and no geo into their segments.
#
# Why a camera would be muted rather than fixed: a position is only as good as
# the depth sample under it. The D435 sits close enough that a person fills its
# frame, so it has no confident hip and falls back to the SHOULDER anchor, whose
# depth lands on the wall behind them: it reported a person 1424 mm away when the
# D455's estimate put them 383 mm away (3.7x — no depth bias does that). Fed into
# the fusion, that placed one person in two spots ~1.1 m apart and counted them
# twice. A camera contributing a wrong position is worse than one contributing
# none, because the wrong one still gets averaged and still gets counted.
NO_SPATIAL = {c.strip() for c in os.environ.get("SMARTROOM_NO_SPATIAL", "").split(",")
              if c.strip()}
# The allowlist form: when set, ONLY these cameras publish room positions and
# every other camera is muted. Preferred over listing the muted ones when the
# intent is "one camera is the source of truth for location", because it holds for
# cameras that do not exist yet -- four NVR cameras appeared in this pipeline
# without anyone deciding they should localize, and a denylist would have let a
# fifth do the same silently.
SPATIAL_ONLY = {c.strip() for c in os.environ.get("SMARTROOM_SPATIAL_ONLY", "").split(",")
                if c.strip()}


# One lock per CUDA device, serializing pose inference across the camera threads
# that share it.
#
# Pose runs in THREADS in this process (ReID and AVA were long ago moved into
# subprocesses because a corrupted CUDA context never recovers; pose never was).
# Ultralytics' predict is not safe to call concurrently on one device, and the
# failure is not a clean exception: a wedge dump caught THREE camera threads
# blocked simultaneously inside the same op, `head.py get_topk_index`, none of
# which ever returned. The watchdog then killed the process at 60s. The companion
# symptom is `CUDA error: misaligned address` — the same collision, seen by
# whichever thread got the exception instead of the hang.
#
# Serializing costs little here: each camera loop runs at 55-75 fps against a
# 15 fps input, so two cameras sharing a device still clear the input rate with
# room to spare. Correctness first; if throughput ever becomes the constraint the
# real fix is a subprocess per camera, matching ReID/AVA and the Pi's depth page.
POSE_LOCKS = defaultdict(threading.Lock)


def spatial_muted(cam_key: str) -> bool:
    """Is this camera barred from publishing room positions?"""
    if SPATIAL_ONLY:
        return cam_key not in SPATIAL_ONLY
    return cam_key in NO_SPATIAL
# The depth back-channel polls at ~8Hz while the pose loop runs 30-60fps, and a
# sample must land near the anchor to match. A person who is STILL matches every
# frame; a MOVING one outruns the last sample, and the person used to be dropped
# from that frame entirely — vanishing from the overlay and the map, which reads
# as flicker. Hold their last known room position briefly instead.
POS_HOLD_S = float(os.environ.get("SMARTROOM_POS_HOLD_S", "0.7"))
ACTION_WINDOW = 48        # skeleton-window length (mirrors action.WINDOW); deque cap
ACTION_TRACK_TTL_S = 2.0  # drop a track's window/label if unseen this long
ACTION_SWEEP_S = 0.35     # how often the action thread re-classifies live tracks
AVA_BUF = 128             # rolling RGB frame buffer (a few seconds at any fps)
AVA_PERIOD_S = 0.4        # how often to run the (heavier) AVA forward
# AVA_SHORT / AVA_THR / the class blacklist live in ava_model, shared with the
# batch pass (action.py --variant ava) so both classify identically.
from ava_model import AVA_SHORT, AVA_THR, resize_short as _resize_short  # noqa: E402
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
# (The old _MODEL_BUILD_LOCK lived here: mmaction's registry could not be
# populated from two threads at once. Moot now — every model is built inside its
# own process, so there is no shared registry left to race on. ava_model keeps
# its own lock for the batch pipeline, which does still build in-process.)

# --- person re-identification (stable identity across gaps and cameras) -------
# ByteTrack ids are per-camera and reset whenever a track is lost, so a person
# who is occluded, leaves, or is seen by the other camera gets a fresh id. The
# registry maps (cam, track id) -> a GLOBAL id using two signals:
#   1. geometry — every camera localizes into the same room frame (measured ~7cm
#      agreement between the two RealSense), so two detections at the same room
#      point at the same moment are the same person. Cheap + strong. "The same
#      moment" is only meaningful once the cameras share a timeline, which is what
#      CAM_OFFSETS below is for — before it, a Reolink detection was compared
#      against a RealSense one hundreds of ms out of step.
#   2. appearance — a ReID embedding (ultralytics' encoder), which is what can
#      bridge a long absence where geometry says nothing.
REID_MODEL = os.environ.get("SMARTROOM_REID_MODEL", "yolo26n-reid.onnx")
REID_ON = os.environ.get("SMARTROOM_REID", "1") != "0"
# Calibrated on live data from this room (see IdentityRegistry.stats). Measured
# same-person cosine ~0.95 median (p05 0.88); different-person ~0.17 median with
# p99 ~0.23-0.44 depending on who is in frame. Swept empirically: 0.70 gave 74
# identities for ~4 people (severe fragmentation), 0.55 gave 17, 0.45 gave 13 but
# put the threshold ON the impostor tail. 0.55 keeps a wide safety margin while
# fixing most fragmentation. Prefer fragmentation over a false merge: a split
# identity is recoverable, two people fused into one is not.
# NOTE: no threshold makes appearance work ACROSS cameras here — they view
# opposite sides of people, so same-person-cross-camera scores below
# different-person-same-camera. Cross-camera fusion needs geometry (see
# GEO_MERGE_MM), gated by an appearance check to stay safe.
REID_THRESH = float(os.environ.get("SMARTROOM_REID_THRESH", "0.55"))  # cosine
REID_EVERY = int(os.environ.get("SMARTROOM_REID_EVERY", "3"))         # frames
# Location proposes cross-camera merges (appearance alone cannot: the cameras see
# opposite sides of people). 0 disables it entirely.
GEO_MERGE_MM = float(os.environ.get("SMARTROOM_GEO_MERGE_MM", "600"))
# ...but location NEVER decides alone. Room positions proved inaccurate enough to
# put two different people at the same point (a curly-haired woman and a man in a
# brown shirt computed to within 256mm and were fused). So a geometric merge also
# requires appearance not to contradict. The bar is deliberately LOW: same-person
# scores across these opposed viewpoints are weak, so a high bar would block every
# genuine cross-camera match. MEASURED RESULT: it does not work at all across
# THESE two cameras. One person, alone in the room, seen by both, scored below
# 0.20 — the D435 sees the back of his head close-up while the D455 sees him
# side-on at distance, so same-person similarity is indistinguishable from two
# strangers. The veto therefore blocked every legitimate cross-camera merge and
# is disabled (0). The real guard is GEO_FUSE_PERSIST plus a correct calibration:
# the false merges that motivated the veto were caused by the D435 pose flip
# making unrelated people compute to the same point, not by geometry itself.
# With a BAD D435 pose the candidate scores sat at p50 0.129 (garbage pairs, 94%
# vetoed); once the pose was fixed they rose to p50 0.334 / p90 0.556 — a useful
# signal that the calibration is sound. If this median collapses again, suspect
# the extrinsics before touching the threshold.
GEO_REID_MIN = float(os.environ.get("SMARTROOM_GEO_REID_MIN", "0"))  # 0 = no veto
GEO_MERGE_S = 0.5          # detections must be this close in time to fuse
# A merge can be wrong (two people who were briefly close). The sticky map would
# keep them fused forever, so re-check: if the other camera places this identity
# implausibly far away at the same moment, break the mapping and re-match.
GEO_SPLIT_MM = float(os.environ.get("SMARTROOM_GEO_SPLIT_MM", "1200"))
# Consecutive co-located observations required before two identities are fused,
# so two people passing each other are not merged on a single coincidence.
GEO_FUSE_PERSIST = int(os.environ.get("SMARTROOM_GEO_FUSE_PERSIST", "5"))
# The split rule must be as reluctant as the fuse rule, or the two fight: a single
# frame where the cameras disagreed split an identity, fuse() immediately re-merged
# it, and one stable track oscillated 1 -> 10 -> 1 -> 11. Require the disagreement
# to persist before believing it.
GEO_SPLIT_PERSIST = int(os.environ.get("SMARTROOM_GEO_SPLIT_PERSIST", "8"))
GALLERY_TTL_S = float(os.environ.get("SMARTROOM_GALLERY_TTL_S", "300"))
# How much of each identity's recent trajectory to remember, so a LATE camera can
# be compared against where that person was when its frame was captured.
#
# Subtracting a measured offset relabels a frame's time; it does not make the
# other cameras' past available to compare against. The gallery used to hold only
# each identity's LATEST position, which is fine while every camera is within a
# frame or two of the others -- and useless the moment one is not. The Reolink
# cameras come through the NVR several seconds behind: by the time one of their
# frames is processed, the RealSense entry for that same instant has been
# overwritten by seconds of newer positions, so the comparison is still between
# two different moments. That is the ORIGINAL bug, surviving the offset fix.
#
# So keep a short trail per identity and look up the position at the queried time.
# Must comfortably exceed the largest camera delay; 20s covers a 4-5s NVR path
# with room to spare, and costs a few hundred (t, x, z) tuples per person.
GEO_HIST_S = float(os.environ.get("SMARTROOM_GEO_HIST_S", "20"))
EMB_MOMENTUM = 0.9         # running-mean weight for a track's stored embedding

# --- one comparable timeline across cameras ----------------------------------
# Every rule above that compares two cameras (GEO_MERGE_S, GEO_SPLIT_MM, fuse())
# needs to know WHEN each detection happened. It used to use `time.time()` at the
# moment this loop picked the frame up, which is two errors deep:
#
#   1. inference-start, not arrival — a camera whose GPU is busier is stamped
#      systematically later than one that is idle, for no physical reason;
#   2. no allowance for transport — the RealSense cameras arrive over one LAN hop
#      from the Pi, while a Reolink frame waits on the NVR's encoder and buffer,
#      crosses RTSP to a second host, and is re-encoded to JPEG there.
#
# (1) is fixed by stamping arrival in the ingest handler (Shared.in_recv_ms).
# (2) cannot be derived — it is a property of the hardware path — so it is
# MEASURED, per camera, by the lights on/off calibration in timing_sync.py, and
# subtracted here. The reference camera is 0 by definition; every other camera's
# frames are treated as having been captured `offset_ms` before they landed.
#
# Every camera gets an offset, INCLUDING the RealSense pair. Their sensor clocks
# are already synchronised with each other by librealsense (that is what the Pi's
# calibration/camera_timing.json measures), but that says nothing about how long
# the depth page, live_forward and the network take to deliver a frame here —
# which is the delay that has to match Reolink's to fuse against it.
CAM_OFFSETS = {}           # cam_key -> ms to subtract from its chosen clock
CAM_CLOCKS = {}            # cam_key -> "hw" (own per-frame stamp) | "arrival"
OFFSETS_LOCK = threading.Lock()


def reload_offsets(cam_keys=None) -> dict:
    """Re-read the stored offsets (env still wins). Called at startup and again
    whenever a calibration finishes, so a new measurement takes effect without a
    restart — the point of a calibration button is not having to bounce six
    models to use its result."""
    stored = timing_sync.load_offsets()
    clocks = timing_sync.load_capture_clocks()
    keys = set(stored) | set(cam_keys or ()) | set(CAM_OFFSETS)
    fresh = {k: timing_sync.offset_ms_for(k, stored) for k in keys}
    with OFFSETS_LOCK:
        CAM_OFFSETS.clear()
        CAM_OFFSETS.update(fresh)
        CAM_CLOCKS.clear()
        CAM_CLOCKS.update({k: clocks.get(k, "arrival") for k in keys})
        return dict(CAM_OFFSETS)


def cam_offset_ms(cam_key: str) -> float:
    with OFFSETS_LOCK:
        return CAM_OFFSETS.get(cam_key, 0.0)


def all_offsets() -> dict:
    with OFFSETS_LOCK:
        return dict(CAM_OFFSETS)


def cam_clock(cam_key: str) -> str:
    with OFFSETS_LOCK:
        return CAM_CLOCKS.get(cam_key, "arrival")


def capture_time_s(cam_key, recv_ms, hw_ms, fallback):
    """When this frame was captured, on the shared timeline.

    Two sources, chosen per camera by the calibration. "hw" means the forwarder
    sends a real per-frame capture instant (librealsense global time), which is the
    ONLY thing that can follow a delay that moves — the Pi's frameset queue holds up
    to ~2s and its depth follows the load, so the same camera has measured 228ms and
    2.7s within a day. "arrival" means the camera has no such clock (RTSP carries
    none), leaving a constant offset as the best available.
    """
    if cam_clock(cam_key) == "hw" and hw_ms:
        return hw_ms / 1000.0 - cam_offset_ms(cam_key) / 1000.0
    if recv_ms:
        return recv_ms / 1000.0 - cam_offset_ms(cam_key) / 1000.0
    return fallback


# Delay each camera is OBSERVED to run behind, as a running mean of
# (now - capture time) per frame. Measured rather than taken from the stored
# offsets because a camera on its own hardware clock has an offset of ~0 while its
# frames are still genuinely seconds old — the D435 reads +0 and arrives 2.7s late.
# Using the offsets would have quietly stopped holding anything back for it.
OBS_DELAY = {}
OBS_DELAY_ALPHA = 0.02         # ~50 frames to settle; ignores single late frames


def note_observed_delay(cam_key: str, t_cap: float):
    d = max(0.0, (time.time() - t_cap) * 1000.0)
    with OFFSETS_LOCK:
        prev = OBS_DELAY.get(cam_key)
        OBS_DELAY[cam_key] = d if prev is None else (
            (1 - OBS_DELAY_ALPHA) * prev + OBS_DELAY_ALPHA * d)


def present_delay_ms() -> float:
    """How far behind live to present every camera, so they show one instant.

    The slowest camera sets the pace — holding anything back by less than its own
    delay could not bring the two into step. Derived fresh each time, so both a new
    calibration and a camera whose backlog grows change the pace without a restart.
    Zero until something has been observed, which leaves the old behaviour intact.
    """
    if not PRESENT_SYNC:
        return 0.0
    if PRESENT_DELAY_MS > 0:
        return PRESENT_DELAY_MS
    with OFFSETS_LOCK:
        observed = [v for v in OBS_DELAY.values() if v > 0]
    worst = max(observed) if observed else max(
        [v for v in all_offsets().values() if v > 0] or [0.0])
    if worst <= 0:
        return 0.0
    return min(worst + PRESENT_MARGIN_MS, PRESENT_MAX_DELAY_MS)


# Sampling for the calibration: the frame-difference energy of a downscaled gray
# frame, which is what a light switch spikes. Computed in the pose loop (the
# frame is already decoded there) and only while a calibration is armed.
# --- presenting every camera at the same instant --------------------------------
# Correcting timestamps fixes the ANALYSIS: fusion compares the right moments. It
# does nothing for the picture, because each camera's frame is still shown the
# moment it arrives — so the D455 displays the present while the security cameras
# display 3.5s ago, side by side, and a person crossing the room appears in
# different places in different panels.
#
# Nothing can make a late camera arrive sooner. On the Pi the D435's latency is a
# deliberate 60-frameset queue (~2s at 30fps) that exists to stop recordings
# dropping frames, and the NVR's buffer is not ours at all. The only way to show one
# instant is therefore to HOLD the fast cameras back to the slowest one, which is
# what this does: every camera's frame waits until `present_delay` after the moment
# it was captured, so what is on screen is one coherent slice of the past.
#
# Positions travel with their own frame, so the 3D map and the video stay in step.
PRESENT_SYNC = os.environ.get("SMARTROOM_PRESENT_SYNC", "1") != "0"
# Extra headroom over the largest measured offset, for arrival jitter: a frame that
# turns up later than its camera's average would otherwise miss its slot entirely.
PRESENT_MARGIN_MS = float(os.environ.get("SMARTROOM_PRESENT_MARGIN_MS", "400"))
# Override to pin the delay by hand; empty/0 means "derive it from the offsets".
PRESENT_DELAY_MS = float(os.environ.get("SMARTROOM_PRESENT_DELAY_MS", "0") or 0)
PRESENT_TICK_S = 0.02          # how often due frames are released
# Ceiling on the derived delay, so one wedged camera cannot push the whole view
# arbitrarily far into the past.
PRESENT_MAX_DELAY_MS = float(os.environ.get("SMARTROOM_PRESENT_MAX_DELAY_MS", "8000"))
# Bound the hold buffer per camera. 240 frames is ~16s at 15fps, far past any
# plausible delay, and caps memory at roughly 10MB of JPEG per camera.
PRESENT_MAX_FRAMES = int(os.environ.get("SMARTROOM_PRESENT_MAX_FRAMES", "240"))

# --- live audio ----------------------------------------------------------------
# ONE source, not a mix. Every Reolink camera offers an aac 16kHz track but only
# channel 1's mic is actually enabled -- measured over a real recording, ch1 is
# -40.9 dB mean with -14.9 dB peaks while 2, 3 and 4 sit at a flat -91.0 dB with mean
# equal to peak, which is digital silence rather than a quiet room. The RealSense have
# no microphone at all. So there is exactly one thing worth listening to.
#
# The audio is relayed, never decoded: it arrives already encoded from the forwarder
# and is fanned out to listeners as bytes, so adding sound costs no CPU here.
AUDIO_ON = os.environ.get("SMARTROOM_LIVE_AUDIO", "1") != "0"
AUDIO_SRC_CAM = os.environ.get("SMARTROOM_AUDIO_CAM", "camera_cam1_color")
# Hand trim for lip-sync, positive = hold the audio back further. The video delay is
# measured; the audio path's own delay is NOT (a light switch is silent), so this is
# the one number that has to be set by ear.
AUDIO_TRIM_MS = float(os.environ.get("SMARTROOM_AUDIO_TRIM_MS", "0") or 0)
# How much recent audio a starting segment is given (see AudioRelay.backlog). It
# needs to cover the trim, since that is exactly how far the sound for the first
# frames has already gone by; the extra second is slack.
AUDIO_BACKLOG_S = float(os.environ.get("SMARTROOM_AUDIO_BACKLOG_S", "0") or 0) or \
    (AUDIO_TRIM_MS / 1000.0 + 1.0)
AUDIO_PENDING_MAX = 512        # chunks held for their slot (~0.17s each)
AUDIO_OUT_MAX = 256            # released chunks kept for listeners that fall behind
AUDIO_SOURCE_STALE_S = 5.0

TIMING_SIZE = (160, 120)
TIMING_MAX_SAMPLES = 4000      # ~4 minutes at 15fps; a hard cap on the buffer
TIMING_DEFAULT_S = float(os.environ.get("SMARTROOM_TIMING_SECONDS", "25"))
TIMING_MAX_S = 180.0

# --- continuous segment recording -------------------------------------------
# Always-on archival of the live feed in fixed-length segments, written straight
# into the recordings tree so the existing API/website list them with no extra
# plumbing. Segments containing nobody are deleted on close — an empty room is
# the overwhelming majority of wall-clock time and is not worth the disk.
SEGMENT_ON = os.environ.get("SMARTROOM_SEGMENT", "1") != "0"
# Recording is ARMED BY HAND (a Record button) rather than always-on. Running
# continuously wrote a segment every 3 minutes forever -- 259 of them in one
# stretch -- so the archive filled with footage nobody asked for, and every one
# of those segments carried a copy of whatever calibration was current, which is
# how a six-day-stale pose kept resurrecting itself. Set
# SMARTROOM_SEGMENT_ALWAYS=1 to restore the old behaviour.
SEGMENT_ALWAYS = os.environ.get("SMARTROOM_SEGMENT_ALWAYS", "0") != "0"
# Hard ceiling on one armed recording, and the default when none is given. The
# button is in a browser tab; a tab can close, sleep, or lose the network, and
# none of those can be allowed to leave the encoder running indefinitely.
RECORD_MAX_S = float(os.environ.get("SMARTROOM_RECORD_MAX_S", "1800"))   # 30 min
# How long after a take ends to keep accepting frames CAPTURED inside it. The
# Reolink path is ~3.5s behind, so without this every one of their clips would lose
# its last few seconds while the RealSense clips kept theirs. Comfortably above the
# largest measured camera delay.
RECORD_LATE_GRACE_S = float(os.environ.get("SMARTROOM_RECORD_LATE_GRACE_S", "15"))
# Chunk length for the ALWAYS-ON mode only (SMARTROOM_SEGMENT_ALWAYS, off by
# default), where the alternative is one file that grows all day. It used to be
# applied to hand-started takes as well, which is why pressing Record once
# produced several "recordings" cut at 11:39, 11:42, 11:45 — the grid outlived
# the mode it was for.
SEGMENT_S = float(os.environ.get("SMARTROOM_SEGMENT_S", "180"))     # 3 minutes
# A started take is ONE clip. This is only the ceiling that stops a forgotten one
# becoming a single unbounded file, and it matches RECORD_MAX_S so the longest
# take the API will grant is still never split.
TAKE_MAX_S = float(os.environ.get("SMARTROOM_TAKE_MAX_S", "1800"))  # 30 minutes
# A couple of stray detections should not preserve an otherwise empty segment.
SEGMENT_MIN_PEOPLE_FRAMES = int(os.environ.get("SMARTROOM_SEGMENT_MIN_FRAMES", "15"))
# Encode on the GPU: two continuous libx264 streams alongside pose + AVA + a
# CPU-fallback ReID saturated the CPU and starved the HTTP server (requests
# queued behind a 190%-CPU process). NVENC is effectively free here.
SEGMENT_ENCODER = os.environ.get("SMARTROOM_SEGMENT_ENCODER", "h264_nvenc")
# Consecutive pose-predict failures before we give up and let systemd restart us.
# Two failure modes are real here: a TRANSIENT burst of CUDA "misaligned address"
# around startup that clears itself (~100 observed, then clean), and a STUCK one
# where the context is poisoned and every later kernel fails forever (10.5M
# errors over hours). The limit has to sit above the transient burst so a normal
# start isn't restarted, and far below "forever".
PREDICT_FAIL_LIMIT = int(os.environ.get("SMARTROOM_PREDICT_FAIL_LIMIT", "300"))
# Seconds a camera may go without producing an annotated frame *while the Pi is
# still pushing frames at it* before we treat the process as wedged.
#
# PREDICT_FAIL_LIMIT above only catches the failure mode that RAISES. The one
# that actually took the service down catches nothing: a CUDA call that never
# returns. Both pose threads sat in state R burning 100% CPU for 8.5 hours,
# GPU 0 pegged at 100% util with no progress, no exception, no log line — and
# systemd happily reported `active` the whole time because the HTTP server and
# the segment recorder were still ticking. A hang produces no error to count,
# so the only possible detector is absence of PROGRESS, checked from outside
# the wedged thread. A poisoned/hung CUDA context also cannot be recovered
# in-process (there is no CUDA API to reset it from the faulting process), so
# exiting for a fresh one is the fix, not a workaround.
STALL_S = float(os.environ.get("SMARTROOM_STALL_S", "60"))
# How long to wait for the AVA worker process to answer one clip before giving
# up on it and respawning. Generous: a forward is ~70ms, but the worker may be
# serving the other camera and queueing behind a cold CUDA context.
AVA_TIMEOUT_S = float(os.environ.get("SMARTROOM_AVA_TIMEOUT_S", "20"))
# ReID sits on the pose loop's critical path (one round trip per embedded
# frame), so its deadline is short: dropping an embed only costs geometry-only
# identity for that frame, whereas waiting costs every camera its frame rate.
REID_TIMEOUT_S = float(os.environ.get("SMARTROOM_REID_TIMEOUT_S", "5"))
# "cpu" until onnxruntime-gpu is installed. Safe to point at a GPU now that the
# encoder runs in its own process.
REID_DEVICE = os.environ.get("SMARTROOM_REID_DEVICE", "cpu")


def _day_dir(root: Path, when: dt.datetime) -> Path:
    """day_NN_YYYY-MM-DD, reusing today's folder and continuing the NN sequence."""
    date = when.strftime("%Y-%m-%d")
    best = None
    for d in (root.iterdir() if root.exists() else []):
        m = re.match(r"day_(\d+)_(\d{4}-\d{2}-\d{2})$", d.name)
        if not m:
            continue
        if m.group(2) == date:
            return d
        best = max(best or 0, int(m.group(1)))
    return root / f"day_{(best or 0) + 1:02d}_{date}"


class RecordControl:
    """Whether recording is armed, shared by every camera.

    ONE flag for all of them, not one per camera: segments are named after the
    wall-clock boundary precisely so both cameras land in the same recording
    folder, and arming them separately would produce takes that no longer pair.

    `until` supports a bounded recording (start with a duration) so an armed
    recorder cannot be left running by a closed browser tab.
    """

    def __init__(self, always: bool):
        self.lock = threading.Lock()
        self.always = always
        self.armed = always
        self.since = time.time() if always else None
        self.until = None
        self.label = None
        self.window = None     # (start, end) of the most recently closed take

    def start(self, seconds=None, label=None):
        with self.lock:
            self.armed = True
            self.since = time.time()
            self.until = (self.since + float(seconds)) if seconds else None
            self.label = label
            return self._state_locked()

    def stop(self):
        with self.lock:
            if self.always:      # configured always-on; a stop would be a lie
                return self._state_locked()
            self._disarm_locked()
            return self._state_locked()

    def is_armed(self):
        """True while recording. Expiry is checked HERE rather than by a timer
        thread, so a bounded recording ends even if nothing else is running."""
        with self.lock:
            if self.armed and self.until is not None and time.time() >= self.until:
                self._disarm_locked()
            return self.armed

    def _disarm_locked(self):
        # Remember the window that just closed. A camera running seconds behind is
        # still delivering frames CAPTURED inside it, and dropping those would clip
        # the tail off its take -- the Reolink cameras would lose their last ~3.5s
        # of every recording while the RealSense kept theirs.
        if self.since is not None:
            self.window = (self.since, time.time())
        self.armed = False
        self.since = self.until = self.label = None

    def accepts(self, t_capture):
        """Should a frame CAPTURED at `t_capture` go into the recording?

        Judged on capture time, not arrival: that is what makes every camera's clip
        cover the same real interval regardless of how late its frames turn up.
        """
        if self.is_armed():
            with self.lock:
                return self.since is None or t_capture >= self.since
        with self.lock:
            if not self.window:
                return False
            start, end = self.window
            # Give up on the tail once no plausible delay could still deliver it.
            if time.time() - end > RECORD_LATE_GRACE_S:
                return False
            return start <= t_capture <= end

    def segment(self, t_capture):
        """Which segment a frame CAPTURED at `t_capture` belongs to: (key, start).

        A take someone started by hand is ONE segment for its whole length. The
        3-minute wall-clock chunking exists for the always-on mode, where the
        alternative is a single file that grows all day; applied to a hand-started
        take it cuts wherever the boundary happens to fall, so pressing Record at
        11:44:58 gave a 2-second clip and then a second, separate recording — one
        press, several "recordings", which is not what anyone means by Record.

        The key is derived from values every camera reads from this one object, so
        their clips still land in the SAME folder without coordinating.
        """
        with self.lock:
            start = self.since if self.armed else (self.window[0] if self.window else None)
        if self.always or start is None:
            idx = int(t_capture // SEGMENT_S)
            return ("wall", idx), idx * SEGMENT_S
        # A ceiling, not a rhythm: only a take running past TAKE_MAX_S is split.
        span = int(max(0.0, t_capture - start) // TAKE_MAX_S) if TAKE_MAX_S > 0 else 0
        return ("take", round(start, 3), span), start + span * TAKE_MAX_S

    def state(self):
        with self.lock:
            return self._state_locked()

    def _state_locked(self):
        now = time.time()
        return {"recording": self.armed, "always": self.always,
                "elapsedS": round(now - self.since, 1) if self.since else None,
                "remainingS": (round(self.until - now, 1)
                              if self.armed and self.until else None),
                "label": self.label}


RECORD = RecordControl(SEGMENT_ALWAYS)


def _segment_mode_note() -> str:
    """Recording mode for the startup banner.

    A plain function, not an inline conditional inside the banner's f-string: a
    newline inside f-string braces is only valid from Python 3.12, so the
    expression form parsed on this laptop (3.13) and was a SyntaxError on the
    server's interpreter, crash-looping the service.
    """
    if not SEGMENT_ON:
        return "off"
    if SEGMENT_ALWAYS:
        return "ALWAYS ON (%gs)" % SEGMENT_S
    return "on demand (one clip per take, split past %gs, POST /record/start)" % TAKE_MAX_S


class AudioRelay:
    """Encoded audio from one camera, held to match the video, fanned out to browsers.

    The video is presented `present_delay` after capture. Audio arriving live would
    therefore run AHEAD of the picture by that much, so it is held too — but by less,
    because the audio has already spent the NVR's delay getting here. The hold is the
    difference: what the video waits for, minus what this camera's frames already
    waited. Both come out of the same measurement, so re-calibrating fixes both.

    Bytes are never decoded. Chunks are timestamped ON ARRIVAL rather than trusting
    the forwarder's clock, which keeps this independent of that host's time.
    """

    def __init__(self):
        self.cond = threading.Condition()
        self.pending = deque()       # (arrival_ms, data) not yet due
        self.out = deque(maxlen=AUDIO_OUT_MAX)   # (seq, data) released
        self.seq = 0
        self.last_push = 0.0
        self.listeners = 0
        self.content_type = "audio/mpeg"
        self.bytes_in = 0
        # Recorders writing this audio into their segment. They tap the RAW stream,
        # not the delayed one the browser hears: a recording is assembled on the
        # capture timeline, and the presentation delay is a viewing concern.
        self.taps = {}          # id -> callable(arrival_ms, data)
        # A few seconds of the recent past, so a segment that opens now can be given
        # the sound that belongs to its first frames. It has to: the audio is dated
        # AUDIO_TRIM_MS later than it arrives (the audio path is faster than the
        # video path), so at the instant recording starts, the sound covering the
        # take's first ~2.8s has already gone by. Without this every recording began
        # with that much silence.
        self.backlog = deque()  # (arrival_ms, data), trimmed to AUDIO_BACKLOG_S

    def hold_ms(self):
        """How long to sit on an arriving chunk before letting it out."""
        delay = present_delay_ms()
        if delay <= 0:
            return max(0.0, AUDIO_TRIM_MS)
        return max(0.0, delay - cam_offset_ms(AUDIO_SRC_CAM) + AUDIO_TRIM_MS)

    def add_tap(self, key, fn, backlog_s=0.0):
        """Register a recorder. `backlog_s` replays that much recent audio into it
        first, so its clip starts with sound rather than with the trim's worth of
        silence."""
        with self.cond:
            self.taps[key] = fn
            past = ([c for c in self.backlog
                     if c[0] >= time.time() * 1000.0 - backlog_s * 1000.0]
                    if backlog_s > 0 else [])
        for arrival_ms, data in past:      # outside the lock, like push's fan-out
            try:
                fn(arrival_ms, data)
            except Exception as exc:  # noqa: BLE001
                print(f"[live] audio backlog replay failed: {exc}", flush=True)

    def remove_tap(self, key):
        with self.cond:
            self.taps.pop(key, None)

    def push(self, data, content_type=None):
        now = time.time() * 1000.0
        with self.cond:
            if content_type:
                self.content_type = content_type
            self.last_push = now
            self.bytes_in += len(data)
            self.pending.append((now, data))
            while len(self.pending) > AUDIO_PENDING_MAX:
                self.pending.popleft()
            self.backlog.append((now, data))
            horizon = now - AUDIO_BACKLOG_S * 1000.0
            while self.backlog and self.backlog[0][0] < horizon:
                self.backlog.popleft()
            taps = list(self.taps.values())
        # Outside the lock: a recorder's disk write must never stall the relay, and
        # a tap that throws must not take the audio down with it.
        for fn in taps:
            try:
                fn(now, data)
            except Exception as exc:  # noqa: BLE001
                print(f"[live] audio tap failed: {exc}", flush=True)

    def release_due(self):
        cutoff = time.time() * 1000.0 - self.hold_ms()
        with self.cond:
            n = 0
            while self.pending and self.pending[0][0] <= cutoff:
                _t, data = self.pending.popleft()
                self.seq += 1
                self.out.append((self.seq, data))
                n += 1
            if n:
                self.cond.notify_all()
            return n

    def follow(self, after_seq, timeout=5.0):
        """Chunks released after `after_seq`, waiting if there are none yet.

        A listener that falls further behind than the buffer is resynced to the
        newest chunk rather than fed stale audio: for live sound, a gap is better
        than drifting permanently behind the picture.
        """
        with self.cond:
            if not self.out or self.out[-1][0] <= after_seq:
                self.cond.wait(timeout=timeout)
            if not self.out:
                return after_seq, []
            oldest = self.out[0][0]
            if after_seq < oldest - 1:
                after_seq = oldest - 1
            fresh = [(s, d) for s, d in self.out if s > after_seq]
            if not fresh:
                return after_seq, []
            return fresh[-1][0], [d for _s, d in fresh]

    def live(self):
        return (time.time() * 1000.0 - self.last_push) / 1000.0 < AUDIO_SOURCE_STALE_S

    def state(self):
        with self.cond:
            return {"available": bool(self.last_push) and self.live(),
                    "sourceCam": AUDIO_SRC_CAM,
                    "contentType": self.content_type,
                    "listeners": self.listeners,
                    "holdMs": round(self.hold_ms(), 1),
                    "trimMs": AUDIO_TRIM_MS,
                    "kbReceived": round(self.bytes_in / 1024.0, 1)}


AUDIO = AudioRelay()


class TimingCalibration:
    """The lights on/off measurement, shared by every camera.

    While armed, each camera's pose loop appends (arrival_ms, frame-difference
    energy) here. When the window closes, the series are cross-correlated against
    a reference and the offsets are written and applied.

    ONE window for all cameras, deliberately: the whole measurement is a
    comparison between cameras, so they must be watching the same light switch at
    the same time. Cameras that see nothing are reported and left at 0 rather
    than guessed at.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.armed = False
        self.until = 0.0
        self.reference = None
        self.series = {}          # cam_key -> [(arrival_ms, energy)]
        self.result = None        # last solve output (also written to disk)
        self.error = None
        self.started = None
        self.diagnostics = None   # per-camera samples/brightness from the last run

    def _reset_run(self):
        self.series = {}
        self.result = None
        self.error = None
        self.diagnostics = None

    def start(self, seconds, reference=None):
        # Floor above timing_sync's minimum window: a shorter run could only ever
        # be rejected by the solve, so refusing to arm it is kinder than letting
        # someone flip the lights for nothing.
        floor_s = timing_sync.MIN_OVERLAP_MS / 1000.0 + 2.0
        seconds = max(floor_s, min(float(seconds), TIMING_MAX_S))
        with self.lock:
            if self.armed:
                return False, self.state_locked()
            self.armed = True
            self.started = time.time()
            self.until = self.started + seconds
            self.reference = reference or None
            self._reset_run()
            return True, self.state_locked()

    def sample(self, cam_key, arrival_ms, energy, brightness=None, hw_ms=0.0):
        """Called from the pose loop for every frame while armed. Cheap: one
        bool test when disarmed, one append when armed.

        hw_ms is the forwarder's own timestamp for this frame. For a RealSense it is
        librealsense global time — a genuine per-frame capture instant, read off the
        frameset AFTER it leaves the Pi's queue, so it tracks a varying backlog that
        no constant offset can. For a Reolink it is merely when the forwarding host
        received the frame, which excludes the NVR's delay. Recording it lets the
        solve work out, per camera, which of the two it is looking at.
        """
        if not self.armed:
            return
        with self.lock:
            if not self.armed:
                return
            rows = self.series.setdefault(cam_key, [])
            if len(rows) < TIMING_MAX_SAMPLES:
                rows.append((float(arrival_ms), float(energy),
                             float(brightness if brightness is not None else 0.0),
                             float(hw_ms or 0.0)))

    def expired(self):
        return self.armed and time.time() >= self.until

    def finish(self, cam_keys):
        """Solve, store and apply. Returns the result dict (or None on failure).

        Cameras that never delivered a frame are passed in as empty series so the
        result names them explicitly — "camera_cam3_color: not streaming" is a
        far more useful answer than that camera silently missing from the output.
        """
        with self.lock:
            if not self.armed:
                return None
            self.armed = False
            series = {k: list(v) for k, v in self.series.items()}
            reference = self.reference
            self.series = {}
        for key in cam_keys:
            series.setdefault(key, [])
        # Per-camera numbers recorded BEFORE the solve, and kept whether it
        # succeeds or not. On the "lights never changed" path the solve raises and
        # its message rounds to whole gray levels, which cannot distinguish a
        # genuinely static room (0.4) from brightness never being sampled at all
        # (exactly 0.00) — and the second would silently block every legitimate
        # run. These make that difference visible instead of a guess.
        diagnostics = {
            key: {"samples": len(rows),
                  "brightness_swing": round(timing_sync.brightness_swing(rows), 2),
                  "median_brightness": round(
                      float(np.median([r[2] for r in rows])) if rows else 0.0, 1),
                  # median transport delay implied by the forwarder's own clock
                  "hw_delay_ms": round(float(np.median(
                      [r[0] - r[3] for r in rows if len(r) > 3 and r[3]])), 1)
                  if any(len(r) > 3 and r[3] for r in rows) else None}
            for key, rows in sorted(series.items())
        }
        with self.lock:
            self.diagnostics = diagnostics
        # Keep the raw series of the LAST run. A rejected camera cannot be
        # diagnosed from a summary line, and the alternative is asking someone to
        # stand at the light switch again for every hypothesis. One run overwrites
        # the previous — this is a scratch pad, not an archive.
        try:
            raw = timing_sync.timing_path().with_name("live_timing_last_run.json")
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(json.dumps(
                {"series": {k: [[round(r[0], 1), round(r[1], 4), round(r[2], 2),
                                 round(r[3], 1)] for r in v]
                            for k, v in series.items()}},
                separators=(",", ":")))
        except (OSError, ValueError) as exc:
            print(f"[live] could not save timing raw series: {exc}", flush=True)
        try:
            result = timing_sync.solve(series, reference=reference)
            path = timing_sync.save(result)
            applied = reload_offsets(cam_keys)
            result["applied_ms"] = {k: applied.get(k, 0.0) for k in cam_keys}
            result["path"] = str(path)
            print(f"[live] timing calibration: {timing_sync.summary_line(result)}",
                  flush=True)
            with self.lock:
                self.result, self.error = result, None
            return result
        except Exception as exc:  # noqa: BLE001 — reported through /timing/status
            print(f"[live] timing calibration FAILED: {exc}", flush=True)
            with self.lock:
                self.result, self.error = None, str(exc)
            return None

    def cancel(self):
        with self.lock:
            self.armed = False
            self.series = {}
            self.error = "cancelled"
            return self.state_locked()

    def state(self):
        with self.lock:
            return self.state_locked()

    def state_locked(self):
        now = time.time()
        return {
            "running": self.armed,
            "remainingS": round(max(0.0, self.until - now), 1) if self.armed else None,
            "elapsedS": round(now - self.started, 1) if self.started else None,
            "samples": {k: len(v) for k, v in self.series.items()},
            "result": self.result,
            "error": self.error,
            "diagnostics": self.diagnostics,
            "offsetsMs": all_offsets(),
            "instructions": ("Flip the room lights fully off and back on 3-4 times, "
                             "a couple of seconds apart, while this runs."),
        }


TIMING = TimingCalibration()


def _timing_note(measured: dict) -> str:
    """Per-camera offsets for the startup banner. Says so plainly when there are
    none: an uncalibrated system compares cameras on raw arrival times, and that
    is worth seeing in the log rather than inferring from silence."""
    if not measured:
        return (f"all cameras at 0 ms (no calibration at {timing_sync.timing_path()}"
                " — POST /timing/start and flip the lights)")
    return ", ".join(f"{k} {v:+.0f}ms" for k, v in sorted(measured.items()))


class SegmentRecorder:
    """Encodes the incoming JPEG stream to fixed-length mp4 segments.

    Frames are handed off to a writer thread so a slow encoder can never stall
    inference. Segments align to wall-clock boundaries, so both cameras land in
    the SAME recording folder without needing to coordinate.
    """

    def __init__(self, cam_key: str, root: Path, stream_meta: dict, node: str,
                 room_frame: dict | None = None, geom: dict | None = None):
        self.cam, self.root, self.node = cam_key, root, node
        self.stream_meta = stream_meta or {}
        self.room_frame = room_frame
        self.geom = geom or {}
        self.q = deque()
        self.cond = threading.Condition()
        self.proc = None
        self.dir = None
        self.idx = None          # RECORD.segment() key of the open segment
        self.frames = 0
        self.people_frames = 0
        self.started = None
        self.rows = []           # (frame_no, hw_ts, sync_ms)
        self.geo_rows = []       # (frame_no, [{id, px, room, src}]) — depth-measured
        self.kept = self.dropped = 0
        # Room audio, muxed into this camera's mp4 at close. Only the camera the
        # audio actually comes from records it: it is ONE room microphone, and
        # copying it onto every camera would make the synced player overlap five
        # identical tracks (an echo) and imply each camera has its own mic.
        self.wants_audio = AUDIO_ON and cam_key == AUDIO_SRC_CAM
        self.audio_fh = None
        self.audio_path = None
        self.audio_t0 = None        # capture time of the first chunk in this segment
        self.audio_last = None      # ...and of the most recent, for the tail wait
        self.audio_bytes = 0
        self.audio_info = None
        threading.Thread(target=self._run, daemon=True).start()

    def add(self, jpeg: bytes, hw_ts: float, positions, sync_ms: float = 0.0,
            people: int = None):
        """positions: this frame's people as [{id, px:[u,v], room:[x,z], src}] —
        the depth-measured room positions the live map already computed. Saving
        them makes an RGB-only segment as localizable as a depth recording; the
        offline pass can't (no depth) and would otherwise floor-ray it into the
        walls.

        sync_ms is this frame's time on the SHARED timeline (arrival here, less
        this camera's measured delay). hw_ts is kept alongside it untouched: it is
        whatever the forwarding host stamped, which for a RealSense is a real
        sensor clock and for a Reolink is that host's wall clock — two different
        clocks, which is exactly why a segment needs a third column that is one.

        `people` is how many people this frame DETECTED, which is not the same as
        how many it published. A spatially muted camera emits no positions by
        design, and counting `positions` meant its segments always looked empty and
        were discarded on close — so with SMARTROOM_SPATIAL_ONLY set, five of six
        cameras silently recorded nothing at all, while their own startup line
        promised "recording continues". Whether footage is worth keeping is a
        question about people being present, not about which camera was elected to
        report where they were.
        """
        if not RECORD.accepts(sync_ms / 1000.0 if sync_ms else time.time()):
            return                         # not recording — drop it here, cheaply
        n_people = len(positions) if people is None else people
        with self.cond:
            if len(self.q) < 240:          # ~8s at 30fps; never block inference
                self.q.append((jpeg, hw_ts, positions, sync_ms, n_people))
                self.cond.notify()

    def _audio_chunk(self, arrival_ms, data):
        """Raw audio from the relay, appended to this segment's sidecar.

        The chunk's CAPTURE time is its arrival less the same delay this camera's
        pictures took — audio and video come down one path from the NVR, so the
        offset largely cancels and what is left is their real skew.

        Except it does not cancel entirely: the measured 3.4 s belongs to the
        camera's VIDEO path (the NVR's encoder and pre-buffer), and the audio
        comes down faster, so subtracting the whole thing dates the sound too
        early and it plays ahead of the picture. AUDIO_TRIM_MS is that
        difference, measured by ear. It ADDS here, exactly as it adds to the live
        hold: in both places a bigger trim means the sound is held back further.
        It used to subtract, which meant one number could not be right for both —
        set to fix the live feed it would have doubled the error in recordings.
        """
        fh = self.audio_fh
        if fh is None:
            return
        t_cap = (arrival_ms - cam_offset_ms(AUDIO_SRC_CAM) + AUDIO_TRIM_MS) / 1000.0
        if self.audio_t0 is None:
            self.audio_t0 = t_cap
        self.audio_last = t_cap
        fh.write(data)
        self.audio_bytes += len(data)

    def _await_audio_tail(self):
        """Wait for the sound that belongs to the video we just finished.

        Dating the audio TRIM ms later than its arrival means that when the last
        picture is in, the sound for the take's final seconds has not been
        recorded yet — it is still crossing the network. Closing immediately and
        muxing left a clip whose audio ran out 3.1 s before its video, which
        -shortest then made look deliberate.

        Bounded, and it stops the moment the sound catches up to the last frame,
        so the usual cost is the trim itself (~2.8 s) and never more than that
        plus a second.
        """
        if not self.wants_audio or self.audio_fh is None or not self.rows:
            return
        target = (self.rows[-1][2] or 0) / 1000.0
        if not target:
            return
        deadline = time.time() + (AUDIO_TRIM_MS / 1000.0) + 1.0
        while time.time() < deadline:
            if (self.audio_last or 0) >= target:
                return
            time.sleep(0.1)

    def _audio_open(self):
        if not self.wants_audio or self.dir is None:
            return
        self.audio_path = self.dir / f".{self.cam}_audio.mp3"
        self.audio_t0 = None
        self.audio_last = None
        self.audio_bytes = 0
        try:
            self.audio_fh = open(self.audio_path, "wb")
        except OSError as exc:
            print(f"[live] {self.cam}: cannot open audio sidecar: {exc}", flush=True)
            self.audio_fh = None
            return
        AUDIO.add_tap(id(self), self._audio_chunk, backlog_s=AUDIO_BACKLOG_S)

    def _real_fps(self):
        """Frames per second this segment actually delivered, from capture times.

        The encoder is fed a blind CFR 30 because the true rate is not known until
        the segment ends. Every other recorder in this system (capture.py,
        realsense_depth_page.py) retimes its container to the measured rate on
        close, and live segments were the one exception -- so 25s of a ~10fps
        camera became an 8.4s container.
        """
        stamps = [r[2] / 1000.0 for r in self.rows if r[2]]
        if len(stamps) < 2:
            return None
        span = stamps[-1] - stamps[0]
        if span <= 0:
            return None
        return (len(stamps) - 1) / span

    def _retime_scale(self):
        """Factor to stretch this segment's container onto real time, or None.

        Applied ONLY to the clip that carries audio, because that clip has to hold
        two tracks on one timeline and audio is inherently real-time. The others
        keep their blind CFR-30 container: the mirror derives each clip's playback
        rate from its own duration and CSV span independently, so a mix of retimed
        and non-retimed clips plays correctly either way, and leaving them alone
        keeps this change off the playback path for four of five cameras.
        """
        fps = self._real_fps()
        if not fps or fps <= 0:
            return None
        scale = 30.0 / fps
        # Below a few percent it is noise, and an itsscale of ~1 only costs a remux.
        return scale if abs(scale - 1.0) > 0.03 else None

    def _audio_finish(self, mp4_path, video_t0):
        """Mux the sidecar into the finished mp4. Returns the audio info, or None.

        A stream copy, so this costs no encoding. Muxed AFTER the fact rather than
        piped into the live ffmpeg on a second input: two pipes into one process
        deadlock the moment one of them stalls, and audio arriving over a network
        from another host is exactly the thing that stalls.
        """
        AUDIO.remove_tap(id(self))
        fh, path = self.audio_fh, self.audio_path
        self.audio_fh = None
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
        if not path or not path.exists():
            return None
        # A sliver of audio is not worth a remux, and mp3 needs a few frames before
        # it decodes at all.
        if self.audio_bytes < 4096 or self.audio_t0 is None or video_t0 is None:
            path.unlink(missing_ok=True)
            return None
        skew = self.audio_t0 - video_t0
        merged = mp4_path.with_suffix(".muxed.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        # Stretch the video's fake CFR-30 timeline onto real time BEFORE muxing.
        # Without this the container is ~3x short for a 10fps camera, and -shortest
        # then truncates real-time audio to the container -- measured: 26s of sound
        # cut to 8.2s. Retiming is also what makes the two tracks line up at all,
        # since audio is inherently real-time and this video was not.
        scale = self._retime_scale()
        if scale:
            cmd += ["-itsscale", f"{scale:.6f}"]
        cmd += ["-i", str(mp4_path)]
        # Positive skew = the audio starts later than the video, so delay it to match.
        if abs(skew) > 0.02:
            cmd += ["-itsoffset", f"{skew:.3f}"]
        cmd += ["-i", str(path), "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "copy", "-shortest",
                "-movflags", "+faststart", str(merged)]
        try:
            r = subprocess.run(cmd, timeout=180, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[live] {self.cam}: audio mux failed: {exc}", flush=True)
            path.unlink(missing_ok=True)
            return None
        if r.returncode != 0 or not merged.exists() or merged.stat().st_size == 0:
            err = (r.stderr or b"").decode(errors="replace").strip().splitlines()
            print(f"[live] {self.cam}: audio mux failed: {err[-1] if err else r.returncode}",
                  flush=True)
            merged.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            return None
        merged.replace(mp4_path)      # the clip now HAS the audio; no separate file
        path.unlink(missing_ok=True)
        return {"codec": "mp3", "source": AUDIO_SRC_CAM, "scope": "room",
                "muxed_with_retimed_video": bool(self._retime_scale()),
                "skew_ms": round(skew * 1000.0, 1),
                "bytes": self.audio_bytes,
                "note": "one room microphone; only this camera's clip carries it"}

    def _cap_now(self):
        """"Now" on the shared capture timeline for THIS camera.

        A camera delivering 3.5s late is, right now, producing frames captured 3.5s
        ago — so its idle bookkeeping (when to rotate, when to give up waiting) has
        to run on that clock too, or it would rotate a segment before the frames
        belonging to it had arrived.

        Uses the OBSERVED lag rather than the stored offset: a camera on its own
        hardware clock has an offset of ~0 while its frames are still seconds old,
        and the offset would put this clock 2.7s into that camera's future.
        """
        with OFFSETS_LOCK:
            lag = OBS_DELAY.get(self.cam)
        if lag is None:
            lag = cam_offset_ms(self.cam)
        return time.time() - lag / 1000.0

    def _run(self):
        while True:
            with self.cond:
                while not self.q:
                    self.cond.wait(timeout=1.0)
                    if not self.q and self.proc:
                        # Recording stopped: close NOW rather than leaving the mp4
                        # open until some later frame happens to rotate it. Until
                        # _close runs, the moov atom is missing and the clip is
                        # unplayable — a Stop button that left the take unreadable
                        # for minutes would look like it had lost the recording.
                        # accepts(), not is_armed(): a late camera's tail frames are
                        # still owed to the take that just ended.
                        if not RECORD.accepts(self._cap_now()):
                            self._close()
                            self.proc = self.idx = None
                        else:
                            key, started = RECORD.segment(self._cap_now())
                            if key != self.idx:
                                self._rotate(key, started)
                item = self.q.popleft()
            jpeg, hw_ts, positions, sync_ms, n_people = item
            # Segment membership follows the frame's CAPTURE time, so every camera's
            # clip for a given boundary covers the same real interval. Cutting on
            # arrival instead meant a Reolink clip named rec_..._120000 actually held
            # 11:59:56.5–12:02:56.5 while the D455's held 12:00:00–12:03:00 — the
            # same folder name over two different 3.5s-shifted spans, which is
            # exactly what breaks paired playback.
            t_cap = sync_ms / 1000.0 if sync_ms else time.time()
            key, started = RECORD.segment(t_cap)
            if self.proc is None or key != self.idx:
                self._rotate(key, started)
            try:
                self.proc.stdin.write(jpeg)
                self.frames += 1
                if self.video_t0 is None and sync_ms:
                    self.video_t0 = sync_ms / 1000.0
                self.rows.append((self.frames, hw_ts, sync_ms))
                if n_people:
                    self.people_frames += 1
                if positions:
                    self.geo_rows.append((self.frames, positions))
            except (BrokenPipeError, OSError) as exc:
                print(f"[live] {self.cam}: segment write failed: {exc}", flush=True)
                self.proc = None

    def _rotate(self, key, started: float):
        self._close()
        self.idx, self.frames, self.people_frames, self.rows = key, 0, 0, []
        self.geo_rows = []
        self.video_t0 = None      # capture time of this segment's first frame
        # Name the segment after the segment's own START, not the instant we
        # happened to rotate. The two cameras rotate a fraction of a second
        # apart, and datetime.now() straddling a second boundary produced
        # rec_..._162530 and rec_..._162531 — one take per camera instead of
        # one take with both. `started` comes from RECORD, so it is identical
        # for every camera.
        self.started = dt.datetime.fromtimestamp(started).astimezone()
        rec = "rec_" + self.started.strftime("%Y%m%d_%H%M%S")
        self.dir = _day_dir(self.root, self.started) / rec / "streams" / "cam2"
        self.dir.mkdir(parents=True, exist_ok=True)
        out = self.dir / f"{self.cam}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe",
               "-vcodec", "mjpeg", "-r", "30", "-i", "-",
               "-c:v", SEGMENT_ENCODER, "-pix_fmt", "yuv420p"]
        cmd += (["-cq", "26", "-preset", "p5"] if "nvenc" in SEGMENT_ENCODER
                else ["-crf", "26", "-preset", "veryfast"])
        # +faststart moves the moov atom to the FRONT on close. Without it the
        # index lands at the end of the file and browsers/WebCodecs cannot begin
        # playback over HTTP — the clips download fine but show no video.
        cmd += ["-movflags", "+faststart", str(out)]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
        except OSError as exc:
            print(f"[live] {self.cam}: cannot start ffmpeg: {exc}", flush=True)
            self.proc = None
        self._audio_open()

    def _close(self):
        """Finish the current segment: keep it only if somebody was in it."""
        if self.dir is None:
            return
        # Cleanup must run even when the encoder is already gone. This used to
        # bail on `self.proc is None`, which is exactly the state left behind when
        # ffmpeg dies mid-segment (the write handler sets proc = None) -- so the
        # zero-byte mp4 that ffmpeg's own -y had already created was never
        # removed, and neither was its directory. The archive then carried a
        # recording with no video, no metadata and no timestamps, which the
        # listing still advertises and whose thumbnail 502s.
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try: self.proc.kill()
                except Exception: pass
            self.proc = None
        rec_dir = self.dir.parent.parent          # .../rec_x
        mp4 = self.dir / f"{self.cam}.mp4"
        # No frames at all is also a discard: a segment that was opened and then
        # interrupted has an empty mp4, which is worse than no recording.
        #
        # The audio-bearing camera is exempt from the people test. It carries the
        # room's ONLY microphone, so discarding it throws the take's sound away —
        # and cam1, which happens to be that camera, localizes nobody at all (it
        # sits 2.6 m up and its floor-ray overshoots MAX_RAY_REACH_MM), so the
        # test discarded it every single time and every recording came out mute.
        # An empty room is still worth hearing.
        keep_for_audio = self.wants_audio and self.audio_bytes > 0
        if self.frames == 0 or (self.people_frames < SEGMENT_MIN_PEOPLE_FRAMES
                                and not keep_for_audio):
            # nobody in it — discard, and remove the folder if the other camera
            # did not keep anything either.
            mp4.unlink(missing_ok=True)
            (self.dir / f"{self.cam}_timestamps.csv").unlink(missing_ok=True)
            # the tap holds a reference to this recorder; a discarded segment must
            # release it or the next one writes into a closed file
            AUDIO.remove_tap(id(self))
            if self.audio_fh is not None:
                try: self.audio_fh.close()
                except OSError: pass
                self.audio_fh = None
            if self.audio_path:
                self.audio_path.unlink(missing_ok=True)
            self.dropped += 1
            # tidy up: cam dir -> streams dir -> rec dir. Stops as soon as one is
            # non-empty (i.e. the other camera kept its clip).
            for d in (self.dir, self.dir.parent, rec_dir):
                try: d.rmdir()
                except OSError: break
            return
        dur = max(1.0, self.frames / 30.0)
        self._await_audio_tail()   # the sound for the last frames is still in flight
        # Before the metadata, so it can record what the clip actually contains.
        self.audio_info = self._audio_finish(mp4, self.video_t0)
        if self.audio_info and self.audio_info.get("muxed_with_retimed_video"):
            # That clip's container now spans real time, so its declared duration
            # has to as well -- anything converting a sidecar time into real time
            # reads this.
            real_fps = self._real_fps()
            if real_fps:
                dur = max(1.0, self.frames / real_fps)
        with open(self.dir / f"{self.cam}_timestamps.csv", "w") as fh:
            # sync_ms is the cross-camera column for these segments: one clock
            # (this server's arrival) with each camera's measured delay already
            # removed. hw_timestamp_ms stays exactly as the forwarder sent it —
            # raw, and only comparable between cameras that share a clock.
            # Consumers pick columns by NAME, so appending one is safe.
            fh.write("frame,hw_timestamp_ms,sync_ms\n")
            for n, ts, sync in self.rows:
                fh.write(f"{n},{ts:.3f},{sync:.3f}\n")
        self._write_metadata(dur)
        self._write_geo()
        self.kept += 1
        # Trigger the analysis pass (smartroom-analyze.path watches this
        # sentinel). It was only ever touched by the Pi's uploader, so segments
        # written locally by this recorder were never analyzed — they had no
        # pose, location or action sidecars at all. Touch it exactly as the
        # uploader does: the path unit deliberately watches the sentinel rather
        # than the recordings tree, so analysis output can't re-trigger itself.
        try:
            (self.root / ".last_upload").touch()
        except OSError as exc:
            print(f"[live] {self.cam}: cannot touch analysis sentinel: {exc}", flush=True)
        print(f"[live] {self.cam}: kept {rec_dir.name} "
              f"({self.frames} frames, {self.people_frames} with people)", flush=True)

    def _write_metadata(self, dur: float):
        """Merge this camera's stream into the shared metadata.json (both cameras
        write the same file, so read-modify-write under a lock)."""
        path = self.dir / "metadata.json"
        with _SEG_META_LOCK:
            try:
                meta = json.loads(path.read_text())
            except (OSError, ValueError):
                meta = {}
            meta.setdefault("recording_id", self.dir.parent.parent.name)
            meta.setdefault("node", self.node)
            meta.setdefault("start_time", self.started.isoformat())
            meta.setdefault("source", "live_segment_recorder")
            # carry the room frame so a segment is a self-contained, calibrated
            # recording (tag height lives here; without it geometry cannot load)
            if self.room_frame and "room_frame" not in meta:
                meta["room_frame"] = self.room_frame
            # Top-level duration is the RECORDING's extent (the longest stream).
            # It is NOT this camera's, and consumers must not treat it as such:
            # the cameras deliver different frame counts, so their containers
            # differ — a segment where the D455 kept 2389 frames and the D435
            # 2302 is 79.6 s of container for one and 76.7 s for the other.
            # Anything converting a sidecar time into real time needs the
            # per-stream value below; using this one stretched the mirror's 3D
            # scene by up to 114 s on the worst clip in the archive.
            meta["duration_seconds"] = round(max(dur, meta.get("duration_seconds", 0)), 2)
            entry = dict(self.stream_meta)        # calibration + extrinsics
            entry.update({"path": f"{self.cam}.mp4",
                          "start_time": self.started.isoformat(),
                          # THIS stream's container duration: frames at the blind
                          # CFR the encoder was fed, which is what the sidecars'
                          # `t` values are expressed in.
                          "duration_seconds": round(dur, 3),
                          "frame_count": self.frames,
                          "people_frames": self.people_frames,
                          # What the CSV's sync_ms column means, and how far this
                          # camera's delivery path was found to lag the reference.
                          # 0 with no calibration on file — see timing_sync.py.
                          "time_offset_ms": round(cam_offset_ms(self.cam), 1),
                          # Segment boundaries are capture-aligned, so this clip and
                          # every other camera's clip in this folder cover the same
                          # real interval even though they arrived seconds apart.
                          "segment_aligned_on": "capture time (arrival - time_offset_ms)",
                          "sync": "match frames across cameras on sync_ms"})
            # Recorded, not assumed: metadata claiming audio on a silent track is
            # worse than none, so this appears only when a track was really muxed.
            if self.audio_info:
                entry["audio"] = self.audio_info
            meta.setdefault("streams", {})[self.cam] = entry
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(meta, indent=2))
            tmp.replace(path)

    def _write_geo(self):
        """Emit the depth-measured room positions as the geo sidecars the mirror
        and LAN API read — the SAME schema localize.py produces (per-person
        {t, x, y (anchor pixel), room:[x,z], src}). Flagged `live` so the offline
        localize pass leaves it alone (it has no depth to do better)."""
        gi = self.geom
        if gi.get("cam_pos_mm") is None:
            return
        persons: dict = {}
        for frame_no, poslist in self.geo_rows:
            t = round(frame_no / 30.0, 3)
            for e in poslist:
                persons.setdefault(str(e["id"]), []).append({
                    "t": t,
                    "x": round(float(e["px"][0]), 1), "y": round(float(e["px"][1]), 1),
                    "room": [round(float(e["room"][0]), 1), round(float(e["room"][1]), 1)],
                    "src": e["src"],
                })
        stem = self.cam
        mp4 = self.dir / f"{stem}.mp4"
        try:
            mtime_ms = mp4.stat().st_mtime * 1000
        except OSError:
            mtime_ms = 0
        cam_pos = [round(float(v), 1) for v in gi.get("cam_pos_mm", [])]
        room_frame = {
            "origin": "floor point directly under the AprilTag's center",
            "axes": "X = tag's right (viewed facing the tag), Z = out of the wall; mm",
            "tagId": gi.get("tag_id"), "tagHeightMm": gi.get("tag_height_mm"),
            "cameraPositionMm": cam_pos, "cameraId": gi.get("camera_id"),
        }
        common = {"schemaVersion": 3, "model": "geo", "source": f"{stem}.mp4",
                  "sourceMtimeMs": mtime_ms, "live": True}

        def _atomic(path: Path, data: dict):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(path)

        try:
            _atomic(self.dir / f"{stem}.centroids.geo.json",
                    {**common, "nativeFps": 30.0, "persons": persons, "roomFrame": room_frame})
            _atomic(self.dir / f"{stem}.detections.geo.json",
                    {**common, "status": "done", "framesAnalyzed": self.frames,
                     "durationSec": round(self.frames / 30.0, 2), "tracks": len(persons)})
        except OSError as exc:
            print(f"[live] {self.cam}: cannot write geo sidecar: {exc}", flush=True)

    def stats(self):
        return {"kept": self.kept, "dropped": self.dropped,
                "frames": self.frames, "peopleFrames": self.people_frames,
                "segment": self.dir.parent.parent.name if self.dir else None}


_SEG_META_LOCK = threading.Lock()


# COCO-17 skeleton edges (for drawing) + a color per limb group.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (5, 6), (11, 12), (5, 11), (6, 12),       # torso
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),  # head
]


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


def find_calib_clips(cam_key: str) -> list:
    """Uploaded <cam_key>.mp4 clips with calibration+extrinsics, freshest first.

    Returns every candidate, not just the freshest: a clip can carry calibration
    yet still fail load_room_geometry (e.g. a recorded segment with no
    room_frame/tag height), and picking only one made startup fail outright
    instead of falling back to a usable one.

    Ordering is by the extrinsics' own `calibrated_at`, and clips this recorder
    WROTE come last. Both parts matter, because ranking by file mtime and
    treating all clips alike created a feedback loop that froze the geometry:
    every segment SegmentRecorder writes carries a verbatim copy of the
    calibration it bootstrapped from, so its own fresh-on-disk output was always
    the "newest" clip and every restart re-adopted its own stale snapshot. A
    recalibration on the Pi could never reach this process — the room's anchor
    tag was physically replaced and 259 segments went on being localized against
    the tag that had been taken down, the two cameras' people landing ~0.6 m
    apart in the fused scene. A real Pi capture is the authority on where the
    cameras are; a segment is only a last-resort fallback.
    """
    root = saved_root()
    out = []
    if not root.exists():
        return out
    for mp4 in root.rglob(f"{cam_key}.mp4"):
        if "undistorted" in mp4.parts:
            continue
        md = mp4.parent / "metadata.json"
        if not md.exists():
            continue
        try:
            meta = json.loads(md.read_text())
            entry = (meta.get("streams") or {}).get(mp4.stem, {})
            if not (entry.get("calibration") and entry.get("extrinsics")):
                continue
            # A segment we wrote is a COPY of someone else's calibration, so it
            # can never be more authoritative than its source.
            derived = meta.get("source") == "live_segment_recorder"
            stamp = (entry["extrinsics"] or {}).get("calibrated_at") or ""
            mtime = mp4.stat().st_mtime
        except (OSError, ValueError, AttributeError):
            continue
        out.append((not derived, stamp, mtime, mp4))
    # Descending, so real captures (not derived -> True) come first, then the most
    # recently CALIBRATED, then the most recently written. mtime only breaks ties
    # so a re-uploaded clip still wins over one with the same or no stamp.
    out.sort(reverse=True, key=lambda t: t[:3])
    return [t[-1] for t in out]


class TimestampLog:
    """Per-camera frame-timestamp CSV on the server. One row per processed frame:
    the sensor hw timestamp (librealsense global clock — the cross-camera sync
    key, matchable to ±1-2ms between the D455 and D435), the server's receive
    time, and how many people were localized. Lives under the DATA dir."""

    HEADER = "frame,hw_timestamp_ms,server_ms,persons\n"

    def __init__(self, cam_key: str, session: str):
        d = Path(os.environ.get("SMARTROOM_LIVE_LOG_DIR")
                 or (saved_root().parent / "live"))
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"live_{session}_{cam_key}_timestamps.csv"
        self.n = 0
        self._fh = open(self.path, "w", buffering=1)   # line-buffered
        self._fh.write(self.HEADER)
        print(f"[live] {cam_key}: timestamps -> {self.path}", flush=True)

    def write(self, hw_ts, persons):
        self.n += 1
        try:
            self._fh.write(f"{self.n},{hw_ts:.3f},{time.time() * 1000:.3f},{persons}\n")
        except OSError:
            pass


class Shared:
    """Newest-frame-wins slots shared across the ingest, inference and HTTP
    threads (mirrors realsense_depth_page.py's ViewCache pattern)."""

    def __init__(self):
        self.cond = threading.Condition()
        self.in_jpeg = None          # latest raw JPEG bytes from the Pi
        self.in_hw_ts = 0.0          # its sensor timestamp (global clock, ms)
        # When THIS server received the frame. The forwarder's own timestamp
        # (in_hw_ts) is on the forwarding host's clock — librealsense's on the Pi,
        # plain wall-clock on the Reolink host — and cross-camera comparisons
        # cannot be built on three clocks whose agreement is nobody's invariant.
        # Arrival is one clock for every camera; CAM_OFFSETS carries the rest.
        self.in_recv_ms = 0.0
        self.in_id = 0
        self.out_jpeg = None         # latest annotated JPEG
        self.out_id = 0
        self.positions = []          # [{id,x,z,src,cam,actions}]
        self.updated_ms = 0
        self.fps = 0.0
        self.hw_ts = 0.0             # sensor timestamp of the newest output frame
        self.frame_cap_ms = 0.0      # capture instant of the DISPLAYED frame
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
        # Annotated frames waiting for their presentation slot, oldest first, each
        # with the capture time that decides when it is due.
        self.pending = deque()

    def put_in(self, jpeg, hw_ts=0.0, recv_ms=None):
        with self.cond:
            self.in_jpeg = jpeg
            self.in_hw_ts = hw_ts
            self.in_recv_ms = time.time() * 1000.0 if recv_ms is None else recv_ms
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

    def put_out(self, jpeg, positions, fps, hw_ts=0.0, t_cap=None):
        """Publish an annotated frame, or hold it until its presentation slot.

        Held rather than shown immediately so that every camera displays the same
        captured instant — see PRESENT_SYNC. With no measured offsets the delay is
        zero and this is a straight publish, exactly as before.
        """
        if t_cap is None or present_delay_ms() <= 0:
            with self.cond:
                self._publish_locked(jpeg, positions, fps, hw_ts, t_cap)
            return
        with self.cond:
            self.pending.append((t_cap, jpeg, positions, fps, hw_ts))
            while len(self.pending) > PRESENT_MAX_FRAMES:
                self.pending.popleft()

    def release_due(self, cutoff_t):
        """Show the newest held frame captured at or before `cutoff_t`.

        Newest, not oldest: if this camera fell behind and several frames came due
        at once, the current one is what belongs on screen — replaying the backlog
        in order would run the view in slow motion until it caught up.
        """
        with self.cond:
            pick = None
            while self.pending and self.pending[0][0] <= cutoff_t:
                pick = self.pending.popleft()
            if pick is None:
                return False
            t_cap, jpeg, positions, fps, hw_ts = pick
            self._publish_locked(jpeg, positions, fps, hw_ts, t_cap)
            return True

    def _publish_locked(self, jpeg, positions, fps, hw_ts, t_cap=None):
        # The instant the displayed frame was CAPTURED, on the shared timeline.
        # Published so "the cameras are in step" is something you can read off the
        # API and check, rather than infer from the delay arithmetic.
        self.frame_cap_ms = (t_cap * 1000.0) if t_cap else 0.0
        self.out_jpeg = jpeg
        self.out_id += 1
        self.positions = positions
        self.fps = fps
        self.hw_ts = hw_ts
        self.updated_ms = int(time.time() * 1000)
        self.cond.notify_all()


class IdentityRegistry:
    """Shared across camera threads: turns per-camera ByteTrack ids into stable
    GLOBAL person ids, so the same human keeps one id across occlusions, across
    re-entries, and across the two cameras."""

    def __init__(self):
        self.lock = threading.Lock()
        self.map = {}        # (cam, track_id) -> gid
        self.gallery = {}    # gid -> {emb, pos, t, cam, seen}
        self.next_gid = 1
        self.misses = []     # recent best-but-rejected ReID scores (threshold tuning)
        # Threshold calibration from live data: `genuine` = a track vs its own
        # stored embedding (definitely the same person); `impostor` = two tracks
        # visible in the SAME frame (definitely different people). The right
        # REID_THRESH sits between the two distributions.
        self.genuine = []
        self.impostor = []
        self.geo_sim = []      # appearance scores seen on candidate geo merges
        self.geo_vetoed = 0    # how many the veto blocked
        self.pending = {}      # (gid_a, gid_b) -> consecutive co-located observations
        self.split_pending = {}  # (cam, tid) -> consecutive far-apart observations
        self.splits = 0
        self.fused = 0         # identities merged by the continuous fusion pass

    @staticmethod
    def _cos(a, b):
        if a is None or b is None:
            return -1.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return -1.0
        return float(np.dot(a, b) / (na * nb))

    def assign(self, cam, tid, emb, pos, t, taken):
        """-> (gid, how). `taken` = gids already used by this camera this frame,
        so one camera can never map two people onto the same identity."""
        with self.lock:
            self._prune(t)
            key = (cam, tid)
            gid = self.map.get(key)
            if gid is not None and gid in self.gallery and gid not in taken:
                e = self.gallery[gid]
                # Against where the other camera put this identity AT THIS FRAME'S
                # time — not at its own latest frame's time, which for a camera
                # seconds behind is a different moment entirely.
                other = self._pos_at(e, t) if e["cam"] != cam else None
                stale_merge = (GEO_MERGE_MM > 0 and other is not None
                               and ((pos[0] - other[0]) ** 2
                                    + (pos[1] - other[1]) ** 2) ** 0.5 > GEO_SPLIT_MM)
                if not stale_merge:
                    self.split_pending.pop(key, None)
                    self._touch(gid, emb, pos, t, cam)
                    return gid, "track"
                # Disagreement seen — but do not split on a single frame (that
                # oscillated against fuse()). Only break the mapping once it has
                # persisted, otherwise ride it out.
                n = self.split_pending.get(key, 0) + 1
                self.split_pending[key] = n
                if n < GEO_SPLIT_PERSIST:
                    self._touch(gid, emb, pos, t, cam)
                    return gid, "track"
                self.split_pending.pop(key, None)
                self.splits += 1
                self.map.pop(key, None)   # sustained mismatch — re-match below

            best, how = None, "new"
            # 1) cross-camera geometry: the other camera, right now, same spot
            best_d = GEO_MERGE_MM
            for g, e in self.gallery.items():
                if GEO_MERGE_MM <= 0:
                    break            # location matching disabled — appearance only
                if g in taken or e["cam"] == cam:
                    continue
                # Signed comparisons were wrong in BOTH directions once cameras
                # differ by seconds: a late camera's frame made every fresher entry
                # look like a negative age and sail through the gate, while a
                # prompt camera saw the late one as permanently stale and never
                # matched it at all. _pos_at is symmetric and picks the moment.
                other = self._pos_at(e, t)
                if other is None:
                    continue
                d = ((pos[0] - other[0]) ** 2 + (pos[1] - other[1]) ** 2) ** 0.5
                if d >= best_d:
                    continue
                # Appearance veto — same place is not enough if they look nothing
                # alike. Record every candidate's score so the bar is tunable.
                if emb is not None and e["emb"] is not None:
                    sim = self._cos(emb, e["emb"])
                    self.geo_sim.append(round(sim, 3))
                    del self.geo_sim[:-100]
                    if sim < GEO_REID_MIN:
                        self.geo_vetoed += 1
                        continue
                best, best_d, how = g, d, "geometry"
            # 2) appearance: bridges gaps geometry can't (re-entry after absence)
            top_s = None
            if best is None and emb is not None:
                best_s = REID_THRESH
                for g, e in self.gallery.items():
                    if g in taken:
                        continue
                    s = self._cos(emb, e["emb"])
                    top_s = s if top_s is None else max(top_s, s)
                    if s > best_s:
                        best, best_s, how = g, s, "reid"

            if best is None:
                # log the near-miss so REID_THRESH can be tuned from real data
                if top_s is not None:
                    self.misses.append(round(top_s, 3))
                    del self.misses[:-50]
                best = self.next_gid
                self.next_gid += 1
                self.gallery[best] = {"emb": emb, "pos": pos, "t": t, "cam": cam, "seen": 0}
            self.map[key] = best
            self._touch(best, emb, pos, t, cam)
            return best, how

    @staticmethod
    def _pos_at(e, t, window=None):
        """Where this identity was at time `t`, or None if it was not seen then.

        The trail is what lets a camera running seconds behind be compared against
        the right moment instead of against "now". Falls back to the latest
        position for an entry with no trail yet.
        """
        window = GEO_MERGE_S if window is None else window
        trail = e.get("trail")
        if not trail:
            return e["pos"] if abs(t - e["t"]) <= window else None
        best, best_dt = None, window
        for ts, p in reversed(trail):
            dt_ = abs(ts - t)
            if dt_ <= best_dt:
                best, best_dt = p, dt_
            elif ts < t - window:
                break        # trail is time-ordered; nothing older can be nearer
        return best

    def _touch(self, gid, emb, pos, t, cam):
        e = self.gallery.setdefault(gid, {"emb": emb, "pos": pos, "t": t, "cam": cam,
                                          "seen": 0, "trail": []})
        if emb is not None:
            e["emb"] = emb if e["emb"] is None else (
                EMB_MOMENTUM * e["emb"] + (1 - EMB_MOMENTUM) * emb)
        e["pos"], e["cam"] = pos, cam
        # `t` is the newest time SEEN, not the newest processed: a late camera must
        # not be able to drag the entry's clock backwards and make fresher
        # observations from another camera look stale.
        e["t"] = max(e.get("t", t), t)
        trail = e.setdefault("trail", [])
        trail.append((t, pos))
        # Keep it ordered (a late camera appends out of order) and bounded.
        if len(trail) > 1 and trail[-2][0] > t:
            trail.sort(key=lambda r: r[0])
        cutoff = e["t"] - GEO_HIST_S
        if trail[0][0] < cutoff:
            e["trail"] = [r for r in trail if r[0] >= cutoff]
        e["seen"] += 1

    def _prune(self, t):
        dead = [g for g, e in self.gallery.items() if t - e["t"] > GALLERY_TTL_S]
        for g in dead:
            self.gallery.pop(g, None)
        if dead:
            for k, g in list(self.map.items()):
                if g in dead:
                    self.map.pop(k, None)

    def note(self, cam, pairs):
        """pairs: [(track_id, emb)] for one frame of one camera."""
        with self.lock:
            live = [(t, e) for t, e in pairs if e is not None]
            for i in range(len(live)):
                t_i, e_i = live[i]
                gid = self.map.get((cam, t_i))
                if gid in self.gallery and self.gallery[gid]["emb"] is not None:
                    self.genuine.append(round(self._cos(e_i, self.gallery[gid]["emb"]), 3))
                for j in range(i + 1, len(live)):
                    self.impostor.append(round(self._cos(e_i, live[j][1]), 3))
            del self.genuine[:-400]
            del self.impostor[:-400]

    def fuse(self, t):
        """Continuously fuse identities that are the same person.

        Merging only at track creation was not enough: each camera creates its
        own track for a person independently (e.g. both at startup for someone
        already seated), so two identities could sit on top of each other
        forever and never combine. This re-checks every live identity pair, and
        requires the agreement to PERSIST for a few observations so two people
        merely passing each other are not fused."""
        if GEO_MERGE_MM <= 0:
            return
        with self.lock:
            # Every identity seen recently enough to still have a usable trail —
            # NOT only those whose newest frame is within GEO_MERGE_S of `t`. That
            # test silently excluded any camera running more than half a second
            # behind, which with the NVR's several seconds meant a Reolink identity
            # could never be a fusion candidate at all, whatever its position.
            live = [(g, e) for g, e in self.gallery.items()
                    if abs(t - e["t"]) < GEO_HIST_S]
            seen = set()
            for i in range(len(live)):
                for j in range(i + 1, len(live)):
                    ga, ea = live[i]
                    gb, eb = live[j]
                    if ea["cam"] == eb["cam"]:
                        continue          # one camera cannot see one person twice
                    key = (min(ga, gb), max(ga, gb))
                    # Compare the two at a moment they BOTH have a position for,
                    # rather than each at its own latest frame.
                    when = min(ea["t"], eb["t"])
                    pa, pb = self._pos_at(ea, when), self._pos_at(eb, when)
                    if pa is None or pb is None:
                        self.pending.pop(key, None)
                        continue
                    d = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
                    ok = d <= GEO_MERGE_MM
                    if ok and ea["emb"] is not None and eb["emb"] is not None:
                        ok = self._cos(ea["emb"], eb["emb"]) >= GEO_REID_MIN
                    if not ok:
                        self.pending.pop(key, None)
                        continue
                    seen.add(key)
                    n = self.pending.get(key, 0) + 1
                    self.pending[key] = n
                    if n < GEO_FUSE_PERSIST:
                        continue
                    keep, drop = key            # keep the older (lower) identity
                    for k, v in list(self.map.items()):
                        if v == drop:
                            self.map[k] = keep
                    if self.gallery.get(drop, {}).get("emb") is not None \
                            and self.gallery.get(keep, {}).get("emb") is None:
                        self.gallery[keep]["emb"] = self.gallery[drop]["emb"]
                    self.gallery.pop(drop, None)
                    self.pending.pop(key, None)
                    self.fused += 1
                    return                      # one fusion per pass; re-check next time
            for k in [k for k in self.pending if k not in seen]:
                self.pending.pop(k, None)

    def known(self, cam, tid):
        with self.lock:
            return (cam, tid) in self.map

    def stats(self):
        with self.lock:
            def pct(v, q):
                if not v: return None
                v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]
            gs = sorted(self.geo_sim)
            return {"known": len(self.gallery), "tracks": len(self.map),
                    "thresh": REID_THRESH,
                    "geoMerge": {"reidMin": GEO_REID_MIN, "vetoed": self.geo_vetoed,
                                 "fused": self.fused, "pending": len(self.pending),
                                 "splits": self.splits,
                                 "n": len(gs),
                                 "p10": gs[len(gs) // 10] if gs else None,
                                 "p50": gs[len(gs) // 2] if gs else None,
                                 "p90": gs[min(len(gs) - 1, 9 * len(gs) // 10)] if gs else None},
                    "genuine": {"n": len(self.genuine), "p05": pct(self.genuine, .05),
                                "p50": pct(self.genuine, .50), "p95": pct(self.genuine, .95)},
                    "impostor": {"n": len(self.impostor), "p50": pct(self.impostor, .50),
                                 "p95": pct(self.impostor, .95), "p99": pct(self.impostor, .99)}}


def _make_bytetrack():
    """ByteTracker, tuned for a fixed camera watching a small room.

    NOT ultralytics' stock bytetrack.yaml any more. Those defaults are set for
    short benchmark clips with a moving camera, and measured here they were the
    direct cause of identity flicker: the track-id counter advanced 284 in 40
    seconds for 6 people (~7 new tracks/second), and ~276 global identities had
    been minted for the same 6 humans. Every death of a ByteTrack track forces
    the global registry to re-match that person by appearance, and the genuine
    match distribution (p05 0.617 vs REID_THRESH 0.55) loses its tail — a failed
    re-match mints a new identity, which is the flicker you see.

    Two defaults did it, both fixed below. Note the global-ID layer was NOT at
    fault: over the same 40s a global id changed under a stable track exactly
    once.
    """
    from types import SimpleNamespace

    from ultralytics.trackers.byte_tracker import BYTETracker
    args = SimpleNamespace(
        # Stock 0.25 starts a track off almost any weak detection, which is
        # where ~7 spurious tracks/second came from. A real person in this room
        # detects far above this.
        new_track_thresh=float(os.environ.get("SMARTROOM_NEW_TRACK_THRESH", "0.55")),
        track_high_thresh=float(os.environ.get("SMARTROOM_TRACK_HIGH_THRESH", "0.4")),
        track_low_thresh=0.1,
        # Frames a lost track stays alive. Ultralytics scales this by
        # frame_rate/30, so stock 30 = about ONE SECOND — shorter than any real
        # occlusion in a room with furniture and people walking past each other.
        # 90 gives ~3s, long enough to survive a pass-behind without being so
        # long that a stale track gets handed to a different person.
        track_buffer=int(os.environ.get("SMARTROOM_TRACK_BUFFER", "90")),
        match_thresh=0.8, fuse_score=True)
    # (No gmc_method: it is inert here. Only BOTSORT constructs self.gmc;
    # BYTETracker checks hasattr(self, "gmc") and never sets it. The camera is
    # bolted to the wall anyway.)
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
               mode: str, cam_key: str = "", tslog: "TimestampLog | None" = None,
               ids: "IdentityRegistry | None" = None,
               recorder: "SegmentRecorder | None" = None):
    from ultralytics import YOLO
    model = YOLO(weights)
    tracker = _make_bytetrack()
    jumps = JumpDetector()
    held = {}          # tid -> (pos, t) last good room position, for POS_HOLD_S
    # Can this camera place a person WITHOUT depth? Localization used to be
    # depth-only, which silently excluded every RGB-only camera: the four NVR
    # cameras run at 30-50fps, see the whole room from 2.6m, and contributed
    # nothing at all -- `depth_near` returns None for them, so every detection
    # hit `pos is None` and was dropped before it was even drawn. That is why the
    # room showed 2 people while nine were standing in it.
    #
    # The ankle floor-ray is the same fallback localize.py uses offline, with the
    # same guard: it only means anything when the camera actually looks DOWN at
    # the floor. Near-horizontal, a distant person's ankle sits by the horizon
    # where the ray grazes the floor and one pixel is worth metres.
    optical_axis = geom["R"] @ np.array([0.0, 0.0, 1.0])
    pitch_down_deg = float(np.degrees(np.arcsin(np.clip(-optical_axis[1], -1.0, 1.0))))
    ray_ok = pitch_down_deg >= MIN_RAY_PITCH_DEG
    print(f"[live] {cam_key}: pitch {pitch_down_deg:.0f}° below horizontal — "
          f"floor-ray fallback {'ON' if ray_ok else 'OFF'} "
          f"(needs >= {MIN_RAY_PITCH_DEG:.0f}°)", flush=True)

    spatial_off = spatial_muted(cam_key)
    if spatial_off:
        why = ("not in SMARTROOM_SPATIAL_ONLY=" + ",".join(sorted(SPATIAL_ONLY))
               if SPATIAL_ONLY else "SMARTROOM_NO_SPATIAL")
        print(f"[live] {cam_key}: SPATIAL MUTED ({why}) — streaming, pose and "
              f"recording continue; it publishes no room positions and no geo "
              f"into its segments", flush=True)
    encoder = None
    if ids is not None and REID_ON:
        # Out-of-process (see _reid_worker). Its ONNX Runtime thread pool
        # busy-waits, and keeping it out of here also leaves the door open to
        # running it on the GPU without adding a CUDA runtime to this process.
        encoder = ModelWorker(f"ReID[{cam_key}]", _reid_worker,
                              (REID_MODEL, REID_DEVICE), timeout=REID_TIMEOUT_S)
        if not encoder.start():
            print(f"[live] {cam_key}: ReID unavailable — geometry only", flush=True)
            encoder = None
    frame_n = 0
    use_half = device not in ("cpu", "intel:cpu")
    if use_half:
        # Pin THIS thread's current cuda device to the one its models live on.
        # A new thread always starts pointed at cuda:0 no matter where its
        # tensors are, and the mismatch corrupts the context — see _on_device()
        # in ava_model.py for the full mechanism.
        try:
            import torch
            torch.cuda.set_device(int(device))
        except Exception as exc:  # noqa: BLE001
            print(f"[live] {cam_key}: cannot pin cuda device {device}: {exc}",
                  flush=True)
    last_id = 0
    ema_fps = 0.0
    predict_fails = 0
    timing_prev = None      # previous gray frame, only while a calibration runs
    print(f"[live] {cam_key}: pose model loaded ({weights}) device={device} "
          f"half={use_half}", flush=True)
    while True:
        with shared.cond:
            while shared.in_id == last_id or shared.in_jpeg is None:
                shared.cond.wait(timeout=5.0)
                if shared.in_jpeg is None:
                    continue
            last_id = shared.in_id
            jpeg = shared.in_jpeg
            hw_ts = shared.in_hw_ts
            recv_ms = shared.in_recv_ms
        t_start = time.time()          # for the fps measurement only
        # When this frame was CAPTURED, as best this server can know: arrival,
        # less the measured delay of this camera's delivery path. Everything that
        # compares one camera against another must use this and not the clock —
        # see the CAM_OFFSETS comment.
        t_frame = capture_time_s(cam_key, recv_ms, hw_ts, t_start)
        note_observed_delay(cam_key, t_frame)
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        h, w = frame.shape[:2]
        if TIMING.armed:
            # Lights on/off calibration: this camera's own view of when the room
            # changed, against its RAW arrival time (the offset being measured
            # must not be subtracted from the measurement).
            gray = cv2.cvtColor(cv2.resize(frame, TIMING_SIZE), cv2.COLOR_BGR2GRAY)
            if timing_prev is not None:
                # energy places the event in time; brightness is what proves the
                # event was the LIGHTS and not somebody walking about.
                TIMING.sample(cam_key, recv_ms or t_start * 1000.0,
                              float(cv2.absdiff(gray, timing_prev).mean()),
                              float(gray.mean()), hw_ms=hw_ts)
            timing_prev = gray
        elif timing_prev is not None:
            timing_prev = None
        clean = frame.copy()   # pristine RGB for the action model (frame gets overlays)
        try:
            # Serialized per device — see POSE_LOCKS. `.cpu()` stays INSIDE the
            # lock: it is what forces the async CUDA work to complete, so
            # releasing before it would hand the device to the next thread while
            # this one's kernels are still in flight, which is the very overlap
            # this prevents.
            with POSE_LOCKS[device]:
                res = model.predict(frame, imgsz=640, device=device, half=use_half,
                                    classes=[0], verbose=False)[0].cpu()
        except Exception as exc:  # noqa: BLE001
            print(f"[live] {cam_key}: predict error: {exc}", flush=True)
            # A CUDA fault ("misaligned address") poisons the whole context: every
            # later kernel fails the same way, so the service stays *active* while
            # silently producing zero detections. Bail out and let systemd give us
            # a fresh process rather than spin on a dead GPU for hours.
            predict_fails += 1
            if predict_fails >= PREDICT_FAIL_LIMIT:
                print(f"[live] {cam_key}: {predict_fails} consecutive predict "
                      f"failures — exiting for a restart", flush=True)
                os._exit(1)
            continue
        predict_fails = 0

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
            # Reject phantom detections (dark chairs etc.): a real human lights up
            # far more than a couple of joints. Gate on pose quality, not the box.
            if sum(c >= KP_CONF for c in p["conf"]) < MIN_POSE_KP:
                continue
            anchor = hip_point(p, w, h)   # mid-hip pixels, or None if low-conf
            src = "depth-hip"
            if anchor is None:
                anchor = _shoulder_point(p, w, h)
                src = "depth-shoulder"
            if anchor is None:
                continue
            anchors_frac.append([anchor[0] / w, anchor[1] / h])
            z_mm = shared.depth_near(anchor[0] / w, anchor[1] / h)
            pos = None
            if z_mm:
                p_room = backproject_room(anchor[0], anchor[1],
                                          z_mm + BODY_HALF_DEPTH_MM, geom)
                if p_room is not None:
                    pos = (float(p_room[0]), float(p_room[2]))
                    held[tid] = (pos, t_frame)
            if pos is None and ray_ok:
                # No depth (an RGB-only camera, or a sample that has not landed
                # near this person yet): cast the ankle ray onto the floor.
                foot = ground_point(p, w, h)
                if foot is not None:
                    hit = pixel_to_floor(foot[0], foot[1], geom, ANKLE_JOINT_HEIGHT_MM)
                    if hit is not None:
                        # Reject a hit implausibly far away: near the horizon the
                        # ray grazes the floor and a 1px error becomes metres,
                        # which is what used to fling people through walls.
                        reach = float(np.hypot(hit[0] - geom["cam_pos_mm"][0],
                                               hit[1] - geom["cam_pos_mm"][2]))
                        if reach <= MAX_RAY_REACH_MM:
                            pos, src = (float(hit[0]), float(hit[1])), "ray-ankles"
                            held[tid] = (pos, t_frame)
            if pos is None:
                # no fresh depth this frame — hold the last known position rather
                # than dropping the person (that is what caused the flicker).
                prev = held.get(tid)
                if prev and t_frame - prev[1] <= POS_HOLD_S:
                    pos, src = prev[0], src + "-hold"
                else:
                    continue
            found.append((tid, pos, anchor, p, src))
        shared.set_hips(anchors_frac)

        # Stable global identities: embed every localized person (throttled —
        # appearance changes slowly), then resolve via geometry + appearance.
        frame_n += 1
        gids = {}
        if ids is not None and found:
            embs = [None] * len(found)
            # Throttle the encoder, BUT never skip a frame containing an unseen
            # track: a new track with no embedding could only ever match on
            # geometry, so it would mint a fresh identity instead of being
            # re-identified — the whole point of the gallery.
            fresh = any(not ids.known(cam_key, t) for t, *_ in found)
            if encoder is not None and (fresh or frame_n % REID_EVERY == 0):
                dets = np.array([[(p["box"][0] + p["box"][2]) / 2,
                                  (p["box"][1] + p["box"][3]) / 2,
                                  p["box"][2] - p["box"][0],
                                  p["box"][3] - p["box"][1]]
                                 for *_, p, _ in found], dtype=np.float32)
                ok, enc_jpg = cv2.imencode(".jpg", clean,   # the CLEAN frame
                                           [cv2.IMWRITE_JPEG_QUALITY, 90])
                # A failed/timed-out embed is not fatal: identities fall back to
                # geometry for this frame, exactly as before.
                got = encoder.call((enc_jpg.tobytes(), dets)) if ok else None
                if got is not None and len(got) == len(found):
                    embs = got
            if any(e is not None for e in embs):
                ids.note(cam_key, [(t, e) for (t, *_), e in zip(found, embs)])
            ids.fuse(t_frame)      # continuously combine co-located identities
            taken = set()
            for (tid, pos, _marker, _p, _src), emb in zip(found, embs):
                gid, how = ids.assign(cam_key, tid, emb, pos, t_frame, taken)
                taken.add(gid)
                gids[tid] = (gid, how)

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
            # multi-label: every class the classifier put above threshold
            acts = [list(a) for a in lab["top"]] if (lab and lab.get("top")) else []
            # geometric jump detector — independent of the classifier; when airborne
            # add "jump" to the set (at the front) rather than replacing it.
            comy, body_h = _hip_com(p)
            if jumps.update(tid, comy, body_h, t_frame):
                acts = [["jump", 1.0]] + [a for a in acts if a[0] != "jump"]
            gid, how = gids.get(tid, (None, None))
            entry = {"id": tid, "x": round(pos[0], 1), "z": round(pos[1], 1),
                     "src": src, "cam": cam_key}
            if gid is not None:
                entry["gid"] = gid          # stable across gaps AND cameras
                entry["idSrc"] = how        # track | geometry | reid | new
            if acts:
                entry["actions"] = acts               # full above-threshold set
                entry["action"] = acts[0][0]          # primary, for the map dot
                entry["actionConf"] = acts[0][1]
            if not spatial_off:
                positions.append(entry)
            _draw_person(frame, p["px"], p["conf"], marker, tid, src, acts,
                         gid if gid is not None else tid)
        jumps.prune({tid for tid, *_ in found}, t_frame)
        for _t in [k for k, v in held.items() if t_frame - v[1] > 5]:
            held.pop(_t, None)
        if ava:
            shared.push_ava(clean, ava_boxes, w, h)   # clean frame, NOT the annotated one

        if recorder is not None:
            # archive the ORIGINAL jpeg (no re-encode) with this frame's
            # depth-measured room positions, so the segment is localizable
            # offline without a depth track. marker = the anchor pixel that was
            # depth-ranged; pos = its room (x,z).
            # Muted camera: record the video but NOT the room positions, or the
            # offline pass and the 3D scene would read them straight back out of
            # the segment and we would have muted nothing.
            geo_frame = [] if spatial_off else [
                {"id": tid, "px": [float(marker[0]), float(marker[1])],
                 "room": [float(pos[0]), float(pos[1])], "src": src}
                for tid, pos, marker, p, src in found]
            # `found` is what was DETECTED; geo_frame is what this camera is allowed
            # to publish. A muted camera has the first without the second.
            recorder.add(jpeg, hw_ts, geo_frame, t_frame * 1000.0, people=len(found))
        if tslog is not None:
            tslog.write(hw_ts, len(positions))

        dt = time.time() - t_start
        ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
        # Count what was DETECTED, not what was published: a muted camera still
        # sees people, and an overlay reading "0 person(s)" over a visible person
        # would look like the detector had failed.
        cv2.putText(frame, f"{cam_key}  {len(found)} person(s)"
                           f"{'  [no spatial]' if spatial_off else ''}  {ema_fps:4.1f} fps",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        ok, enc = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            shared.put_out(enc.tobytes(), positions, round(ema_fps, 1), hw_ts,
                           t_cap=t_frame)


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


class ModelWorker:
    """A model living in its own process, called request/response over a Pipe.

    Every GPU model this service runs gets one of these. Compiled CUDA
    extensions from different projects corrupt each other's context when they
    issue kernels from one process (mmaction/mmcv vs ultralytics/torchvision —
    see `_ava_worker`), and a corrupted context never recovers, so the only
    durable isolation is a process boundary. It also means a model that dies or
    wedges costs one respawn instead of the whole service.

    `call` never blocks the caller indefinitely: a worker that misses its
    deadline is killed and respawned, and the caller gets None for that request.
    """

    def __init__(self, name: str, target, args: tuple, timeout: float,
                 ready_timeout: float = 300.0):
        self.name, self.target, self.args = name, target, args
        self.timeout, self.ready_timeout = timeout, ready_timeout
        self.ctx = mp.get_context("spawn")
        self.conn = self.proc = None
        self.info = None

    def start(self) -> bool:
        """(Re)spawn the worker; False if it could not be brought up."""
        self.stop()
        parent, child = self.ctx.Pipe()
        p = self.ctx.Process(target=self.target, daemon=True,
                             args=(child,) + self.args)
        p.start()
        child.close()
        try:
            # model build is slow (checkpoint load), so wait — but not forever
            if not parent.poll(self.ready_timeout):
                raise TimeoutError(f"not ready in {self.ready_timeout:g}s")
            kind, payload = parent.recv()
            if kind != "ready":
                raise RuntimeError(payload)
        except (OSError, EOFError, TimeoutError, RuntimeError) as exc:
            print(f"[live] {self.name} worker failed to start: {exc}", flush=True)
            p.kill()
            parent.close()
            return False
        self.conn, self.proc, self.info = parent, p, payload
        print(f"[live] {self.name} worker up (pid {p.pid}): {payload}", flush=True)
        return True

    def stop(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        if self.proc is not None and self.proc.is_alive():
            self.proc.kill()
        self.conn = self.proc = None

    def start_blocking(self, retry_s: float = 30.0):
        while not self.start():
            time.sleep(retry_s)

    def call(self, payload):
        """Round-trip one request. Returns the reply, or None on any failure
        (the worker is respawned in the background of the next call)."""
        if self.conn is None and not self.start():
            return None
        try:
            self.conn.send(payload)
            if not self.conn.poll(self.timeout):
                raise TimeoutError(f"no reply in {self.timeout:g}s")
            kind, reply = self.conn.recv()
        except (OSError, EOFError, TimeoutError) as exc:
            print(f"[live] {self.name} worker lost ({exc}) — respawning", flush=True)
            self.start()
            return None
        if kind != "ok":
            print(f"[live] {self.name} infer error: {reply}", flush=True)
            return None
        return reply


def _reid_worker(conn, model_path: str, device):
    """Subprocess entry point: the ReID appearance encoder.

    Isolated for two reasons. It keeps ONNX Runtime's thread pool — which
    busy-waits, and was measured pinning several cores — out of the pose
    process. And it makes a GPU ReID safe to enable later: onnxruntime-gpu's
    CUDA provider would be a THIRD compiled CUDA runtime inside the pose
    process, the exact hazard that took this service down, but inside its own
    process it cannot reach anyone else's context.
    """
    import cv2 as _cv2
    import numpy as _np
    try:
        from ultralytics.trackers.utils.reid import ReID
        enc = ReID(model_path, device=device)
        conn.send(("ready", f"{model_path} on {device}"))
    except Exception as exc:  # noqa: BLE001
        conn.send(("fatal", str(exc)))
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        jpeg, dets = msg
        try:
            frame = _cv2.imdecode(_np.frombuffer(jpeg, _np.uint8), _cv2.IMREAD_COLOR)
            embs = enc(frame, dets)
            conn.send(("ok", [None if e is None else _np.asarray(e) for e in embs]))
        except Exception as exc:  # noqa: BLE001
            conn.send(("error", str(exc)))


def _ava_worker(conn, config_path: str, ckpt: str, label_map_path: str,
                device: str, thr: float):
    """Subprocess entry point: one SlowFast-AVA model answering infer requests.

    AVA runs in its own process because mmaction/mmcv's compiled CUDA ops and
    ultralytics/torchvision's cannot coexist in one. Measured on the quad
    server, 180s each: AVA alone 2605 forwards / 0 failures; pose alone clean
    over thousands of predicts at fp16 on real frames with real detections;
    the two together in one process fail catastrophically — every pose predict
    dead from iteration 26 with "CUDA error: misaligned address", preceded by
    cuDNN CUDNN_STATUS_MAPPING_ERROR / _INTERNAL_ERROR from AVA. GPU layout is
    irrelevant: splitting them across separate GPUs only delays it (iteration
    4411 instead of 26) and a single shared GPU is the fastest to die. Nothing
    inside one process avoids it, and a corrupted CUDA context never recovers,
    which is how the service silently served zero people for 8.5 hours.

    Requests are (jpegs, boxes); frames cross as JPEG rather than raw arrays —
    ~15KB instead of ~350KB each, so a 32-frame clip is ~0.5MB per call
    instead of 11MB.
    """
    import cv2 as _cv2
    import numpy as _np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ava_model import AvaDetector
    try:
        det = AvaDetector(config_path, ckpt, label_map_path, device, thr)
        conn.send(("ready", len(det.label_map)))
    except Exception as exc:  # noqa: BLE001
        conn.send(("fatal", str(exc)))
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        jpegs, boxes = msg
        try:
            frames = [_cv2.imdecode(_np.frombuffer(j, _np.uint8), _cv2.IMREAD_COLOR)
                      for j in jpegs]
            conn.send(("ok", det.infer([f for f in frames if f is not None], boxes)))
        except Exception as exc:  # noqa: BLE001
            conn.send(("error", str(exc)))


def ava_loop(shared: Shared, config_path: str, ckpt: str, label_map_path: str,
             device: str, action_thr: float):
    """SlowFast-AVA spatiotemporal detection: per prediction step, hand the
    trailing RGB clip + the current person boxes to the AVA worker PROCESS and
    publish each track's above-threshold labels. The model reads pixels, not
    keypoints — the skeletons are still drawn, they just don't drive the label.

    This thread now only marshals; the model itself lives in `_ava_worker`,
    see there for why."""
    worker = ModelWorker("AVA", _ava_worker,
                         (config_path, ckpt, label_map_path, device, action_thr),
                         timeout=AVA_TIMEOUT_S)
    worker.start_blocking()

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
        jpegs = []
        for e in seg:
            ok, enc = cv2.imencode(".jpg", e[0], [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                jpegs.append(enc.tobytes())
        if len(jpegs) < AVA_MIN_FRAMES:
            continue
        labels = worker.call((jpegs, boxes))
        if labels is None:          # timed out / died; the clip is dropped
            continue
        for tid, labs in labels.items():
            # multi-label: keep EVERY class above the threshold, not just top-1
            shared.set_label(tid, labs[0][0] if labs else None,
                             labs[0][1] if labs else 0.0, labs)


def presenter(cams: dict):
    """Release each camera's held frames when they come due.

    One thread for every camera, deliberately: they must all be released against
    the SAME cutoff, or the synchrony this exists to create would depend on how
    each camera's own thread happened to be scheduled.
    """
    while True:
        time.sleep(PRESENT_TICK_S)
        delay = present_delay_ms()
        if delay <= 0:
            if AUDIO_ON:
                AUDIO.release_due()   # its own trim may still hold chunks back
            continue          # uncalibrated: put_out published directly
        cutoff = time.time() - delay / 1000.0
        for e in cams.values():
            e["shared"].release_due(cutoff)
        if AUDIO_ON:
            AUDIO.release_due()


def timing_driver(cams: dict):
    """Close the lights on/off window when its time is up and solve.

    Its own thread rather than a check in the pose loop: the loop only runs while
    frames arrive, so a camera that dies mid-calibration would leave the window
    armed forever and the offsets never written. The solve must happen even when
    every camera has gone quiet — that outcome is itself the result.
    """
    while True:
        time.sleep(0.25)
        if TIMING.expired():
            TIMING.finish(list(cams))


def stall_watchdog(cams: dict):
    """Exit the process when a camera stops producing frames although the Pi is
    still feeding it.

    Deliberately compares two counters rather than watching a clock: `in_id`
    advances every time the ingest handler receives a frame from the Pi, and
    `updated_ms` advances every time the pose loop finishes one. An idle Pi (or
    a dropped uplink) freezes both and is NOT a stall — that is the Pi's problem
    and restarting here would only hide it. Only ingest advancing while output
    does not means *we* are stuck.

    On detection it dumps every thread's Python stack before exiting, so the
    hang leaves behind the one piece of evidence needed to fix it at source: a
    wedged process is otherwise un-introspectable (`ptrace_scope=1` blocks
    py-spy/gdb from another session, and /proc/<pid>/syscall reads back empty).
    """
    last_out = {k: -1 for k in cams}
    last_in = {k: -1 for k in cams}
    stalled_for = {k: 0.0 for k in cams}
    period = 5.0
    while True:
        time.sleep(period)
        for key, e in cams.items():
            sh = e["shared"]
            with sh.cond:
                in_id, out_id = sh.in_id, sh.out_id
            feeding = in_id != last_in[key]
            producing = out_id != last_out[key]
            last_in[key], last_out[key] = in_id, out_id
            if producing or not feeding:
                stalled_for[key] = 0.0
                continue
            stalled_for[key] += period
            if stalled_for[key] < STALL_S:
                continue
            print(f"[live] WEDGED: {key} produced no frame in "
                  f"{stalled_for[key]:.0f}s while ingest advanced to {in_id}. "
                  f"Dumping all thread stacks, then exiting for a restart.",
                  flush=True)
            sys.stdout.flush()
            faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
            sys.stdout.flush()
            # _exit, not sys.exit: the wedged thread never returns, so a clean
            # shutdown would block on it forever.
            os._exit(3)


def _draw_person(frame, px, conf, marker, tid, src, actions=None, label_id=None):
    # colour by GLOBAL id so the same person keeps one colour across cameras
    color = _track_color(label_id if label_id is not None else tid)
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
    mx, my = int(marker[0]) + 8, int(marker[1])
    cv2.circle(frame, (int(marker[0]), int(marker[1])), 6, mcol, 2)
    cv2.putText(frame, f"#{label_id if label_id is not None else tid}", (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # every above-threshold class, stacked below the id
    for i, a in enumerate(actions or []):
        cv2.putText(frame, f"{a[0]} {a[1]:.2f}", (mx, my + 16 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def _track_color(tid):
    rng = (37 * (tid + 1)) % 180
    hsv = np.uint8([[[rng, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(b), int(g), int(r))


def make_handler(cams: dict, ids: "IdentityRegistry | None" = None):
    """cams: {cam_key: {"shared": Shared, "roomFrame": {...}}} — every endpoint
    selects a camera with ?cam=<key> (defaults to the first registered)."""

    default_cam = next(iter(cams))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")

        def _cam(self):
            q = parse_qs(urlparse(self.path).query)
            key = (q.get("cam") or [default_cam])[0]
            return cams.get(key)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/depths":
                self._recv_depths()
                return
            if path in ("/record/start", "/record/stop"):
                self._record(path.endswith("start"))
                return
            if path in ("/timing/start", "/timing/cancel"):
                self._timing(path.endswith("start"))
                return
            if path == "/audio":
                self._recv_audio()
                return
            if path != "/ingest":
                self.send_error(404)
                return
            entry = self._cam()
            if entry is None:
                self.send_error(404, "unknown cam")
                return
            shared = entry["shared"]
            # length-prefixed JPEG stream over one persistent connection:
            # [4B len][8B double hw_ts_ms][jpeg]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            n = 0
            try:
                while True:
                    hdr = self._readn(12)
                    if not hdr:
                        break
                    length, hw_ts = struct.unpack(">Id", hdr)
                    if length == 0 or length > 20_000_000:
                        break
                    jpeg = self._readn(length)
                    if jpeg is None:
                        break
                    # Arrival stamped HERE, not in the pose loop: this is the last
                    # point that is purely about the camera's delivery path. Once
                    # the frame is queued, how long until a GPU picks it up is a
                    # property of this server's load, and letting that leak into
                    # the timestamp made a busy camera look like a late one.
                    shared.put_in(jpeg, hw_ts, time.time() * 1000.0)
                    n += 1
            except (ConnectionError, OSError):
                pass
            print(f"[live] ingest closed after {n} frames", flush=True)

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
            entry = self._cam()
            try:
                samples = json.loads(self.rfile.read(length) or b"[]")
                if entry is not None:
                    entry["shared"].put_depths(
                        [(float(s["u"]), float(s["v"]), float(s["m"]))
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
            elif path == "/record/status":
                self._json(RECORD.state())
            elif path == "/timing/status":
                self._json(TIMING.state())
            elif path == "/audio/status":
                self._json(AUDIO.state())
            elif path in ("/audio.mp3", "/audio"):
                self._serve_audio()
            else:
                self.send_error(404)

        def do_OPTIONS(self):
            # The Record button is served from the mirror on :3000 and posts here
            # on :8010, so the browser preflights it.
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _record(self, start):
            q = parse_qs(urlparse(self.path).query)
            if start:
                secs = (q.get("seconds") or q.get("duration") or [None])[0]
                try:
                    secs = float(secs) if secs else None
                except ValueError:
                    secs = None
                # Cap an open-ended request: a tab that closes mid-recording must
                # not leave the encoder running until someone notices.
                if secs is None:
                    secs = RECORD_MAX_S
                secs = max(1.0, min(secs, RECORD_MAX_S))
                state = RECORD.start(secs, (q.get("label") or [None])[0])
                print(f"[live] RECORD started ({secs:.0f}s max)", flush=True)
            else:
                state = RECORD.stop()
                print("[live] RECORD stopped", flush=True)
            self._json(state)

        def _recv_audio(self):
            """Sink for the forwarder: [4B len][8B double ts_ms][encoded audio]xN.

            Same framing as /ingest so there is one wire format to understand. The
            forwarder's timestamp is read but not used for timing — arrival is
            stamped here instead, so this does not depend on that host's clock.
            """
            if not AUDIO_ON:
                self.send_error(503, "audio relay disabled")
                return
            ctype = self.headers.get("X-Audio-Content-Type") or "audio/mpeg"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            n = 0
            try:
                while True:
                    hdr = self._readn(12)
                    if not hdr:
                        break
                    length, _ts = struct.unpack(">Id", hdr)
                    if length == 0 or length > 4_000_000:
                        break
                    data = self._readn(length)
                    if data is None:
                        break
                    AUDIO.push(data, ctype)
                    n += 1
            except (ConnectionError, OSError):
                pass
            print(f"[live] audio ingest closed after {n} chunks", flush=True)

        def _serve_audio(self):
            """Continuous encoded audio for an <audio> element."""
            if not AUDIO_ON:
                self.send_error(503, "audio relay disabled")
                return
            if not AUDIO.live():
                # 503 rather than an empty stream: a silent <audio> that never errors
                # is indistinguishable from a room with nobody in it.
                self.send_error(503, "no audio source connected")
                return
            with AUDIO.cond:
                AUDIO.listeners += 1
                ctype = AUDIO.content_type
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache, no-store")
            self._cors()
            self.end_headers()
            seq = AUDIO.seq
            try:
                while True:
                    seq, chunks = AUDIO.follow(seq)
                    for data in chunks:
                        self.wfile.write(data)
                    if not chunks and not AUDIO.live():
                        break
            except (ConnectionError, OSError):
                pass
            finally:
                with AUDIO.cond:
                    AUDIO.listeners = max(0, AUDIO.listeners - 1)

        def _timing(self, start):
            q = parse_qs(urlparse(self.path).query)
            if not start:
                self._json(TIMING.cancel())
                return
            raw = (q.get("seconds") or q.get("duration") or [None])[0]
            try:
                secs = float(raw) if raw else TIMING_DEFAULT_S
            except ValueError:
                secs = TIMING_DEFAULT_S
            ref = (q.get("ref") or q.get("reference") or [None])[0]
            if ref is not None and ref not in cams:
                self._json({"error": f"unknown reference camera {ref!r}",
                            **TIMING.state()}, code=400)
                return
            ok, state = TIMING.start(secs, ref)
            if ok:
                # the state's own remaining time, not the requested `secs` —
                # start() clamps, and a log line naming the rejected number
                # would send anyone reading it looking for a bug
                print(f"[live] timing calibration armed for "
                      f"{state.get('remainingS') or secs:.0f}s "
                      f"(reference={ref or 'auto'}) — flip the room lights",
                      flush=True)
            self._json(state)

        def _hips(self):
            entry = self._cam()
            hips = entry["shared"].get_hips() if entry else []
            body = json.dumps({"hips": hips}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _positions(self):
            # merged across every camera — they share the tag-1 room frame, so
            # one map shows everyone. Each entry carries its `cam`.
            merged, per_cam = [], {}
            for key, e in cams.items():
                sh = e["shared"]
                with sh.cond:
                    merged.extend(sh.positions)
                    per_cam[key] = {"fps": sh.fps, "updatedMs": sh.updated_ms,
                                    "hwTimestampMs": round(sh.hw_ts, 3),
                                    # capture instant of the frame on screen —
                                    # equal across cameras when synced
                                    "frameCaptureMs": round(sh.frame_cap_ms, 1),
                                    # ms subtracted from this camera's arrival
                                    # times to put it on the shared timeline
                                    "timeOffsetMs": round(cam_offset_ms(key), 1),
                                    # which clock this camera's frame time comes
                                    # from: its own per-frame stamp, or arrival
                                    "captureClock": cam_clock(key),
                                    "persons": len(sh.positions),
                                    "roomFrame": e["roomFrame"],
                                    "recording": (e["recorder"].stats()
                                                  if e.get("recorder") else None)}
            first = cams[default_cam]["shared"]
            body = json.dumps({
                "positions": merged,
                "cams": per_cam,
                "identities": ids.stats() if ids is not None else None,
                "timing": TIMING.state(),
                # How far behind live the whole view deliberately sits, so every
                # camera shows the same captured instant. 0 = not synced.
                "presentDelayMs": round(present_delay_ms(), 1),
                "audio": AUDIO.state() if AUDIO_ON else None,
                "updatedMs": first.updated_ms,
                "fps": first.fps,
                "roomFrame": cams[default_cam]["roomFrame"],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            entry = self._cam()
            if entry is None:
                self.send_error(404, "unknown cam")
                return
            shared = entry["shared"]
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
            body = PAGE_HTML.replace("__CAMS__", json.dumps(list(cams))).encode()
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
 button{background:#292524;color:#e7e5e4;border:1px solid #44403c;border-radius:7px;
        padding:7px 11px;font:13px system-ui;cursor:pointer}
 button:hover{background:#3f3f46}
 button[disabled]{opacity:.55;cursor:default}
 .sync{max-width:420px}
 .sync td{font-size:12px;padding:1px 8px 1px 0;color:#a8a29e}
 .sync td.n{text-align:right;font-variant-numeric:tabular-nums;color:#e7e5e4}
</style></head><body>
<h1>smartroom — live pose + room localization</h1>
<div class=wrap id=cards>
 <div class=card><div>Top-down room map (mm) — all cameras</div>
   <canvas id=map width=420 height=420></canvas>
   <div class=meta id=cnt></div></div>
 <div class="card sync"><div>Camera timing</div>
   <div class=meta>Each camera's frames reach this server after a different delay
     (the Reolink cameras come via the NVR and a second host). Fusing two cameras
     needs that delay measured. Press below, then turn the room lights fully OFF
     and back ON 3–4 times: a light change hits every camera at once, so they need
     no shared view. Moving around the room instead does not work — it correlates
     the cameras that can see you and is rejected.</div>
   <div style="margin-top:8px"><button id=syncbtn>Measure timing (lights on/off)</button></div>
   <div class=meta id=syncmsg></div>
   <table id=synctab></table></div>
</div>
<script>
const CAMS=__CAMS__;
// one video card per camera, inserted before the map card
const cards=document.getElementById('cards');
CAMS.forEach(function(c){
  const d=document.createElement('div');d.className='card';
  d.innerHTML='<div>'+c+'</div><img src="/live.mjpg?cam='+c+'">'+
              '<div class=meta id="fps_'+c+'"></div>';
  cards.insertBefore(d,cards.firstChild);
});
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
  // one colour per camera so you can see which camera saw whom
  const CAMCOL={};CAMS.forEach(function(c,i){CAMCOL[c]=['#f59e0b','#38bdf8','#a3e635'][i%3];});
  for(const p of pos){
    ctx.fillStyle=CAMCOL[p.cam]||'#f59e0b';
    ctx.beginPath();ctx.arc(tx(p.x),tz(p.z),8,0,7);ctx.fill();
    // GLOBAL id: same human keeps this across gaps and across cameras
    ctx.fillStyle='#0c0a09';ctx.fillText('#'+(p.gid!=null?p.gid:p.id),tx(p.x)-6,tz(p.z)+4);
    if(p.action){ctx.fillStyle='#fde68a';ctx.fillText(p.action,tx(p.x)+11,tz(p.z)+4);}
  }
  CAMS.forEach(function(c,i){ctx.fillStyle=CAMCOL[c];ctx.fillText('● '+c,pad+i*130,H-8);});
}
async function poll(){
  try{const r=await fetch('/positions');const d=await r.json();
    const pos=d.positions||[];room=d.roomFrame;draw(pos);
    const cams=d.cams||{};
    for(const c in cams){const el=document.getElementById('fps_'+c);
      const off=cams[c].timeOffsetMs;
      if(el)el.textContent='inference '+(cams[c].fps||0)+' fps · '+cams[c].persons+
        ' person(s) · timing '+(off?(off>0?'+':'')+off.toFixed(0)+' ms':'0 ms (unmeasured)');}
    renderTiming(d.timing);
    const acts=pos.map(p=>'#'+(p.gid!=null?p.gid:p.id)+' ['+p.cam.replace('camera_','').replace('_color','')+
      (p.idSrc?'/'+p.idSrc:'')+']: '+((p.actions&&p.actions.length)?
      p.actions.map(a=>a[0]+' '+a[1].toFixed(2)).join(', '):'…'));
    const idn=d.identities?(' · '+d.identities.known+' known identities'):'';
    document.getElementById('cnt').innerHTML=pos.length+' detection(s)'+idn+'<br>'+
      (acts.length?acts.join('<br>'):'—');
  }catch(e){}
  setTimeout(poll,200);
}
const btn=document.getElementById('syncbtn'),msg=document.getElementById('syncmsg'),
      tab=document.getElementById('synctab');
btn.onclick=async function(){
  btn.disabled=true;
  try{await fetch('/timing/start?seconds=25',{method:'POST'});}
  catch(e){msg.textContent='could not start: '+e;btn.disabled=false;}
};
function renderTiming(t){
  if(!t)return;
  btn.disabled=!!t.running;
  if(t.running){
    const n=Object.values(t.samples||{}).reduce((a,b)=>a+b,0);
    msg.textContent='measuring — '+(t.remainingS||0).toFixed(0)+'s left · '+n+
      ' frames · FLIP THE LIGHTS OFF AND ON';
  } else if(t.error){
    msg.textContent='failed: '+t.error;
  } else if(t.result){
    const r=t.result;
    msg.textContent='measured '+(r.measured_at||'').slice(0,19).replace('T',' ')+
      ' · reference '+r.reference;
  } else {
    msg.textContent='no measurement yet — every camera is treated as arriving at the same instant';
  }
  const offs=t.result?t.result.offsets_ms:(t.offsetsMs||{});
  const q=(t.result&&t.result.quality)||{}, bad=(t.result&&t.result.rejected)||{};
  let rows='';
  Object.keys(offs).sort().forEach(function(c){
    const o=offs[c], k=q[c]||{};
    rows+='<tr><td>'+c.replace('camera_','').replace('_color','')+'</td>'+
      '<td class=n>'+(o>0?'+':'')+o.toFixed(0)+' ms</td>'+
      '<td>'+(k.correlation!=null?'r='+k.correlation:'reference')+
      (k.jitter_ms!=null?' · jitter '+k.jitter_ms+' ms':'')+'</td></tr>';
  });
  Object.keys(bad).sort().forEach(function(c){
    rows+='<tr><td>'+c.replace('camera_','').replace('_color','')+'</td>'+
      '<td class=n>—</td><td>'+bad[c]+'</td></tr>';
  });
  tab.innerHTML=rows;
}
poll();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cam", default="camera_d455_color",
                    help="comma-separated stream keys to serve, e.g. "
                         "camera_d455_color,camera_d435_color (calibration is "
                         "found per camera in the uploaded recordings)")
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

    # `kill -USR1 <pid>` dumps every thread's Python stack. The service is
    # otherwise a black box once a thread wedges inside a native call.
    faulthandler.register(signal.SIGUSR1, file=sys.stdout, all_threads=True)

    weights = os.environ.get("SMARTROOM_LIVE_WEIGHTS") or str(
        Path.home() / "Code/yolo-bench/yolo26n-pose.pt")
    device = os.environ.get("SMARTROOM_DETECT_DEVICE")
    ngpu = 0
    if device not in ("cpu", "intel:cpu"):
        try:
            import torch
            ngpu = torch.cuda.device_count()
        except Exception:  # noqa: BLE001
            ngpu = 0
    if not device:
        device = "0" if ngpu else "cpu"

    # One entry per camera: its own Shared buffers, geom, pose thread and
    # timestamp log. They all share the tag-1 room frame, so /positions merges.
    session = time.strftime("%Y%m%d_%H%M%S")
    ids = IdentityRegistry()      # SHARED by every camera -> one id per person
    cams = {}
    for cam_key in [c.strip() for c in args.cam.split(",") if c.strip()]:
        candidates = ([Path(args.clip)] if (args.clip and len(cams) == 0)
                      else find_calib_clips(cam_key))
        clip = geom = None
        for cand in candidates:
            if not cand.exists():
                continue
            g = load_room_geometry(cand, args.width, args.height, undistorted=False)
            if g is not None:
                clip, geom = cand, g
                break
        if geom is None:
            print(f"[live] SKIP {cam_key}: no recording with usable room geometry "
                  f"under {saved_root()} ({len(candidates)} candidate(s) tried)",
                  file=sys.stderr)
            continue
        room_frame = {
            "cameraPositionMm": [round(float(v), 1) for v in geom["cam_pos_mm"]],
            "tagId": geom.get("tag_id"),
            "tagHeightMm": geom.get("tag_height_mm"),
            "cameraId": geom.get("camera_id"),
            "calibClip": str(clip.relative_to(saved_root())),
        }
        # reuse the calibration/extrinsics from the clip we took geom from, so
        # recorded segments are themselves calibrated (and analysable later).
        stream_meta = {}
        try:
            src_meta = json.loads((clip.parent / "metadata.json").read_text())
            e = (src_meta.get("streams") or {}).get(cam_key) or {}
            stream_meta = {k: e[k] for k in ("calibration", "extrinsics") if k in e}
            node_name = src_meta.get("node") or "smartroom2"
            room_frame_meta = src_meta.get("room_frame")
        except (OSError, ValueError):
            node_name, room_frame_meta = "smartroom2", None
        recorder = (SegmentRecorder(cam_key, saved_root(), stream_meta, node_name,
                                    room_frame_meta, geom=geom) if SEGMENT_ON else None)
        shared = Shared()
        cams[cam_key] = {"shared": shared, "roomFrame": room_frame, "geom": geom,
                         "recorder": recorder}
        # One GPU per camera. Everything used to share cuda:0 — two pose models,
        # two ReID encoders and an AVA recognizer issuing kernels concurrently
        # into ONE context — while GPUs 1 and 2 sat idle. A context is
        # per-device, so splitting the cameras across devices means a hang on
        # one camera's GPU can no longer stall the other's, and the pose models
        # stop contending for the same streams.
        cam_device = str((len(cams) - 1) % ngpu) if ngpu else device
        print(f"[live] {cam_key}: geom from {clip}  "
              f"cam_pos_mm={room_frame['cameraPositionMm']}  gpu={cam_device}",
              flush=True)
        threading.Thread(
            target=infer_loop,
            args=(shared, geom, weights, cam_device, args.flip, mode, cam_key,
                  TimestampLog(cam_key, session), ids, recorder),
            daemon=True).start()
    if not cams:
        print("[live] FATAL: no usable cameras", file=sys.stderr)
        return 2

    if mode == "ava":
        # Deliberately NOT importing mmaction here to resolve the config path:
        # this process must stay free of mmcv/mmaction entirely. The worker
        # resolves its own defaults (ava_model.default_paths) inside its own
        # interpreter, so None means "let the worker decide".
        cfg = os.environ.get("SMARTROOM_AVA_CONFIG")
        ckpt = os.environ.get("SMARTROOM_AVA_CKPT") or str(
            Path.home() / "Code/yolo-bench/slowfast_ava.pth")
        lm = os.environ.get("SMARTROOM_AVA_LABELS") or str(
            Path(__file__).resolve().parent / "ava_label_map.txt")
        # AVA used to pin every camera to the LAST GPU, on the premise that this
        # kept it "off the pose devices". That held while there were two cameras
        # and three GPUs: pose took 0 and 1, AVA took 2. It broke silently as soon
        # as the cameras outnumbered the GPUs -- pose round-robins over ALL of
        # them, so with six cameras GPU 2 was running two pose models AND all six
        # AVA models (5.7 GB, 71% utilisation) while GPUs 0 and 1 idled at 5-7%.
        #
        # That imbalance is what made the service "freeze": cam1's pose predict,
        # on GPU 2, hung inside the head's topk for 60s until the wedge watchdog
        # killed the process, and the same device threw `CUDA error: misaligned
        # address` until the 300-failure limit did. Restarting reloads six models,
        # so each event blanked the live view for ~90s.
        #
        # With more cameras than GPUs, separation is not achievable -- so balance
        # instead. Offsetting AVA one device from pose keeps a camera's own two
        # models apart, and spreads six AVA models 2-per-GPU rather than 6-on-one.
        for i, (cam_key, e) in enumerate(cams.items()):
            adev = f"cuda:{(i + 1) % ngpu}" if ngpu else "cpu"
            print(f"[live] {cam_key}: AVA on {adev}", flush=True)
            threading.Thread(target=ava_loop,
                             args=(e["shared"], cfg, ckpt, lm, adev, AVA_THR),
                             daemon=True).start()
    elif mode in ("ntu", "hmdb"):
        for e in cams.values():
            threading.Thread(target=action_loop,
                             args=(e["shared"], args.width, args.height, mode),
                             daemon=True).start()

    threading.Thread(target=stall_watchdog, args=(cams,), daemon=True).start()
    threading.Thread(target=timing_driver, args=(cams,), daemon=True).start()

    offsets = reload_offsets(list(cams))
    measured = {k: v for k, v in offsets.items() if k in cams and v}
    print(f"[live] timeline: {_timing_note(measured)}", flush=True)
    threading.Thread(target=presenter, args=(cams,), daemon=True).start()
    pd = present_delay_ms()
    print(f"[live] presentation: " + (
        f"all cameras held to one instant, {pd / 1000:.2f}s behind live"
        if pd > 0 else
        "each camera shown on arrival (no offsets measured, so nothing to sync to)"),
        flush=True)

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(cams, ids))
    print(f"[live] serving on :{args.port}  cams={list(cams)}  action={mode}  "
          f"segments={_segment_mode_note()}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
