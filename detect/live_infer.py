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
import datetime as dt
import json
import os
import re
import shutil
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
from localize import backproject_room, hip_point, joint_px  # noqa: E402

# COCO-17 shoulders (fallback anchor when the hips are occluded, e.g. seated at
# a desk). Both anchors are ranged by real depth — no floor-ray, never the feet.
L_SHOULDER, R_SHOULDER = 5, 6

BOUNDARY = "frame"
JPEG_QUALITY = 75
KP_CONF = float(os.environ.get("SMARTROOM_ROOM_KP_CONF", "0.5"))
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
# The depth back-channel polls at ~8Hz while the pose loop runs 30-60fps, and a
# sample must land near the anchor to match. A person who is STILL matches every
# frame; a MOVING one outruns the last sample, and the person used to be dropped
# from that frame entirely — vanishing from the overlay and the map, which reads
# as flicker. Hold their last known room position briefly instead.
POS_HOLD_S = float(os.environ.get("SMARTROOM_POS_HOLD_S", "0.7"))
ACTION_WINDOW = 48        # skeleton-window length (mirrors action.WINDOW); deque cap
ACTION_TRACK_TTL_S = 2.0  # drop a track's window/label if unseen this long
ACTION_SWEEP_S = 0.35     # how often the action thread re-classifies live tracks
AVA_SHORT = 256           # short-side the SlowFast-AVA clip is resized to
AVA_BUF = 128             # rolling RGB frame buffer (a few seconds at any fps)
AVA_PERIOD_S = 0.4        # how often to run the (heavier) AVA forward
AVA_THR = float(os.environ.get("SMARTROOM_AVA_THR", "0.4"))  # multi-label: every class above this is output
# Classes to suppress entirely (never output). ';'-separated (AVA names contain
# commas), case-insensitive exact match. Override/extend via SMARTROOM_AVA_BLACKLIST.
AVA_BLACKLIST = {s.strip().lower() for s in
                 os.environ.get("SMARTROOM_AVA_BLACKLIST",
                                "watch (a person);talk to (e.g., self, a person, a group)").split(";")
                 if s.strip()}
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
# mmaction's registry isn't safe to populate from two threads at once (a second
# concurrent build raced and failed with "MaxIoUAssignerAVA is not in the
# registry"), so per-camera model construction is serialized.
_MODEL_BUILD_LOCK = threading.Lock()

# --- person re-identification (stable identity across gaps and cameras) -------
# ByteTrack ids are per-camera and reset whenever a track is lost, so a person
# who is occluded, leaves, or is seen by the other camera gets a fresh id. The
# registry maps (cam, track id) -> a GLOBAL id using two signals:
#   1. geometry — both cameras localize into the same tag-2 room frame with
#      hw-synced timestamps (measured ~7cm agreement), so two detections at the
#      same room point at the same moment are the same person. Cheap + strong.
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
EMB_MOMENTUM = 0.9         # running-mean weight for a track's stored embedding

# --- continuous segment recording -------------------------------------------
# Always-on archival of the live feed in fixed-length segments, written straight
# into the recordings tree so the existing API/website list them with no extra
# plumbing. Segments containing nobody are deleted on close — an empty room is
# the overwhelming majority of wall-clock time and is not worth the disk.
SEGMENT_ON = os.environ.get("SMARTROOM_SEGMENT", "1") != "0"
SEGMENT_S = float(os.environ.get("SMARTROOM_SEGMENT_S", "180"))     # 3 minutes
# A couple of stray detections should not preserve an otherwise empty segment.
SEGMENT_MIN_PEOPLE_FRAMES = int(os.environ.get("SMARTROOM_SEGMENT_MIN_FRAMES", "15"))
# Encode on the GPU: two continuous libx264 streams alongside pose + AVA + a
# CPU-fallback ReID saturated the CPU and starved the HTTP server (requests
# queued behind a 190%-CPU process). NVENC is effectively free here.
SEGMENT_ENCODER = os.environ.get("SMARTROOM_SEGMENT_ENCODER", "h264_nvenc")
# Consecutive pose-predict failures before we give up and let systemd restart us.
PREDICT_FAIL_LIMIT = int(os.environ.get("SMARTROOM_PREDICT_FAIL_LIMIT", "60"))


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


class SegmentRecorder:
    """Encodes the incoming JPEG stream to fixed-length mp4 segments.

    Frames are handed off to a writer thread so a slow encoder can never stall
    inference. Segments align to wall-clock boundaries, so both cameras land in
    the SAME recording folder without needing to coordinate.
    """

    def __init__(self, cam_key: str, root: Path, stream_meta: dict, node: str,
                 room_frame: dict | None = None):
        self.cam, self.root, self.node = cam_key, root, node
        self.stream_meta = stream_meta or {}
        self.room_frame = room_frame
        self.q = deque()
        self.cond = threading.Condition()
        self.proc = None
        self.dir = None
        self.idx = None          # wall-clock segment index
        self.frames = 0
        self.people_frames = 0
        self.started = None
        self.rows = []           # (frame_no, hw_ts)
        self.kept = self.dropped = 0
        threading.Thread(target=self._run, daemon=True).start()

    def add(self, jpeg: bytes, hw_ts: float, has_people: bool):
        with self.cond:
            if len(self.q) < 240:          # ~8s at 30fps; never block inference
                self.q.append((jpeg, hw_ts, has_people))
                self.cond.notify()

    def _run(self):
        while True:
            with self.cond:
                while not self.q:
                    self.cond.wait(timeout=1.0)
                    if not self.q and self.proc and time.time() // SEGMENT_S != self.idx:
                        self._rotate(int(time.time() // SEGMENT_S))
                item = self.q.popleft()
            jpeg, hw_ts, people = item
            idx = int(time.time() // SEGMENT_S)
            if self.proc is None or idx != self.idx:
                self._rotate(idx)
            try:
                self.proc.stdin.write(jpeg)
                self.frames += 1
                self.rows.append((self.frames, hw_ts))
                if people:
                    self.people_frames += 1
            except (BrokenPipeError, OSError) as exc:
                print(f"[live] {self.cam}: segment write failed: {exc}", flush=True)
                self.proc = None

    def _rotate(self, idx: int):
        self._close()
        self.idx, self.frames, self.people_frames, self.rows = idx, 0, 0, []
        self.started = dt.datetime.now().astimezone()
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

    def _close(self):
        """Finish the current segment: keep it only if somebody was in it."""
        if self.proc is None or self.dir is None:
            return
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try: self.proc.kill()
            except Exception: pass
        self.proc = None
        rec_dir = self.dir.parent.parent          # .../rec_x
        mp4 = self.dir / f"{self.cam}.mp4"
        if self.people_frames < SEGMENT_MIN_PEOPLE_FRAMES:
            # nobody in it — discard, and remove the folder if the other camera
            # did not keep anything either.
            mp4.unlink(missing_ok=True)
            (self.dir / f"{self.cam}_timestamps.csv").unlink(missing_ok=True)
            self.dropped += 1
            # tidy up: cam dir -> streams dir -> rec dir. Stops as soon as one is
            # non-empty (i.e. the other camera kept its clip).
            for d in (self.dir, self.dir.parent, rec_dir):
                try: d.rmdir()
                except OSError: break
            return
        dur = max(1.0, self.frames / 30.0)
        with open(self.dir / f"{self.cam}_timestamps.csv", "w") as fh:
            fh.write("frame,hw_timestamp_ms\n")
            for n, ts in self.rows:
                fh.write(f"{n},{ts:.3f}\n")
        self._write_metadata(dur)
        self.kept += 1
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
            meta["duration_seconds"] = round(max(dur, meta.get("duration_seconds", 0)), 2)
            entry = dict(self.stream_meta)        # calibration + extrinsics
            entry.update({"path": f"{self.cam}.mp4",
                          "start_time": self.started.isoformat(),
                          "frame_count": self.frames,
                          "people_frames": self.people_frames})
            meta.setdefault("streams", {})[self.cam] = entry
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(meta, indent=2))
            tmp.replace(path)

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
    """Uploaded <cam_key>.mp4 clips with calibration+extrinsics, newest first.

    Returns every candidate, not just the newest: a clip can carry calibration
    yet still fail load_room_geometry (e.g. a recorded segment with no
    room_frame/tag height), and picking only the newest made startup fail
    outright instead of falling back to a usable one.
    """
    root = saved_root()
    out = []
    if not root.exists():
        return out
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
                out.append(mp4)
        except (OSError, ValueError):
            continue
    return out


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
        self.in_id = 0
        self.out_jpeg = None         # latest annotated JPEG
        self.out_id = 0
        self.positions = []          # [{id,x,z,src,cam,actions}]
        self.updated_ms = 0
        self.fps = 0.0
        self.hw_ts = 0.0             # sensor timestamp of the newest output frame
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

    def put_in(self, jpeg, hw_ts=0.0):
        with self.cond:
            self.in_jpeg = jpeg
            self.in_hw_ts = hw_ts
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

    def put_out(self, jpeg, positions, fps, hw_ts=0.0):
        with self.cond:
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
                stale_merge = (GEO_MERGE_MM > 0 and e["cam"] != cam
                               and t - e["t"] < GEO_MERGE_S
                               and ((pos[0] - e["pos"][0]) ** 2
                                    + (pos[1] - e["pos"][1]) ** 2) ** 0.5 > GEO_SPLIT_MM)
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
                if g in taken or e["cam"] == cam or t - e["t"] > GEO_MERGE_S:
                    continue
                d = ((pos[0] - e["pos"][0]) ** 2 + (pos[1] - e["pos"][1]) ** 2) ** 0.5
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

    def _touch(self, gid, emb, pos, t, cam):
        e = self.gallery.setdefault(gid, {"emb": emb, "pos": pos, "t": t, "cam": cam, "seen": 0})
        if emb is not None:
            e["emb"] = emb if e["emb"] is None else (
                EMB_MOMENTUM * e["emb"] + (1 - EMB_MOMENTUM) * emb)
        e["pos"], e["t"], e["cam"] = pos, t, cam
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
            live = [(g, e) for g, e in self.gallery.items() if t - e["t"] < GEO_MERGE_S]
            seen = set()
            for i in range(len(live)):
                for j in range(i + 1, len(live)):
                    ga, ea = live[i]
                    gb, eb = live[j]
                    if ea["cam"] == eb["cam"]:
                        continue          # one camera cannot see one person twice
                    key = (min(ga, gb), max(ga, gb))
                    d = ((ea["pos"][0] - eb["pos"][0]) ** 2
                         + (ea["pos"][1] - eb["pos"][1]) ** 2) ** 0.5
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
               mode: str, cam_key: str = "", tslog: "TimestampLog | None" = None,
               ids: "IdentityRegistry | None" = None,
               recorder: "SegmentRecorder | None" = None):
    from ultralytics import YOLO
    model = YOLO(weights)
    tracker = _make_bytetrack()
    jumps = JumpDetector()
    held = {}          # tid -> (pos, t) last good room position, for POS_HOLD_S
    encoder = None
    if ids is not None and REID_ON:
        try:
            with _MODEL_BUILD_LOCK:
                from ultralytics.trackers.utils.reid import ReID
                encoder = ReID(REID_MODEL,
                               device=("cpu" if device in ("cpu", "intel:cpu") else 0))
            print(f"[live] {cam_key}: ReID encoder {REID_MODEL} ready", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[live] {cam_key}: ReID unavailable ({exc}) — geometry only",
                  flush=True)
    frame_n = 0
    use_half = device not in ("cpu", "intel:cpu")
    last_id = 0
    ema_fps = 0.0
    predict_fails = 0
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
        t0 = time.time()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        h, w = frame.shape[:2]
        clean = frame.copy()   # pristine RGB for the action model (frame gets overlays)
        try:
            res = model.predict(frame, imgsz=640, device=device, half=use_half,
                                classes=[0], verbose=False)[0].cpu()
        except Exception as exc:  # noqa: BLE001
            print(f"[live] predict error: {exc}", flush=True)
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
                    held[tid] = (pos, t0)
            if pos is None:
                # no fresh depth this frame — hold the last known position rather
                # than dropping the person (that is what caused the flicker).
                prev = held.get(tid)
                if prev and t0 - prev[1] <= POS_HOLD_S:
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
                try:
                    dets = np.array([[(p["box"][0] + p["box"][2]) / 2,
                                      (p["box"][1] + p["box"][3]) / 2,
                                      p["box"][2] - p["box"][0],
                                      p["box"][3] - p["box"][1]]
                                     for *_, p, _ in found], dtype=np.float32)
                    embs = encoder(clean, dets)      # xywh, on the CLEAN frame
                except Exception as exc:  # noqa: BLE001
                    print(f"[live] {cam_key}: ReID embed failed: {exc}", flush=True)
            if any(e is not None for e in embs):
                ids.note(cam_key, [(t, e) for (t, *_), e in zip(found, embs)])
            ids.fuse(t0)      # continuously combine co-located identities
            taken = set()
            for (tid, pos, _marker, _p, _src), emb in zip(found, embs):
                gid, how = ids.assign(cam_key, tid, emb, pos, t0, taken)
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
            if jumps.update(tid, comy, body_h, t0):
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
            positions.append(entry)
            _draw_person(frame, p["px"], p["conf"], marker, tid, src, acts,
                         gid if gid is not None else tid)
        jumps.prune({tid for tid, *_ in found}, t0)
        for _t in [k for k, v in held.items() if t0 - v[1] > 5]:
            held.pop(_t, None)
        if ava:
            shared.push_ava(clean, ava_boxes, w, h)   # clean frame, NOT the annotated one

        if recorder is not None:
            # archive the ORIGINAL jpeg (no re-encode) with this frame's verdict
            recorder.add(jpeg, hw_ts, bool(positions))
        if tslog is not None:
            tslog.write(hw_ts, len(positions))

        dt = time.time() - t0
        ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
        cv2.putText(frame, f"{cam_key}  {len(positions)} person(s)  {ema_fps:4.1f} fps",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        ok, enc = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            shared.put_out(enc.tobytes(), positions, round(ema_fps, 1), hw_ts)


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

    with _MODEL_BUILD_LOCK:
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
                    if i in label_map and float(scores[j, i]) > action_thr
                    and label_map[i].lower() not in AVA_BLACKLIST]
            labs.sort(key=lambda x: -x[1])
            # multi-label: keep EVERY class above the threshold, not just top-1
            shared.set_label(tid, labs[0][0] if labs else None,
                             labs[0][1] if labs else 0.0,
                             [[a, round(s, 3)] for a, s in labs])


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
                    shared.put_in(jpeg, hw_ts)
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
            else:
                self.send_error(404)

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
                                    "persons": len(sh.positions),
                                    "roomFrame": e["roomFrame"],
                                    "recording": (e["recorder"].stats()
                                                  if e.get("recorder") else None)}
            first = cams[default_cam]["shared"]
            body = json.dumps({
                "positions": merged,
                "cams": per_cam,
                "identities": ids.stats() if ids is not None else None,
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
</style></head><body>
<h1>smartroom — live pose + room localization</h1>
<div class=wrap id=cards>
 <div class=card><div>Top-down room map (mm) — all cameras</div>
   <canvas id=map width=420 height=420></canvas>
   <div class=meta id=cnt></div></div>
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
      if(el)el.textContent='inference '+(cams[c].fps||0)+' fps · hw_ts '+
        (cams[c].hwTimestampMs||0).toFixed(0)+' · '+cams[c].persons+' person(s)';}
    const acts=pos.map(p=>'#'+(p.gid!=null?p.gid:p.id)+' ['+p.cam.replace('camera_','').replace('_color','')+
      (p.idSrc?'/'+p.idSrc:'')+']: '+((p.actions&&p.actions.length)?
      p.actions.map(a=>a[0]+' '+a[1].toFixed(2)).join(', '):'…'));
    const idn=d.identities?(' · '+d.identities.known+' known identities'):'';
    document.getElementById('cnt').innerHTML=pos.length+' detection(s)'+idn+'<br>'+
      (acts.length?acts.join('<br>'):'—');
  }catch(e){}
  setTimeout(poll,200);
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

    weights = os.environ.get("SMARTROOM_LIVE_WEIGHTS") or str(
        Path.home() / "Code/yolo-bench/yolo26n-pose.pt")
    device = os.environ.get("SMARTROOM_DETECT_DEVICE")
    if not device:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

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
                                    room_frame_meta) if SEGMENT_ON else None)
        shared = Shared()
        cams[cam_key] = {"shared": shared, "roomFrame": room_frame, "geom": geom,
                         "recorder": recorder}
        print(f"[live] {cam_key}: geom from {clip}  "
              f"cam_pos_mm={room_frame['cameraPositionMm']}", flush=True)
        threading.Thread(
            target=infer_loop,
            args=(shared, geom, weights, device, args.flip, mode, cam_key,
                  TimestampLog(cam_key, session), ids, recorder),
            daemon=True).start()
    if not cams:
        print("[live] FATAL: no usable cameras", file=sys.stderr)
        return 2

    if mode == "ava":
        import mmaction
        cfg = os.environ.get("SMARTROOM_AVA_CONFIG") or os.path.join(
            os.path.dirname(mmaction.__file__), ".mim", "configs", "detection",
            "slowfast", "slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.py")
        ckpt = os.environ.get("SMARTROOM_AVA_CKPT") or str(
            Path.home() / "Code/yolo-bench/slowfast_ava.pth")
        lm = os.environ.get("SMARTROOM_AVA_LABELS") or str(
            Path(__file__).resolve().parent / "ava_label_map.txt")
        # one recognizer per camera, spread over the available GPUs so a second
        # camera doesn't contend with the first (or with the pose loop on 0).
        ngpu = 0
        if device not in ("cpu", "intel:cpu"):
            try:
                import torch
                ngpu = torch.cuda.device_count()
            except Exception:  # noqa: BLE001
                ngpu = 0
        for i, (cam_key, e) in enumerate(cams.items()):
            adev = f"cuda:{i % ngpu}" if ngpu else "cpu"
            threading.Thread(target=ava_loop,
                             args=(e["shared"], cfg, ckpt, lm, adev, AVA_THR),
                             daemon=True).start()
    elif mode in ("ntu", "hmdb"):
        for e in cams.values():
            threading.Thread(target=action_loop,
                             args=(e["shared"], args.width, args.height, mode),
                             daemon=True).start()

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(cams, ids))
    print(f"[live] serving on :{args.port}  cams={list(cams)}  action={mode}  "
          f"segments={'on (%gs)' % SEGMENT_S if SEGMENT_ON else 'off'}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
