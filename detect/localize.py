#!/usr/bin/env python3
"""
Spatial localization pass: person room positions from pose keypoints + depth.

Runs AFTER detect.py (consumes its .keypoints.yolo26n-pose.json sidecars) on
every color clip that has pose keypoints plus embedded calibration/extrinsics.
For the RealSense streams the recorded lossless depth (camera_*_depth.mkv,
aligned to color) supplies each person's true range: the mid-hip pixel's depth
is back-projected through the factory intrinsics and rotated into the shared
AprilTag room frame. Clips without a depth stream (camera_main) fall back to
the monocular pixel->floor-plane ray (calib_utils.pixel_to_floor), as does any
sample where depth is missing/invalid.

Both cameras' outputs share the tag-1 room frame, so the dashboard can overlay
them on one map. Cross-camera fusion is deliberately NOT done here — each clip
gets its own sidecar; tracks are per-camera greedy nearest-neighbor chains.

Outputs per clip (same schema family as action.py so the map code is shared):
  <stem>.centroids.geo.json   persons{tid:[{t,x,y,room,src}]} + roomFrame
  <stem>.detections.geo.json  status sidecar so /api/saved exposes the model

Env:
  SMARTROOM_SAVE_DIR       recordings root (default: <project>/recordings)
  SMARTROOM_ROOM_KP_CONF   min keypoint confidence for localization (0.5)

Usage:
  python detect/localize.py [--force] [--path day_x/rec_y/streams/cam/clip.mp4 ...]
"""

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calib_utils import (ANKLE_JOINT_HEIGHT_MM, MAX_FLOOR_RANGE_MM,  # noqa: E402
                         load_room_geometry, pixel_to_floor, stream_entry)

SCHEMA_VERSION = 3   # v3: per-person `joints` (full 3D skeleton from depth)
MODEL_KEY = "geo"
KEYPOINTS_MODEL = "yolo26n-pose"
ROOM_KP_CONF = float(os.environ.get("SMARTROOM_ROOM_KP_CONF", "0.5"))

# COCO-17 joint indices
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16
# Trunk joints (shoulders + hips): the body's core, least likely to be all
# occluded at once, and the depths that best define its true range.
TRUNK = (5, 6, 11, 12)
N_KPTS = 17

DEPTH_PATCH = 3            # half-size: (2*3+1)^2 = 7x7 median patch
HW_PAIR_MAX_MS = 50.0      # color<->depth hw-timestamp pairing tolerance
PERSON_HEIGHT_MIN_MM = -200.0   # sanity band for the mid-hip point's room height
PERSON_HEIGHT_MAX_MM = 2200.0
# A joint whose depth is more than this NEARER than the body's median range is
# reading an occluder in front of the person, not the person — half a body
# thickness plus slack. Used both to find the robust range and to reject a
# joint's own depth in favour of the billboard fallback.
BODY_DEPTH_TOL_MM = 450.0
# Ankle floor-ray fallback (used only when a clip has NO depth). It triangulates
# a person's distance from where their ankle pixel meets the floor, which is
# only trustworthy when the camera looks DOWN at the floor. Our cameras look
# nearly horizontally down the room, so a distant ankle sits near the horizon
# where the ray grazes the floor and pixel noise becomes metres — that flung
# people through the walls. Disable the ray unless the camera is pitched down at
# least this much, and never trust a hit farther than this from the camera.
MIN_RAY_PITCH_DEG = 20.0
MAX_RAY_REACH_MM = 5500.0
TRACK_GATE_MM_PER_S = 4000.0    # association gate grows with the sample gap
TRACK_GATE_MIN_MM = 500.0
TRACK_DEAD_S = 1.0


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


def _atomic_write_json(path: Path, data: dict):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def sidecar_paths(mp4: Path):
    stem = mp4.stem
    return (mp4.parent / f"{stem}.detections.{MODEL_KEY}.json",
            mp4.parent / f"{stem}.centroids.{MODEL_KEY}.json")


def is_live_geo(mp4: Path) -> bool:
    """True when the geo sidecar was written live by the segment recorder from
    real depth. That is the best this clip can get — it has no depth track for
    the offline pass to use — so we must never overwrite it with a floor-ray
    guess."""
    _, centroids_path = sidecar_paths(mp4)
    try:
        return bool(json.loads(centroids_path.read_text()).get("live"))
    except (OSError, ValueError):
        return False


def needs_processing(mp4: Path, force: bool) -> bool:
    # Never clobber a live, depth-measured geo sidecar — not even on --force;
    # there is no depth track here to do better, only worse.
    if is_live_geo(mp4):
        return False
    if force:
        return True
    status_path, centroids_path = sidecar_paths(mp4)
    if not status_path.exists() or not centroids_path.exists():
        return True
    try:
        data = json.loads(status_path.read_text())
    except Exception:  # noqa: BLE001
        return True
    if data.get("status") != "done":
        return True
    if data.get("sourceMtimeMs", 0) + 2000 < mp4.stat().st_mtime * 1000:
        return True
    # re-run when the pose sidecar is newer than our output (fresh detect run)
    kp = mp4.parent / f"{mp4.stem}.keypoints.{KEYPOINTS_MODEL}.json"
    if kp.exists() and kp.stat().st_mtime > status_path.stat().st_mtime:
        return True
    return False


# --------------------------------------------------------- frame mapping ---

def container_pts_slots(mp4: Path):
    """Per-container-frame pts SLOT index. The Pi's hw encoder occasionally
    drops a frame mid-encode but leaves its time slot in the container, so
    slot index (round(pts/median_delta)) — not frame ordinal — is what lines
    up with the timestamps CSV rows."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True, timeout=120, check=True)
    pts = sorted(float(x) for x in out.stdout.split() if x.strip())
    if len(pts) < 2:
        return [0] * len(pts), 1 / 30.0
    deltas = sorted(b - a for a, b in zip(pts, pts[1:]))
    step = max(deltas[len(deltas) // 2], 1e-3)
    return [round(p / step) for p in pts], step


def read_csv_hw(path: Path):
    """hw_timestamp_ms column (list of float, one per CSV row) or None."""
    try:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        hw = [float(r["hw_timestamp_ms"]) for r in rows]
        return hw if hw and any(x > 0 for x in hw) else None
    except (OSError, KeyError, ValueError):
        return None


# ----------------------------------------------------------- depth reader ---

def read_depth_frames(mkv: Path, indices, width, height):
    """Decode the FFV1 depth mkv sequentially, keeping only `indices` (set of
    frame numbers) as uint16 arrays. One pass — the sampled frames are a tiny
    fraction of the clip."""
    wanted = set(indices)
    if not wanted:
        return {}
    frames = {}
    frame_bytes = width * height * 2
    last = max(wanted)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(mkv),
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
        stdout=subprocess.PIPE)
    try:
        idx = 0
        while idx <= last:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if idx in wanted:
                frames[idx] = np.frombuffer(buf, dtype=np.uint16).reshape(height, width)
            idx += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    return frames


def depth_at(depth, u, v, depth_scale_m):
    """Median valid depth (mm) in a small patch around pixel (u,v), or None."""
    h, w = depth.shape
    x, y = int(round(u)), int(round(v))
    if not (0 <= x < w and 0 <= y < h):
        return None
    patch = depth[max(0, y - DEPTH_PATCH):y + DEPTH_PATCH + 1,
                  max(0, x - DEPTH_PATCH):x + DEPTH_PATCH + 1]
    valid = patch[patch > 0]
    if valid.size < 3:  # mostly holes — don't trust it
        return None
    return float(np.median(valid)) * depth_scale_m * 1000.0


# ------------------------------------------------------------ geometry ---

def _ray_room(u, v, geom):
    """Unit-ish camera ray for pixel (u,v), in room-frame direction."""
    import cv2
    pt = np.array([[[float(u), float(v)]]], dtype=np.float64)
    n = cv2.undistortPoints(pt, geom["K"], geom["dist"]).reshape(2)
    return geom["R"] @ np.array([n[0], n[1], 1.0])


def backproject_cam(u, v, z_mm, geom):
    """Depth pixel + range -> room-frame point (mm), no sanity band. The range
    is along the camera's optical (Z) axis, matching how RealSense reports it."""
    d = _ray_room(u, v, geom)               # d[2 in cam] == 1 before rotation
    # d was built from [nx, ny, 1]; scaling by z_mm puts the point at optical
    # depth z_mm, then rotate+translate is already folded into d and cam_pos.
    return z_mm * d + geom["cam_pos_mm"]


def backproject_room(u, v, z_mm, geom):
    """Depth pixel -> room-frame point within the plausible person-height band,
    or None. Used for the footprint anchor (not per joint)."""
    p_room = backproject_cam(u, v, z_mm, geom)
    if not (PERSON_HEIGHT_MIN_MM <= p_room[1] <= PERSON_HEIGHT_MAX_MM):
        return None
    if z_mm > MAX_FLOOR_RANGE_MM:
        return None
    return p_room


def billboard_joint(u, v, geom, anchor_room):
    """Room point where the joint's ray meets the vertical plane through
    `anchor_room` (normal = horizontal camera->anchor). The occlusion-robust
    fallback for a joint with no trustworthy depth: it keeps the joint at the
    body's measured RANGE instead of collapsing it toward a nearer occluder."""
    cam = geom["cam_pos_mm"]
    horiz = np.array([anchor_room[0] - cam[0], 0.0, anchor_room[2] - cam[2]])
    ln = float(np.linalg.norm(horiz))
    if ln < 1e-6:
        return None
    nrm = horiz / ln
    d = _ray_room(u, v, geom)
    den = float(d @ nrm)
    if abs(den) < 1e-9:
        return None
    s = float((anchor_room - cam) @ nrm) / den
    if s <= 0:
        return None
    return cam + s * d


def joint_depths(person, dframe, depth_scale, width, height):
    """{joint idx: optical range mm} for every confident joint with valid depth."""
    out = {}
    for i in range(N_KPTS):
        px = joint_px(person, i, width, height)
        if px is None:
            continue
        z = depth_at(dframe, px[0], px[1], depth_scale)
        if z:
            out[i] = z
    return out


def robust_body_range(zj):
    """The body's true optical range (mm) from its joint depths, rejecting
    joints that read an occluder in front of it, or None when too few land.

    Prefers the trunk (shoulders+hips); a person's core is rarely all occluded
    and its depth is the cleanest. Takes the median, drops anything more than a
    body-thickness off it, and re-medians — so a single hip reading a desk edge
    can't drag the range toward the camera."""
    trunk = [zj[i] for i in TRUNK if i in zj]
    pool = trunk if len(trunk) >= 2 else list(zj.values())
    if not pool:
        return None
    med = float(np.median(pool))
    inliers = [z for z in pool if abs(z - med) <= BODY_DEPTH_TOL_MM]
    return float(np.median(inliers)) if inliers else med


def joint_px(person, idx, width, height):
    """(u, v) pixels for a conf-gated joint, or None."""
    if person["conf"][idx] < ROOM_KP_CONF:
        return None
    x, y = person["kpts"][idx]
    return x * width, y * height


def ground_point(person, width, height):
    """The pixel to ray-cast for the fallback path: ankle avg (conf-gated)."""
    pts = [p for p in (joint_px(person, L_ANKLE, width, height),
                       joint_px(person, R_ANKLE, width, height)) if p]
    if not pts:
        return None
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def hip_point(person, width, height):
    pts = [p for p in (joint_px(person, L_HIP, width, height),
                       joint_px(person, R_HIP, width, height)) if p]
    if not pts:
        return None
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


# -------------------------------------------------------------- tracking ---

class Tracks:
    """Greedy nearest-neighbor association in room space. Just enough to give
    the map stable polylines — not a real tracker (fusion comes later)."""

    def __init__(self):
        self.next_id = 0
        self.active = {}  # tid -> {"pos": (x,z), "t": last time}

    def assign(self, t, positions):
        """positions: list of (x_mm, z_mm). Returns list of track ids."""
        ids = [None] * len(positions)
        used = set()
        # match each detection to the nearest live track inside the gate
        order = sorted(range(len(positions)),
                       key=lambda i: min((self._dist(positions[i], tr["pos"])
                                          for tr in self.active.values()), default=1e18))
        for i in order:
            best_tid, best_d = None, 1e18
            for tid, tr in self.active.items():
                if tid in used or t - tr["t"] > TRACK_DEAD_S:
                    continue
                gate = max(TRACK_GATE_MIN_MM, TRACK_GATE_MM_PER_S * (t - tr["t"]))
                d = self._dist(positions[i], tr["pos"])
                if d < gate and d < best_d:
                    best_tid, best_d = tid, d
            if best_tid is None:
                best_tid = self.next_id
                self.next_id += 1
            used.add(best_tid)
            ids[i] = best_tid
            self.active[best_tid] = {"pos": positions[i], "t": t}
        # cull the dead
        self.active = {tid: tr for tid, tr in self.active.items()
                       if t - tr["t"] <= TRACK_DEAD_S}
        return ids

    @staticmethod
    def _dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ------------------------------------------------------------- per clip ---

def load_keypoints(mp4: Path):
    kp_path = mp4.parent / f"{mp4.stem}.keypoints.{KEYPOINTS_MODEL}.json"
    try:
        data = json.loads(kp_path.read_text())
    except (OSError, ValueError):
        return None
    if not data.get("frames"):
        return None
    return data


def depth_stream_for(mp4: Path):
    """(mkv path, depth csv path, depth_scale_m) for a camera_*_color clip,
    or None when this clip has no recorded depth."""
    stem = mp4.stem
    if not stem.endswith("_color"):
        return None
    base = stem[:-len("_color")]
    mkv = mp4.parent / f"{base}_depth.mkv"
    dcsv = mp4.parent / f"{base}_depth_timestamps.csv"
    if not mkv.exists() or not dcsv.exists():
        return None
    try:
        entry = stream_entry(mp4.parent / f"{base}_depth.mkv")
        scale = float(entry.get("depth_scale_m") or 0.001)
    except Exception:  # noqa: BLE001
        scale = 0.001
    return mkv, dcsv, scale


def process_clip(mp4: Path) -> bool:
    status_path, centroids_path = sidecar_paths(mp4)
    source_mtime_ms = mp4.stat().st_mtime * 1000
    _atomic_write_json(status_path, {
        "schemaVersion": SCHEMA_VERSION, "status": "analyzing",
        "model": MODEL_KEY, "source": mp4.name, "sourceMtimeMs": source_mtime_ms})

    kp = load_keypoints(mp4)
    if kp is None:
        _atomic_write_json(status_path, {
            "schemaVersion": SCHEMA_VERSION, "status": "error", "model": MODEL_KEY,
            "source": mp4.name, "sourceMtimeMs": source_mtime_ms,
            "error": "no pose keypoints sidecar (run detect.py first)"})
        return False

    try:
        cal = stream_entry(mp4).get("calibration") or {}
        width, height = int(cal.get("width") or 0), int(cal.get("height") or 0)
    except Exception:  # noqa: BLE001
        width = height = 0
    if not width or not height:
        # fall back to probing the clip
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(mp4)],
            capture_output=True, text=True, timeout=60)
        try:
            width, height = (int(x) for x in out.stdout.strip().split(","))
        except ValueError:
            width, height = 640, 480

    geom = load_room_geometry(mp4, width, height, undistorted=False)
    if geom is None:
        # not an error — the clip simply can't be located (no extrinsics etc.)
        _atomic_write_json(status_path, {
            "schemaVersion": SCHEMA_VERSION, "status": "error", "model": MODEL_KEY,
            "source": mp4.name, "sourceMtimeMs": source_mtime_ms,
            "error": "clip has no room geometry (extrinsics/intrinsics/tag height)"})
        return False

    native_fps = float(kp.get("nativeFps") or 30.0)

    # Is this camera pitched down enough for the ankle floor-ray to mean anything?
    # optical axis in the room frame; its downward angle below horizontal.
    optical_axis = geom["R"] @ np.array([0.0, 0.0, 1.0])
    pitch_down_deg = float(np.degrees(np.arcsin(np.clip(-optical_axis[1], -1.0, 1.0))))
    ray_ok = pitch_down_deg >= MIN_RAY_PITCH_DEG

    # --- depth plumbing (None for camera_main / missing depth) ---
    depth = depth_stream_for(mp4)
    depth_frames = {}
    color_row_to_depth_idx = {}
    slots = None
    if depth is not None:
        mkv, dcsv, depth_scale = depth
        color_hw = read_csv_hw(mp4.parent / f"{mp4.stem}_timestamps.csv")
        depth_hw = read_csv_hw(dcsv)
        if color_hw and depth_hw:
            try:
                slots, _ = container_pts_slots(mp4)
            except (subprocess.SubprocessError, OSError):
                slots = None
        if slots and color_hw and depth_hw:
            dhw = np.array(depth_hw)
            # container frame k lives at CSV row slots[k]
            needed = {}
            for fr in kp["frames"]:
                k = round(float(fr["t"]) * native_fps)
                if k >= len(slots):
                    continue
                row = slots[k]
                if row >= len(color_hw):
                    continue
                j = int(np.argmin(np.abs(dhw - color_hw[row])))
                if abs(dhw[j] - color_hw[row]) <= HW_PAIR_MAX_MS:
                    needed[k] = j
            color_row_to_depth_idx = needed
            depth_frames = read_depth_frames(mkv, needed.values(), width, height)
        else:
            depth = None  # no hw timestamps — ray fallback only

    # --- per sampled frame: localize every confident person ---
    tracks = Tracks()
    persons_out = {}
    n_samples = 0
    n_depth = n_ray = n_ray_rejected = 0
    for fr in kp["frames"]:
        t = float(fr["t"])
        n_samples += 1
        k = round(t * native_fps)
        dframe = None
        if depth is not None and k in color_row_to_depth_idx:
            dframe = depth_frames.get(color_row_to_depth_idx[k])

        found = []  # (position (x,z), entry dict)
        for person in fr.get("persons") or []:
            hip = hip_point(person, width, height)
            ank = ground_point(person, width, height)
            zj = (joint_depths(person, dframe, depth[2], width, height)
                  if dframe is not None else {})
            rng = robust_body_range(zj)

            # Anchor: the body's ground point. First choice is the robust
            # depth range along the mid-hip ray (true distance, occlusion-proof);
            # otherwise the ankle floor-ray. Everything hangs off this, so a bad
            # anchor is what used to shrink an occluded person to a dot.
            pos = src = anchor_room = None
            if rng is not None and hip is not None:
                a = backproject_room(hip[0], hip[1], rng, geom)
                if a is not None:
                    anchor_room, pos, src = a, (float(a[0]), float(a[2])), "depth-body"
            if pos is None and ank is not None and ray_ok:
                hit = pixel_to_floor(ank[0], ank[1], geom, ANKLE_JOINT_HEIGHT_MM)
                if hit is not None:
                    # The ankle floor-ray is only trustworthy when the camera
                    # actually looks DOWN at the floor. These cameras look nearly
                    # horizontally, so a distant person's ankle sits near the
                    # horizon where the ray grazes the floor and a 1px error
                    # becomes metres — that's what flung people through the walls.
                    # Reject a hit farther than a sane room radius from the camera.
                    reach = float(np.hypot(hit[0] - geom["cam_pos_mm"][0],
                                           hit[1] - geom["cam_pos_mm"][2]))
                    if reach <= MAX_RAY_REACH_MM:
                        anchor_room = np.array([hit[0], ANKLE_JOINT_HEIGHT_MM, hit[1]])
                        pos, src = hit, "ray-ankles"
                    else:
                        n_ray_rejected += 1
            if pos is None:
                continue

            # Full 3D skeleton: each confident joint gets its OWN depth when that
            # depth is consistent with the body's range, and a billboard point at
            # the body range when it's occluded, missing, or reading an occluder.
            joints3d = [None] * N_KPTS
            n_joint_depth = 0
            for i in range(N_KPTS):
                jp = joint_px(person, i, width, height)
                if jp is None:
                    continue
                z = zj.get(i)
                if z is not None and rng is not None and abs(z - rng) <= BODY_DEPTH_TOL_MM:
                    p = backproject_cam(jp[0], jp[1], z, geom)
                    n_joint_depth += 1
                else:
                    p = billboard_joint(jp[0], jp[1], geom, anchor_room)
                if p is not None:
                    joints3d[i] = [round(float(p[0]), 1), round(float(p[1]), 1),
                                   round(float(p[2]), 1)]

            px = hip or ank
            found.append((pos, {
                "t": round(t, 3),
                "x": round(px[0], 1), "y": round(px[1], 1),
                "room": [round(pos[0], 1), round(pos[1], 1)],
                "src": src,
                "joints": joints3d,           # 17 x [x,y,z] mm (room frame) or null
                "jointsWithDepth": n_joint_depth,
            }))
            if src == "depth-body":
                n_depth += 1
            else:
                n_ray += 1

        ids = tracks.assign(t, [f[0] for f in found])
        for tid, (_, entry) in zip(ids, found):
            persons_out.setdefault(str(tid), []).append(entry)

    duration = float(kp["frames"][-1]["t"]) if kp["frames"] else 0.0
    _atomic_write_json(centroids_path, {
        "schemaVersion": SCHEMA_VERSION,
        "model": MODEL_KEY,
        "source": mp4.name,
        "sourceMtimeMs": source_mtime_ms,
        "nativeFps": native_fps,
        "persons": persons_out,
        "roomFrame": {
            "origin": "floor point directly under the AprilTag's center",
            "axes": "X = tag's right (viewed facing the tag), Z = out of the wall; mm",
            "tagId": geom.get("tag_id"),
            "tagHeightMm": geom.get("tag_height_mm"),
            "cameraPositionMm": [round(float(v), 1) for v in geom["cam_pos_mm"]],
            "cameraId": geom.get("camera_id"),
        },
    })
    _atomic_write_json(status_path, {
        "schemaVersion": SCHEMA_VERSION, "status": "done", "model": MODEL_KEY,
        "source": mp4.name, "sourceMtimeMs": source_mtime_ms,
        "hasAnnotated": False,
        "framesAnalyzed": n_samples, "durationSec": round(duration, 2),
        "samplesDepth": n_depth, "samplesRay": n_ray, "samplesRayRejected": n_ray_rejected,
        "rayDisabled": not ray_ok, "cameraPitchDownDeg": round(pitch_down_deg, 1),
        "tracks": len(persons_out),
    })
    ray_note = (f", ray OFF (camera pitch {pitch_down_deg:.0f}deg < {MIN_RAY_PITCH_DEG:.0f})"
                if not ray_ok else
                (f", {n_ray_rejected} ray samples rejected as out-of-room" if n_ray_rejected else ""))
    print(f"  {mp4.parent.parent.parent.name}/{mp4.parent.parent.name}/{mp4.name}: "
          f"{len(persons_out)} track(s), {n_depth} depth / {n_ray} ray samples{ray_note}",
          file=sys.stderr)
    return True


# ------------------------------------------------------------------ main ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-run even when fresh")
    ap.add_argument("--path", nargs="*", help="specific clips (relative to recordings root)")
    args = ap.parse_args()

    root = saved_root()
    if not root.exists():
        print(f"no recordings dir: {root}", file=sys.stderr)
        return 0

    # Suffix lets GPU-sharded workers hold separate locks (see run-analysis.sh).
    sfx = os.environ.get("SMARTROOM_LOCK_SUFFIX", "")
    lock_file = open(root / f".localize.lock{sfx}", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another localize run is in progress; exiting", file=sys.stderr)
        return 0

    SOURCES = ("camera_main.mp4", "camera_d455_color.mp4", "camera_d435_color.mp4")
    if args.path:
        clips = [root / p for p in args.path]
    else:
        clips = sorted((p for name in SOURCES for p in root.rglob(name)),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    clips = [c for c in clips if c.exists() and "undistorted" not in c.parts]

    todo = [c for c in clips if needs_processing(c, args.force)]
    print(f"[{MODEL_KEY}] {len(todo)}/{len(clips)} clip(s) to localize", file=sys.stderr)
    errors = 0
    for mp4 in todo:
        try:
            if not process_clip(mp4):
                errors += 1
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            errors += 1
            status_path, _ = sidecar_paths(mp4)
            _atomic_write_json(status_path, {
                "schemaVersion": SCHEMA_VERSION, "status": "error", "model": MODEL_KEY,
                "source": mp4.name, "sourceMtimeMs": mp4.stat().st_mtime * 1000,
                "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR {mp4}: {exc}", file=sys.stderr)
    return 1 if errors and errors == len(todo) else 0


if __name__ == "__main__":
    raise SystemExit(main())
