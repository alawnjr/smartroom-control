"""Undistortion from a recording's embedded camera calibration.

The Pis' capture.py embeds the camera's intrinsics (from checkerboard
calibration) into each recording's metadata.json under
streams.camera_main.calibration. Analysis on the laptop undistorts frames with
those values BEFORE detection / pose estimation, so keypoints and boxes live in
lens-corrected coordinates. Recordings without a calibration analyze exactly as
before (load_undistort_maps returns None -> no remap).
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

# Max believable floor-hit distance from the camera (mm); beyond this the ray is
# near-horizontal and a pixel of ankle error means meters of position error.
MAX_FLOOR_RANGE_MM = 15000.0
# The tag sits ON a wall, so nobody stands meaningfully behind its plane (Z < 0).
# Small negatives are legit — the tag's mount (breaker-panel door) sits proud of
# the actual wall — but beyond this the "ankle" was a mis-detection.
BEHIND_WALL_TOLERANCE_MM = 300.0


def analysis_source(mp4: Path) -> Path:
    """The file analysis should decode: the undistorted copy (written by
    undistort.py into streams/<cam>/undistorted/) when it exists and is at
    least as new as the raw clip and its calibration; else the raw clip."""
    out = mp4.parent / "undistorted" / mp4.name
    try:
        newest_src = max(mp4.stat().st_mtime, (mp4.parent / "metadata.json").stat().st_mtime)
        if out.stat().st_mtime >= newest_src:
            return out
    except OSError:
        pass
    return mp4


def stream_entry(mp4: Path) -> dict:
    """This clip's entry in metadata.json's streams{} — keyed by the clip's
    stem (camera_main, camera_d455_color, ...), matching capture.py."""
    meta = json.loads((mp4.parent / "metadata.json").read_text())
    return meta["streams"][mp4.stem]


def parse_intrinsics(cal: dict):
    """(K, dist, (w, h)) from either calibration schema: the checkerboard one
    (camera_matrix / dist_coeffs / image_size, from calibrate_camera.py) or the
    RealSense factory one (fx/fy/ppx/ppy/width/height; distortion negligible —
    the color sensor reports (inverse_)brown_conrady with tiny coefficients)."""
    if "camera_matrix" in cal:
        mtx = np.array(cal["camera_matrix"], dtype=np.float64)
        dist = np.array(cal["dist_coeffs"], dtype=np.float64)
        w, h = (int(v) for v in cal["image_size"])
    else:
        mtx = np.array([[cal["fx"], 0.0, cal["ppx"]],
                        [0.0, cal["fy"], cal["ppy"]],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        w, h = int(cal["width"]), int(cal["height"])
    return mtx, dist, (w, h)


def load_undistort_maps(mp4: Path, frame_w: int, frame_h: int):
    """(map1, map2) for cv2.remap, or None when this clip has no calibration
    (or none worth applying — RealSense factory distortion is negligible).

    Reads the metadata.json saved next to the clip. If the calibration was done
    at a different resolution than the clip, fx/fy/cx/cy are scaled (distortion
    coefficients are resolution-invariant). The undistorted image keeps the same
    camera matrix, so image size and framing are unchanged.
    """
    try:
        cal = stream_entry(mp4)["calibration"]
        mtx, dist, (cal_w, cal_h) = parse_intrinsics(cal)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not np.any(dist):
        return None  # zero distortion — remapping would be a no-op
    if mtx.shape != (3, 3) or not frame_w or not frame_h or not cal_w or not cal_h:
        return None
    if (cal_w, cal_h) != (frame_w, frame_h):
        sx, sy = frame_w / cal_w, frame_h / cal_h
        mtx = mtx.copy()
        mtx[0, 0] *= sx  # fx
        mtx[0, 2] *= sx  # cx
        mtx[1, 1] *= sy  # fy
        mtx[1, 2] *= sy  # cy
    return cv2.initUndistortRectifyMap(mtx, dist, None, mtx,
                                       (int(frame_w), int(frame_h)), cv2.CV_16SC2)


# ---------------------------------------------------------------------------
# Room geometry: pixel -> metric floor position in the AprilTag frame.
#
# calibrate_extrinsics.py (on the Pi) embeds the camera's pose relative to the
# wall-mounted AprilTag into metadata.json (streams.camera_main.extrinsics).
# OpenCV's aruco module returns AprilTag corners in an order that leaves that
# frame rolled 180° about the tag normal — its Y axis points physically DOWN.
# We normalize here with F = diag(-1,-1,1) rather than in the Pi script, so the
# extrinsics already embedded in existing recordings stay valid. Output frame:
#   origin = the floor point directly under the tag center
#   X = tag's right (viewed facing the tag), Y = up, Z = out of the wall; mm.
# The tag-center height above the floor comes from SMARTROOM_TAG_HEIGHT_MM or
# recordings/.floor.json (written by floor_calib.py's known-stature solve).
# ---------------------------------------------------------------------------

_ROLL_FIX = np.diag([-1.0, -1.0, 1.0])


def _floor_config(mp4: Path) -> dict:
    """Contents of the recordings root's .floor.json ({} when absent)."""
    for parent in mp4.resolve().parents:
        cfg = parent / ".floor.json"
        if cfg.exists():
            try:
                return json.loads(cfg.read_text())
            except (OSError, ValueError):
                return {}
    return {}


def tag_height_mm(mp4: Path, camera_id: str):
    env = os.environ.get("SMARTROOM_TAG_HEIGHT_MM")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    entry = _floor_config(mp4).get(camera_id)
    if isinstance(entry, dict) and entry.get("tag_height_mm"):
        return float(entry["tag_height_mm"])
    # Fall back to the recording's own metadata: capture.py embeds the room
    # frame's tag height (room_frame.tag_center_above_floor_mm) in every
    # recording. .floor.json only covers cameras that went through the
    # person-stature calibration (the webcams) — without this fallback the
    # RealSense clips can never be located.
    try:
        meta = json.loads((mp4.parent / "metadata.json").read_text())
        height = (meta.get("room_frame") or {}).get("tag_center_above_floor_mm")
        if height:
            return float(height)
    except (OSError, ValueError):
        pass
    return None


def load_room_geometry(mp4: Path, frame_w: int, frame_h: int, undistorted: bool):
    """Everything pixel_to_floor needs, or None when this clip can't be located
    (no extrinsics, no intrinsics, or no known tag height).

    `undistorted` says whether pixels will come from the lens-corrected copy
    (analysis_source) — if not, distortion is removed per-point instead.
    """
    try:
        stream = stream_entry(mp4)
        cal, ext = stream["calibration"], stream["extrinsics"]
        mtx, dist, (cal_w, cal_h) = parse_intrinsics(cal)
        R = np.array(ext["rotation_cam_to_room"], dtype=np.float64)
        cam = np.array(ext["camera_position_mm"], dtype=np.float64)
        camera_id = ext.get("camera_id", "")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if mtx.shape != (3, 3) or R.shape != (3, 3) or cam.shape != (3,):
        return None
    height = tag_height_mm(mp4, camera_id)
    if height is None or not frame_w or not frame_h or not cal_w or not cal_h:
        return None
    if (cal_w, cal_h) != (frame_w, frame_h):
        sx, sy = frame_w / cal_w, frame_h / cal_h
        mtx = mtx.copy()
        mtx[0, 0] *= sx
        mtx[0, 2] *= sx
        mtx[1, 1] *= sy
        mtx[1, 2] *= sy
    R_up = _ROLL_FIX @ R
    cam_up = _ROLL_FIX @ cam
    cam_up[1] += height  # origin: floor under the tag, not the tag center
    return {
        "K": mtx,
        "dist": None if undistorted else dist,
        "R": R_up,                  # camera frame -> upright room frame
        "cam_pos_mm": cam_up,       # camera position, origin at tag's floor point
        "tag_height_mm": height,
        "tag_id": (ext.get("tag") or {}).get("id"),
        "camera_id": camera_id,
    }


# The pose model's ankle keypoint is the ankle JOINT, ~75mm above the sole; a
# ray through it must be cut at that height, not at the floor, or the position
# overshoots away from the camera along shallow rays.
ANKLE_JOINT_HEIGHT_MM = 75.0


def pixel_to_floor(u: float, v: float, geom: dict, plane_height_mm: float = 0.0):
    """(x_mm, z_mm) where the ray through pixel (u, v) crosses the horizontal
    plane plane_height_mm above the floor, or None when it doesn't (pointing
    up / behind the camera / farther than MAX_FLOOR_RANGE_MM)."""
    pt = np.array([[[float(u), float(v)]]], dtype=np.float64)
    n = cv2.undistortPoints(pt, geom["K"], geom["dist"]).reshape(2)
    d = geom["R"] @ np.array([n[0], n[1], 1.0])
    cam = geom["cam_pos_mm"]
    if d[1] >= -1e-9:  # must point downward to reach the plane
        return None
    t = (plane_height_mm - cam[1]) / d[1]
    if t <= 0:
        return None
    hit = cam + t * d
    if float(np.linalg.norm(hit - cam)) > MAX_FLOOR_RANGE_MM:
        return None
    if hit[2] < -BEHIND_WALL_TOLERANCE_MM:
        return None
    return float(hit[0]), float(hit[2])
