"""Undistortion from a recording's embedded camera calibration.

The Pis' capture.py embeds the camera's intrinsics (from checkerboard
calibration) into each recording's metadata.json under
streams.camera_main.calibration. Analysis on the laptop undistorts frames with
those values BEFORE detection / pose estimation, so keypoints and boxes live in
lens-corrected coordinates. Recordings without a calibration analyze exactly as
before (load_undistort_maps returns None -> no remap).
"""

import json
from pathlib import Path

import cv2
import numpy as np


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


def load_undistort_maps(mp4: Path, frame_w: int, frame_h: int):
    """(map1, map2) for cv2.remap, or None when this clip has no calibration.

    Reads the metadata.json saved next to the clip. If the calibration was done
    at a different resolution than the clip, fx/fy/cx/cy are scaled (distortion
    coefficients are resolution-invariant). The undistorted image keeps the same
    camera matrix, so image size and framing are unchanged.
    """
    try:
        meta = json.loads((mp4.parent / "metadata.json").read_text())
        cal = meta["streams"]["camera_main"]["calibration"]
        mtx = np.array(cal["camera_matrix"], dtype=np.float64)
        dist = np.array(cal["dist_coeffs"], dtype=np.float64)
        cal_w, cal_h = (int(v) for v in cal["image_size"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
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
