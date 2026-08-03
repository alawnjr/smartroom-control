#!/usr/bin/env python3
"""
Cross-camera timing sync: measure and store each live camera's delay to the
server, so detections from different cameras can be compared at the same moment.

WHY THIS EXISTS. live_infer fuses one person seen by two cameras by asking
whether both cameras put them at the same room point *at the same time*. It used
to take "the same time" from `time.time()` at the moment the inference loop
picked the frame up -- which is not when the frame was captured, and is later by
a different amount for every camera:

    RealSense (Pi)   sensor -> depth page -> live_forward -> LAN  -> ingest
    Reolink   (NVR)  sensor -> NVR encode+buffer -> RTSP -> ffmpeg -> another
                     host -> ingest

The RealSense cameras share one path so their errors cancel; the Reolink path
adds the NVR's encode and buffer latency on top of a second host's network, and
that does NOT cancel against anything. A constant lag L makes the two cameras'
views of a walking person compared L out of step: at 1.4 m/s, L = 300 ms is
42 cm of pure fiction, against a GEO_MERGE_MM of 600. The fusion then either
misses a real pair or matches the wrong person.

WHAT IS MEASURED. One number per camera: milliseconds by which THIS camera's
frames arrive later than the reference camera's. Subtract it from the arrival
time and every camera lands on one comparable timeline. The reference is 0 by
definition -- only differences matter, and no camera here can report a true
capture instant that the server can trust.

WHY ARRIVAL TIME AND NOT THE SENSOR CLOCK. librealsense does give real
mid-exposure timestamps, and for two RealSense on one bus they are the better
answer (that is what the Pi's own calibration/camera_timing.json is for). They
are useless for this job: RTSP carries no such clock, so the Reolink forwarder
sends its own host's wall clock, and mixing a sensor clock on one camera with
two different hosts' wall clocks on the others means the numbers are only as
aligned as NTP happens to be. Arrival at the server is ONE clock for every
camera, and the measurement below absorbs everything upstream of it -- sensor
latency, encode, buffer and network -- into the single number that fusion needs.
The cost is arrival jitter, which is why the calibration reports how much of it
there was instead of pretending the answer is exact.

HOW IT IS MEASURED: LIGHTS ON/OFF. Flip the room lights a few times. A light
change hits every pixel of every camera in the room simultaneously, whatever
each camera is pointed at -- so no shared field of view is required, which
matters because these six cameras do not have one. Each camera reports a
frame-difference energy series against its own arrival times; the lag that
maximizes the cross-correlation of two cameras' series IS the delay between
them. Waving a hand works too, but only for cameras that can both see the hand.

Stored in calibration/live_timing.json (machine-generated, gitignored) and read
back by live_infer at startup; SMARTROOM_TIME_OFFSET_<CAM_KEY> overrides one
camera from the environment.

    python detect/timing_sync.py --show        # print the stored offsets
    python detect/timing_sync.py --selftest    # check the correlation math
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A camera whose series barely correlates with the reference has not measured
# anything -- the lights did not change, or it was looking at a wall the lights
# do not reach. Refusing is right: a wrong offset is worse than none, because it
# silently shifts a camera that was previously merely uncalibrated.
MIN_CORRELATION = float(os.environ.get("SMARTROOM_TIMING_MIN_CORR", "0.35"))
# Search window. The Reolink path measured in the hundreds of ms; 2s is far
# past any plausible value and keeps a spurious far-off peak from winning.
MAX_LAG_MS = float(os.environ.get("SMARTROOM_TIMING_MAX_LAG_MS", "2000"))
# Resample grid. 5ms is finer than one frame interval at any rate these cameras
# run, so the peak's position is limited by the data and not by this.
GRID_MS = 5.0
# Minimum window two cameras must share. This is what makes MIN_CORRELATION mean
# something: the peak is a maximum over ~800 candidate lags, so an uncorrelated
# pair scores well above zero, and how far above depends entirely on how much data
# there is. Measured worst-of-12 noise peaks against a real signal of ~0.85:
#
#     window     5s     8s    10s    15s    20s    25s    30s
#     noise   0.398  0.315  0.275  0.296  0.270  0.197  0.167
#
# At 5s the noise floor is ABOVE the 0.35 bar, so a camera that saw nothing could
# pass it — which is how a static-scene camera scored 0.31 in a 5s test run. From
# 10s the floor is clear of the bar; the default 25s window leaves ~4x margin.
MIN_OVERLAP_MS = 10000.0
# ...and enough frames within it. A camera delivering under ~6fps has not sampled
# the light switch well enough to place it, whatever the window length.
MIN_SAMPLES = 60
# Gray levels (0-255) a camera's mean brightness must swing across the window for
# it to count as having SEEN the lights change.
#
# The correlation alone cannot tell what it correlated ON. A first live run
# measured the D435 at +112.8 ms with r=0.73 while all four Reolink cameras were
# rejected — from cameras in the SAME room with the ceiling lights in frame, which
# is impossible if the lights actually flipped. What the two RealSense had in
# common was a seated person moving, not a light switch, and the run would happily
# have published that as a timing offset.
#
# Brightness is the direct test: a light switch moves the whole frame's mean by
# tens of levels, while a person moving across a static room moves it by ~1-2.
# Deliberately well below a real switch (40+ levels) because auto-exposure claws
# some of the swing back within a second or so.
MIN_BRIGHTNESS_SWING = float(os.environ.get("SMARTROOM_TIMING_MIN_SWING", "12"))


def timing_path() -> Path:
    return Path(os.environ.get("SMARTROOM_LIVE_TIMING")
                or (PROJECT_ROOT / "calibration" / "live_timing.json"))


def _env_key(cam_key: str) -> str:
    return "SMARTROOM_TIME_OFFSET_" + cam_key.upper()


def load_capture_clocks() -> dict:
    """{cam_key: "hw" | "arrival"} — which clock to take that camera's frame time
    from. Absent means "arrival", the only thing available before a calibration."""
    try:
        data = json.loads(timing_path().read_text())
        return {str(k): ("hw" if v == "hw" else "arrival")
                for k, v in (data.get("capture_clocks") or {}).items()}
    except (OSError, ValueError, TypeError):
        return {}


def load_offsets() -> dict:
    """{cam_key: offset_ms} — subtract from that camera's arrival times.

    Missing file, unreadable file and missing camera all mean 0.0: an
    uncalibrated system must behave exactly as it did before this existed.
    """
    out = {}
    try:
        data = json.loads(timing_path().read_text())
        for cam, ms in (data.get("offsets_ms") or {}).items():
            out[str(cam)] = float(ms)
    except (OSError, ValueError, TypeError):
        pass
    return out


def offset_ms_for(cam_key: str, stored: dict) -> float:
    """Stored offset for one camera, with the environment winning.

    The env override exists so a known-bad measurement can be neutralised (or a
    hand-measured value pinned) without editing a machine-generated file.
    """
    raw = os.environ.get(_env_key(cam_key))
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass
    try:
        return float(stored.get(cam_key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def correlate(series_a, series_b, grid_ms=GRID_MS, max_lag_ms=MAX_LAG_MS):
    """(offset_ms, peak_correlation, overlap_ms) for two (time_ms, energy) series.

    offset_ms > 0 means series B's frames carry the same physical event LATER
    than A's — i.e. B is the laggier camera and B's times need it subtracted.

    Both series are resampled onto a common grid, then compared by PEARSON
    correlation at each candidate lag: the cameras have different exposures, gains
    and resolutions, so only the SHAPE of each energy curve is comparable, never
    its magnitude, and Pearson is invariant to both.

    The correlation is recomputed over each lag's overlapping window rather than
    z-scoring the whole series once and taking a sliding dot product. The cheaper
    version is fine for LOCATING the peak but its value is not a correlation — the
    sliced windows are no longer zero-mean unit-variance, so it can exceed 1.0
    (measured: 1.07) and the "is this camera correlated enough to trust" threshold
    was being applied to an unbounded number. That let a camera watching a static
    scene through sensor noise pass at exactly the threshold and be assigned a
    -1168 ms offset it had no basis for.
    """
    if len(series_a) < MIN_SAMPLES or len(series_b) < MIN_SAMPLES:
        raise ValueError("too few frames — was the camera streaming?")
    ta = np.asarray([p[0] for p in series_a], dtype=float)
    ea = np.asarray([p[1] for p in series_a], dtype=float)
    tb = np.asarray([p[0] for p in series_b], dtype=float)
    eb = np.asarray([p[1] for p in series_b], dtype=float)
    t0, t1 = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    overlap = t1 - t0
    if overlap < MIN_OVERLAP_MS:
        raise ValueError(f"cameras overlapped for only {overlap / 1000:.1f}s, "
                         f"under the {MIN_OVERLAP_MS / 1000:.0f}s a trustworthy "
                         "correlation needs — rerun for longer")
    grid = np.arange(t0, t1, grid_ms)
    a = np.interp(grid, ta, ea)
    b = np.interp(grid, tb, eb)
    if a.std() < 1e-9 or b.std() < 1e-9:
        raise ValueError("a camera saw no change at all — flip the lights")

    n = len(grid)
    span = int(max_lag_ms / grid_ms)
    lags = np.arange(-span, span + 1)
    corr = np.full(len(lags), -2.0)
    for i, lag in enumerate(lags):
        # b shifted later than a by `lag` samples
        if lag >= 0:
            x, y = a[: n - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: n + lag]
        m = len(x)
        if m <= MIN_SAMPLES:
            continue
        sx, sy = x.std(), y.std()
        if sx < 1e-9 or sy < 1e-9:
            continue
        corr[i] = float(((x * y).mean() - x.mean() * y.mean()) / (sx * sy))
    if not np.any(corr > -2.0):
        raise ValueError("no comparable window between these cameras — rerun")
    i = int(np.argmax(corr))
    best_lag, best_corr = float(lags[i]), float(corr[i])
    # Sub-grid refinement: a sharp common edge (a light switch) peaks between two
    # samples, and a 3-point parabola through the peak recovers where.
    if 0 < i < len(lags) - 1 and corr[i - 1] > -2.0 and corr[i + 1] > -2.0:
        y0, y1, y2 = corr[i - 1], corr[i], corr[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom
            if abs(shift) <= 1.0:
                best_lag += shift
    return best_lag * grid_ms, best_corr, float(overlap)


def brightness_swing(series) -> float:
    """p95 - p05 of this camera's mean frame brightness across the window.

    Percentiles rather than max-min: one glint or a passing reflection should not
    look like the room lights coming on. Returns 0.0 for a series recorded without
    brightness (nothing to test, so nothing is claimed).
    """
    vals = [row[2] for row in series if len(row) > 2]
    if len(vals) < 5:
        return 0.0
    v = np.sort(np.asarray(vals, dtype=float))
    return float(v[int(0.95 * (len(v) - 1))] - v[int(0.05 * (len(v) - 1))])


def _has_brightness(series) -> bool:
    return any(len(row) > 2 for row in series)


# --- primary estimator: match the light switches themselves ------------------
# Cross-correlating difference energy has two limits that both bit on real
# cameras. It can only find a lag inside its search window, and it needs the
# EDGE SHAPE to survive: the Reolink sub-stream arrives at 10fps through the
# NVR's encoder, so its energy spike is blunter and differently smeared than the
# D455's at 15fps. Measured result: all four Reolink cameras cleared the
# brightness gate (they plainly saw the room go dark) yet correlated at only
# 0.17-0.20 and were rejected, while the two RealSense correlated fine.
#
# The brightness STEP is the far better signal for this event. A light switch is
# a step, its 50% crossing is a well-defined instant, and locating it needs no
# assumption about lag magnitude at all -- which matters because the NVR path
# could plausibly be seconds, i.e. outside any window worth searching by
# correlation.
EDGE_TOL_MS = float(os.environ.get("SMARTROOM_TIMING_EDGE_TOL_MS", "150"))
EDGE_MAX_LAG_MS = float(os.environ.get("SMARTROOM_TIMING_EDGE_MAX_LAG_MS", "10000"))
# Two. It is tempting to demand three -- a 2-edge vote is thin -- but on real data
# the cameras do not all detect the same transitions: over one 22s run with 4 flips
# the D455 resolved 5 edges, the D435 6, cam4 5, and only 2-3 of them lined up per
# camera. Demanding 3 would have REJECTED the D435's +2732 ms, which independent
# observation (a hand waved in front of both cameras) confirmed is correct.
#
# The real protection against a thin vote is the ambiguity check in
# offset_from_edges, which refuses when a rival alignment explains just as many
# edges -- that is what actually distinguishes a sparse-but-unique answer from a
# coin toss between offsets one flip apart. Matches below this are reported as
# low-confidence rather than silently trusted.
MIN_EDGES_MATCHED = 2
CONFIDENT_EDGES = 3
MAX_EDGE_SPREAD_MS = 250.0


def light_edges(series, min_swing=None):
    """[(time_ms, +1 on / -1 off)] for each light transition this camera saw.

    A hysteresis state machine on mean brightness: the signal has to reach the
    far quarter of its own range before the next edge counts, so flicker around
    the midpoint cannot manufacture edges. The reported time is the interpolated
    50% crossing, which is the closest thing to the instant the switch was
    thrown, and is independent of each camera's exposure and gain.
    """
    min_swing = MIN_BRIGHTNESS_SWING if min_swing is None else min_swing
    rows = [r for r in series if len(r) > 2]
    if len(rows) < 5:
        return []
    t = np.asarray([r[0] for r in rows], dtype=float)
    b = np.asarray([r[2] for r in rows], dtype=float)
    srt = np.sort(b)
    lo = float(srt[int(0.05 * (len(srt) - 1))])
    hi = float(srt[int(0.95 * (len(srt) - 1))])
    if hi - lo < min_swing:
        return []
    mid = (lo + hi) / 2.0
    hi_th, lo_th = lo + 0.75 * (hi - lo), lo + 0.25 * (hi - lo)

    def crossing(i):
        """Interpolated time at which b crossed `mid` at or before index i."""
        j = i
        while j > 0 and (b[j] > mid) == (b[i] > mid):
            j -= 1
        if j == i or b[j] == b[j + 1]:
            return float(t[i])
        frac = (mid - b[j]) / (b[j + 1] - b[j])
        frac = min(max(frac, 0.0), 1.0)
        return float(t[j] + frac * (t[j + 1] - t[j]))

    state = "high" if b[0] > mid else "low"
    edges = []
    for i in range(1, len(b)):
        if state == "low" and b[i] >= hi_th:
            edges.append((crossing(i), 1))
            state = "high"
        elif state == "high" and b[i] <= lo_th:
            edges.append((crossing(i), -1))
            state = "low"
    return edges


def _align_edges(ref_edges, cam_edges, flip, tol_ms):
    """Best (matched, -spread, offset_ms) alignment under one polarity hypothesis.

    Every compatible pairing is a candidate offset, scored by how many OTHER edges
    it also aligns -- a vote, not a fit, so one spurious edge cannot drag the answer.
    """
    best = None
    for tc, dc in cam_edges:
        want = -dc if flip else dc
        for tr, dr in ref_edges:
            if dr != want:
                continue
            cand = tc - tr
            if abs(cand) > EDGE_MAX_LAG_MS:
                continue
            diffs = []
            for tc2, dc2 in cam_edges:
                w2 = -dc2 if flip else dc2
                same = [tr2 for tr2, dr2 in ref_edges
                        if dr2 == w2 and abs((tc2 - tr2) - cand) <= tol_ms]
                if same:
                    nearest = min(same, key=lambda x: abs((tc2 - x) - cand))
                    diffs.append(tc2 - nearest)
            if not diffs:
                continue
            spread = (max(diffs) - min(diffs)) if len(diffs) > 1 else 0.0
            candidate = (len(diffs), -spread, float(np.median(diffs)))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return best


def offset_from_edges(ref_edges, cam_edges, tol_ms=EDGE_TOL_MS):
    """(offset_ms, matched, spread_ms, inverted) by aligning two cameras' switches.

    BOTH POLARITIES ARE TRIED, because one camera's mean brightness need not move
    the same way as another's -- aggressive auto-exposure ramps gain when the room
    goes dark and can overshoot. Whichever hypothesis explains MORE edges wins, so a
    camera behaving normally is unaffected: this opens a door, it does not push
    anything through it.

    An ambiguous winner is refused rather than picked. With sparse edges and
    evenly-spaced flips, an alignment off by one whole flip scores almost as well as
    the truth, and guessing between them silently produces an answer wrong by the
    flip interval. This, not a minimum edge count, is the real guard: on real data
    the cameras disagree about which transitions they even saw, so a correct answer
    is often supported by only two or three edges.
    """
    if len(ref_edges) < MIN_EDGES_MATCHED or len(cam_edges) < MIN_EDGES_MATCHED:
        raise ValueError(f"only {min(len(ref_edges), len(cam_edges))} light "
                         f"transition(s) seen, need {MIN_EDGES_MATCHED} — flip the "
                         "lights more times")
    straight = _align_edges(ref_edges, cam_edges, False, tol_ms)
    inverted = _align_edges(ref_edges, cam_edges, True, tol_ms)
    options = [(o, f) for o, f in ((straight, False), (inverted, True)) if o]
    if not options:
        raise ValueError("this camera's light transitions did not line up with the "
                         "reference's — flip the lights more times, more slowly")
    options.sort(key=lambda of: of[0][:2], reverse=True)
    best, flip = options[0]
    matched, spread, off = best[0], -best[1], best[2]
    if matched < MIN_EDGES_MATCHED:
        raise ValueError(f"only {matched} of its {len(cam_edges)} light transitions "
                         f"line up with the reference (need {MIN_EDGES_MATCHED}) — "
                         "flip the lights more times")
    # A rival alignment that explains just as many edges means the answer is a
    # coin toss between offsets a whole flip apart. Uneven gaps between flips
    # break the tie, so that is what to ask for.
    rivals = [o for o, _ in options[1:] if o[0] >= matched
              and abs(o[2] - off) > tol_ms]
    if rivals:
        raise ValueError(
            f"two different delays ({off:+.0f} ms and {rivals[0][2]:+.0f} ms) explain "
            "this camera's light transitions equally well — flip the lights at UNEVEN "
            "intervals (e.g. 1s, 4s, 2s, 6s) so only one can fit")
    if spread > MAX_EDGE_SPREAD_MS:
        raise ValueError(f"its light transitions disagree by up to {spread:.0f} ms, "
                         "so it has no single delay (dropped frames or a variable "
                         "buffer) — rerun, and check that camera's frame rate")
    return off, matched, spread, flip


# A camera whose own timestamp is a TRUE capture clock will place the light switches
# at the same instants as the reference's does — that is what "same clock domain"
# means. One whose timestamp is merely a receive time will place them late by
# whatever happens upstream of it (the NVR's buffer), so it will not agree.
#
# This matters because a constant offset can only correct a constant delay. The
# D435's delay is NOT constant: the Pi holds a 60-frameset queue (~2s at 30fps) to
# keep recordings whole, and its depth follows the load, so the same camera measured
# 228ms under light load and ~2.7s with six cameras and people in the room. Its
# librealsense timestamp tracks that per frame, and using it is the only way to stay
# in step. Cameras with no such clock keep the constant, which is all they have.
HW_CLOCK_TOL_MS = float(os.environ.get("SMARTROOM_HW_CLOCK_TOL_MS", "300"))


def _view(series, time_index):
    """The same series expressed on another of its own time columns."""
    out = []
    for r in series:
        if len(r) > time_index and r[time_index]:
            out.append((r[time_index],) + tuple(r[1:]))
    return out


def _jitter_ms(series) -> float:
    """Spread of this camera's frame intervals — how steady its arrivals are.

    Reported next to every offset because it bounds what the offset can mean: a
    camera whose frames arrive in bursts does not HAVE one constant delay, and a
    single number for it is an average, not a correction.
    """
    if len(series) < 3:
        return 0.0
    d = np.diff(np.asarray([p[0] for p in series], dtype=float))
    return float(np.std(d)) if len(d) else 0.0


def solve(series_by_cam: dict, reference: str = None, min_corr: float = None) -> dict:
    """Offsets for every camera against one reference.

    Cameras that fail to correlate are reported in `rejected` and left OUT of
    offsets_ms, so load_offsets() gives them 0.0 and they keep behaving exactly
    as they did before the calibration ran. Partial success is the common case
    with six cameras and one light switch, and it is worth keeping.
    """
    min_corr = MIN_CORRELATION if min_corr is None else min_corr
    usable = {c: s for c, s in series_by_cam.items() if len(s) >= MIN_SAMPLES}
    if not usable:
        raise ValueError("no camera produced enough frames — are they streaming?")

    # Did the lights actually change? A correlation says two cameras saw the same
    # thing, never WHAT — and two cameras watching one person move correlate just
    # fine. Brightness is the direct test; see MIN_BRIGHTNESS_SWING.
    swings = {c: brightness_swing(s) for c, s in usable.items()}
    checking_light = any(_has_brightness(s) for s in usable.values())
    saw_light = {c for c, sw in swings.items() if sw >= MIN_BRIGHTNESS_SWING}
    if checking_light and not saw_light:
        worst = max(swings.values()) if swings else 0.0
        raise ValueError(
            f"the room lights do not appear to have changed — the brightest swing "
            f"any camera saw was {worst:.0f} gray levels, under the "
            f"{MIN_BRIGHTNESS_SWING:.0f} a light switch produces. Turn the lights "
            "fully OFF and back ON 3-4 times during the window (dimming or moving "
            "around the room is not enough).")

    # The reference must be a camera that saw the switch: every other offset is
    # measured against it, so a reference that missed the event poisons all of them.
    pool = {c: s for c, s in usable.items() if not checking_light or c in saw_light}
    if reference is None or reference not in pool:
        reference = max(pool, key=lambda c: len(pool[c]))

    edges = {c: light_edges(s) for c, s in usable.items()}
    ref_edges = edges.get(reference, [])
    offsets = {reference: 0.0}
    quality = {reference: {"correlation": 1.0, "samples": len(usable[reference]),
                           "jitter_ms": round(_jitter_ms(usable[reference]), 1),
                           "brightness_swing": round(swings.get(reference, 0.0), 1),
                           "light_edges": len(ref_edges), "method": "reference"}}
    rejected = {}
    for cam, series in sorted(usable.items()):
        if cam == reference:
            continue
        # Checked BEFORE anything else: "the lights never reached this camera" is a
        # different problem from "this camera disagrees about when they changed",
        # and only the first one tells you to go re-aim a camera.
        if checking_light and cam not in saw_light:
            rejected[cam] = (f"brightness barely moved ({swings[cam]:.0f} of the "
                             f"{MIN_BRIGHTNESS_SWING:.0f} gray levels a light switch "
                             "needs) — this camera did not see the lights change: "
                             "pointed away from the room's lighting, or lit by "
                             "something else")
            continue
        common = {"samples": len(series),
                  "jitter_ms": round(_jitter_ms(series), 1),
                  "brightness_swing": round(swings.get(cam, 0.0), 1),
                  "light_edges": len(edges.get(cam, []))}
        # Edges first: they place the switch directly and impose no limit on how
        # large the delay may be, which correlation cannot do.
        edge_err = None
        if ref_edges and edges.get(cam):
            try:
                off, matched, spread, inverted = offset_from_edges(ref_edges, edges[cam])
                offsets[cam] = round(off, 1)
                quality[cam] = {**common, "method": "light-edges",
                                "edges_matched": matched,
                                "edge_spread_ms": round(spread, 1),
                                # Unique but thin: trustworthy enough to publish
                                # (nothing else explained the edges as well) while
                                # worth re-running with more flips to confirm.
                                "low_confidence": matched < CONFIDENT_EDGES,
                                # True = its brightness moves opposite the
                                # reference's, i.e. auto-exposure is overshooting.
                                "polarity_inverted": inverted}
                continue
            except ValueError as exc:
                edge_err = str(exc)
        # Fallback: correlation, for a run with no brightness recorded or too few
        # clean edges. Still bounded by its search window — see correlate().
        try:
            off, corr, overlap = correlate(usable[reference], series)
        except ValueError as exc:
            rejected[cam] = edge_err or str(exc)
            continue
        if corr < min_corr:
            rejected[cam] = edge_err or (
                f"correlation too weak ({corr:.2f} < {min_corr:.2f}) — it saw the "
                "lights change but not at a consistent time; rerun with more flips")
            continue
        offsets[cam] = round(off, 1)
        quality[cam] = {**common, "method": "energy-correlation",
                        "correlation": round(corr, 3),
                        "overlap_s": round(overlap / 1000.0, 1)}
    for cam, series in series_by_cam.items():
        if cam not in usable:
            rejected[cam] = f"only {len(series)} frame(s) — not streaming?"
    # --- which cameras carry a true capture clock, and what to use for each -----
    # Solve a second time on the forwarders' own timestamps. A camera that lands
    # within HW_CLOCK_TOL_MS of the reference there shares its clock domain, so its
    # own per-frame timestamp is a real capture instant and beats any constant.
    HW = 3
    clocks, hw_offsets = {}, {}
    ref_hw = _view(usable.get(reference, []), HW)
    ref_transport = None
    if ref_hw:
        d = [r[0] - h[0] for r, h in zip(usable[reference], ref_hw)]
        ref_transport = float(np.median(d)) if d else None
        ref_hw_edges = light_edges(ref_hw)
        for cam in list(offsets):
            if cam == reference:
                clocks[cam] = "hw" if ref_hw_edges else "arrival"
                hw_offsets[cam] = 0.0
                continue
            cam_hw = _view(usable.get(cam, []), HW)
            if not (cam_hw and ref_hw_edges):
                clocks[cam] = "arrival"
                continue
            try:
                hw_off, _m, _s, _f = offset_from_edges(ref_hw_edges, light_edges(cam_hw))
            except ValueError:
                clocks[cam] = "arrival"
                continue
            hw_offsets[cam] = round(hw_off, 1)
            clocks[cam] = "hw" if abs(hw_off) <= HW_CLOCK_TOL_MS else "arrival"
    # Both families must end up on the SAME timeline. The hw family lands on the
    # reference's capture clock; the arrival family was measured against the
    # reference's ARRIVAL, which is its capture plus its own transport delay — so
    # that delay has to be added back or the two families sit apart by it.
    if clocks.get(reference) == "hw" and ref_transport:
        for cam, how in clocks.items():
            if how == "arrival" and cam in offsets:
                offsets[cam] = round(offsets[cam] + ref_transport, 1)
        quality.setdefault(reference, {})["reference_transport_ms"] = round(ref_transport, 1)
    for cam, how in clocks.items():
        if cam in quality:
            quality[cam]["capture_clock"] = how
            if how == "hw":
                quality[cam]["hw_offset_ms"] = hw_offsets.get(cam, 0.0)
                offsets[cam] = hw_offsets.get(cam, 0.0)
    # A frame cannot arrive before it was captured. Once the reference sits on a
    # true capture clock, an arrival-clock camera's offset IS its transport delay,
    # so a negative one is not a small error — it is proof the edges were matched to
    # the wrong transitions. Observed: the D435, whose auto-exposure ripple against
    # a weak 48-level swing produced crossings that are not light switches, aligned
    # at -2853 ms with three "matching" edges, and that impossible number was applied.
    if clocks.get(reference) == "hw":
        for cam in [c for c in list(offsets) if clocks.get(c) == "arrival"]:
            if offsets[cam] < -EDGE_TOL_MS:
                rejected[cam] = (
                    f"came out at {offsets[cam]:+.0f} ms, which would mean its frames "
                    "arrive before they were captured — its brightness transitions "
                    "were matched to the wrong light switches. Its swing is probably "
                    "too weak (auto-exposure fighting the change); rerun with more, "
                    "unevenly spaced flips")
                offsets.pop(cam, None)
                quality.pop(cam, None)

    return {
        "schema_version": "2",
        "reference": reference,
        # Subtract from whichever clock `capture_clocks` names for that camera.
        "offsets_ms": offsets,
        "capture_clocks": clocks,
        "quality": quality,
        "rejected": rejected,
        "basis": "reference camera's capture clock",
        "method": "frame-difference energy cross-correlation (lights on/off)",
        "measured_at": dt.datetime.now().astimezone().isoformat(),
    }


def save(result: dict, path: Path = None) -> Path:
    path = path or timing_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(path)     # atomic: live_infer may be reading it
    return path


def summary_line(result: dict) -> str:
    offs = result.get("offsets_ms") or {}
    q = result.get("quality") or {}
    parts = []
    for cam in sorted(offs):
        if cam == result.get("reference"):
            continue
        corr = (q.get(cam) or {}).get("correlation")
        parts.append(f"{cam} {offs[cam]:+.0f}ms"
                     + (f" (r={corr:.2f})" if corr is not None else ""))
    text = ", ".join(parts) or "nothing to compare"
    bad = result.get("rejected") or {}
    if bad:
        text += "; no offset for " + ", ".join(sorted(bad))
    return f"vs {result.get('reference')}: {text}"


# ---------------------------------------------------------------- self-test ---
def _with_brightness(series, swing, n_flips=4, seed=0):
    """Attach a mean-brightness column that swings by `swing` gray levels.

    swing=0 models the case that motivated the gate: a real correlated event
    (someone moving) in a room whose lights never changed.
    """
    rng = np.random.default_rng(1000 + seed)
    t0 = series[0][0]
    span = series[-1][0] - t0
    out = []
    for t, e in series:
        phase = int((t - t0) / max(span / (n_flips * 2), 1.0)) % 2
        level = 40.0 + (swing if phase else 0.0) + rng.normal(0, 0.4)
        out.append((t, e, level))
    return out


def _synthetic(lag_ms, n=600, rate_ms=66.0, noise=0.05, seed=0):
    """Two energy series of the same room, one delayed by `lag_ms`.

    Built from light-switch steps rather than smooth noise: the estimator has to
    work on exactly the signal the procedure produces, a few sharp edges in an
    otherwise flat series.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * rate_ms
    def energy(times):
        e = rng.normal(0, noise, len(times))
        for edge in (4000.0, 9000.0, 14000.0, 21000.0):
            # a switch shows up as one frame of large difference
            e += 1.0 * np.exp(-0.5 * ((times - edge) / 40.0) ** 2)
        return e
    # Both cameras see the SAME physical events at the same physical times; B
    # simply stamps them `lag_ms` later. Independent noise draws per camera.
    a = list(zip(t, energy(t)))
    b = list(zip(t + lag_ms, energy(t)))
    return a, b


def _selftest() -> int:
    fails = []

    def check(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not cond:
            fails.append(name)

    print("correlate(): recovers a known lag")
    for truth in (0.0, 50.0, 120.0, -200.0, 450.0):
        a, b = _synthetic(truth)
        off, corr, _ = correlate(a, b)
        check(f"lag {truth:+.0f}ms", abs(off - truth) <= 12.0 and corr > 0.5,
              f"got {off:+.1f}ms r={corr:.2f}")

    print("correlate(): the score really is a correlation")
    for truth in (0.0, 120.0, -200.0):
        a, b = _synthetic(truth)
        _, corr, _ = correlate(a, b)
        check(f"|r| <= 1 at lag {truth:+.0f}ms", -1.0 <= corr <= 1.0, f"r={corr:.3f}")

    print("correlate(): rejects garbage")
    rng = np.random.default_rng(1)
    t = np.arange(400) * 66.0
    a, _ = _synthetic(0.0)
    noise_only = list(zip(t, rng.normal(0, 1, len(t))))
    _, corr, _ = correlate(a, noise_only)
    check("uncorrelated series scores low", corr < MIN_CORRELATION, f"r={corr:.2f}")
    # A camera pointed at something the lights do not reach sees only its own
    # sensor noise on a flat scene. This is what slipped through the old
    # unbounded score at exactly the threshold; it must be well clear now.
    static = list(zip(t, 40.0 + rng.normal(0, 0.3, len(t))))
    _, corr, _ = correlate(a, static)
    check("static-scene camera scores well below the bar",
          corr < MIN_CORRELATION * 0.7, f"r={corr:.2f}")
    try:
        correlate(a[:5], a[:5])
        check("too-few-frames raises", False)
    except ValueError:
        check("too-few-frames raises", True)
    try:
        flat = [(x, 1.0) for x in t]
        correlate(flat, flat)
        check("no-change raises", False)
    except ValueError:
        check("no-change raises", True)
    # A window too short for the correlation to mean anything must be refused
    # rather than scored — see the MIN_OVERLAP_MS table.
    short_a, short_b = _synthetic(120.0, n=90, rate_ms=66.0)   # ~6s
    try:
        correlate(short_a, short_b)
        check("too-short window raises", False)
    except ValueError as exc:
        check("too-short window raises", "under the" in str(exc), str(exc)[:60])

    print("solve(): reference at 0, laggards positive, bad cameras rejected")
    a, b = _synthetic(300.0, seed=2)
    _, c = _synthetic(-90.0, seed=3)
    res = solve({"ref": a, "late": b, "early": c,
                 "dead": [(0.0, 0.0)], "blind": noise_only}, reference="ref")
    check("reference is 0", res["offsets_ms"]["ref"] == 0.0)
    check("late camera ~+300ms", abs(res["offsets_ms"]["late"] - 300.0) <= 12.0,
          f"got {res['offsets_ms']['late']:+.1f}")
    check("early camera ~-90ms", abs(res["offsets_ms"]["early"] + 90.0) <= 12.0,
          f"got {res['offsets_ms']['early']:+.1f}")
    check("non-streaming camera rejected", "dead" in res["rejected"])
    check("uncorrelated camera rejected", "blind" in res["rejected"])
    check("rejected cameras get no offset",
          "dead" not in res["offsets_ms"] and "blind" not in res["offsets_ms"])
    check("summary mentions the reference", res["reference"] in summary_line(res))

    print("solve(): refuses to call correlated MOTION a lights measurement")
    a2, b2 = _synthetic(300.0, seed=8)
    # Both cameras genuinely correlate (a person moving) but no light ever changed.
    dark = {"ref": _with_brightness(a2, 0.0, seed=1),
            "other": _with_brightness(b2, 0.0, seed=2)}
    try:
        solve(dark, reference="ref")
        check("no-light-change run is refused", False)
    except ValueError as exc:
        check("no-light-change run is refused", "lights do not appear" in str(exc),
              str(exc)[:64])
    # Same pair, lights actually flipped -> the offset comes back.
    lit = {"ref": _with_brightness(a2, 80.0, seed=3),
           "other": _with_brightness(b2, 80.0, seed=4)}
    res3 = solve(lit, reference="ref")
    check("with a real light change it measures", abs(res3["offsets_ms"]["other"] - 300.0) <= 12,
          f"got {res3['offsets_ms'].get('other')}")
    check("brightness swing is reported",
          res3["quality"]["other"]["brightness_swing"] >= MIN_BRIGHTNESS_SWING,
          str(res3["quality"]["other"]["brightness_swing"]))
    # One camera the lights don't reach: rejected for THAT reason, not "too weak".
    mixed = {"ref": _with_brightness(a2, 80.0, seed=5),
             "unlit": _with_brightness(b2, 0.0, seed=6)}
    res4 = solve(mixed, reference="ref")
    check("camera the lights miss is named as such",
          "did not see the lights change" in res4["rejected"].get("unlit", ""),
          res4["rejected"].get("unlit", "")[:56])
    check("a reference that missed the switch is not chosen",
          solve(mixed)["reference"] == "ref")

    print("light_edges(): finds the switches, ignores flicker")
    a5, _ = _synthetic(0.0, seed=11)
    lit5 = _with_brightness(a5, 80.0, n_flips=4, seed=11)
    ed = light_edges(lit5)
    check("found the transitions", len(ed) >= 6, f"{len(ed)} edges")
    check("they alternate on/off",
          all(ed[i][1] != ed[i + 1][1] for i in range(len(ed) - 1)))
    check("a static room yields none", light_edges(_with_brightness(a5, 0.0)) == [])

    print("edges beat correlation on a lag OUTSIDE the search window")
    # 2600ms: past MAX_LAG_MS, so correlation cannot even represent the answer.
    BIG = 2600.0
    a6, b6 = _synthetic(BIG, n=900, rate_ms=50.0, seed=12)
    res6 = solve({"ref": _with_brightness(a6, 80.0, seed=12),
                  "far": _with_brightness(b6, 80.0, seed=12)}, reference="ref")
    got6 = res6["offsets_ms"].get("far")
    check("large lag recovered", got6 is not None and abs(got6 - BIG) <= 60,
          f"got {got6}  (MAX_LAG_MS={MAX_LAG_MS:.0f})")
    check("and it used the edge method",
          res6["quality"]["far"]["method"] == "light-edges",
          str(res6["quality"]["far"].get("method")))
    # Correlation on its own genuinely cannot: proves the fallback was the limit.
    _, corr6, _ = correlate(a6, b6)
    off6, _, _ = correlate(a6, b6)
    check("correlation alone would have been wrong", abs(off6 - BIG) > 500,
          f"correlation said {off6:+.0f}ms r={corr6:.2f}")

    print("edges still work at a normal lag, and refuse when unusable")
    for truth in (0.0, 120.0, 400.0):
        a7, b7 = _synthetic(truth, n=900, rate_ms=50.0, seed=13)
        r7 = solve({"ref": _with_brightness(a7, 80.0, seed=13),
                    "cam": _with_brightness(b7, 80.0, seed=13)}, reference="ref")
        g7 = r7["offsets_ms"].get("cam")
        check(f"edge lag {truth:+.0f}ms", g7 is not None and abs(g7 - truth) <= 60,
              f"got {g7}")
    # Only one flip -> not enough to vote on; must refuse, not guess.
    a8, b8 = _synthetic(300.0, n=900, rate_ms=50.0, seed=14)
    one = solve({"ref": _with_brightness(a8, 80.0, n_flips=1, seed=14),
                 "cam": _with_brightness(b8, 80.0, n_flips=1, seed=14)}, reference="ref")
    check("a single transition is handled without crashing",
          "cam" in one["offsets_ms"] or "cam" in one["rejected"])

    print("solve(): a camera with a real capture clock is put ON it, not given a constant")
    def _cam(lag_ms, hw_is_capture, seed=1):
        base = _with_brightness(_synthetic(0.0, n=900, rate_ms=50.0, seed=seed)[0],
                                80.0, seed=seed)
        return [(t + lag_ms, e, b, (t if hw_is_capture else t + lag_ms))
                for t, e, b in base]
    # ref: a RealSense, small lag, true hw clock. backlog: the D435 — big lag from the
    # Pi's frameset queue, but still a true per-frame clock. nvr: Reolink, whose hw
    # stamp is only when the forwarding host received the frame.
    tri = {"ref": _cam(100.0, True), "backlog": _cam(2700.0, True),
           "nvr": _cam(3460.0, False)}
    r9 = solve(tri, reference="ref")
    cl = r9["capture_clocks"]
    check("reference on its hw clock", cl["ref"] == "hw")
    check("backlogged camera on its hw clock", cl.get("backlog") == "hw", str(cl))
    check("so its constant offset collapses to ~0",
          abs(r9["offsets_ms"]["backlog"]) < 60, str(r9["offsets_ms"].get("backlog")))
    check("receive-time camera stays on arrival", cl.get("nvr") == "arrival", str(cl))
    # capture = arrival - offset, so the offset must be the FULL lag. The raw edge
    # measurement only sees nvr-vs-ref arrival (3360); adding the reference's own
    # transport delay back is what makes it 3460.
    check("its offset is its full lag, not its lag vs the reference's arrival",
          abs(r9["offsets_ms"]["nvr"] - 3460.0) < 80,
          f"{r9['offsets_ms'].get('nvr')} (raw edge measure would be 3360)")
    for name, hw_is_cap, lag in (("ref", True, 100.0), ("backlog", True, 2700.0),
                                 ("nvr", False, 3460.0)):
        off = r9["offsets_ms"][name]
        errs = [abs(((hw if cl[name] == "hw" else arr) - off) - (arr - lag))
                for arr, _e, _b, hw in tri[name][:200]]
        check(f"{name} recovers the true capture instant",
              float(np.median(errs)) < 80, f"median err {np.median(errs):.0f}ms")

    print("solve(): an impossible negative delay is refused, not applied")
    # A camera whose edges get matched to the wrong switches can align at a negative
    # offset, which would mean frames arriving before they were captured.
    bad = _cam(3460.0, False)
    shifted = [(t - 6000.0, e, b, hw) for t, e, b, hw in bad]   # edges land far early
    r10 = solve({"ref": _cam(100.0, True), "wrong": shifted}, reference="ref")
    off10 = r10["offsets_ms"].get("wrong")
    check("either rejected or non-negative, never an impossible offset",
          "wrong" in r10["rejected"] or (off10 is not None and off10 >= -EDGE_TOL_MS),
          f"offset={off10} rejected={'wrong' in r10['rejected']}")
    if "wrong" in r10["rejected"]:
        check("and the reason says why", "before they were captured" in r10["rejected"]["wrong"]
              or "did not line up" in r10["rejected"]["wrong"]
              or "equally well" in r10["rejected"]["wrong"],
              r10["rejected"]["wrong"][:60])

    print("solve(): picks a reference when none is given")
    res2 = solve({"few": a[:40], "many": b})
    check("densest series becomes reference", res2["reference"] == "many")

    print("offset_ms_for(): env beats the file, unknown camera is 0")
    stored = {"camera_x": 123.0}
    check("stored value used", offset_ms_for("camera_x", stored) == 123.0)
    check("unknown camera is 0", offset_ms_for("camera_absent", stored) == 0.0)
    os.environ["SMARTROOM_TIME_OFFSET_CAMERA_X"] = "-45"
    try:
        check("env overrides the file", offset_ms_for("camera_x", stored) == -45.0)
        os.environ["SMARTROOM_TIME_OFFSET_CAMERA_X"] = "not-a-number"
        check("unparseable env falls back to the file",
              offset_ms_for("camera_x", stored) == 123.0)
    finally:
        os.environ.pop("SMARTROOM_TIME_OFFSET_CAMERA_X", None)

    print()
    if fails:
        print(f"{len(fails)} FAILED: {', '.join(fails)}")
        return 1
    print("all checks passed")
    return 0


def _replay(path=None) -> int:
    """Re-solve a saved run and print what each camera looked like.

    Read-only: nothing is written and no offsets are applied. This exists because
    a rejected camera cannot be diagnosed from a one-line reason, and the
    alternative is asking somebody to work the light switch again per hypothesis.
    """
    p = Path(path) if path else timing_path().with_name("live_timing_last_run.json")
    try:
        series = {k: [tuple(r) for r in v]
                  for k, v in json.loads(p.read_text())["series"].items()}
    except (OSError, ValueError, KeyError) as exc:
        print(f"cannot read {p}: {exc}")
        return 1
    print(f"{p}\n")
    print("%-10s %7s %7s %7s  %s" % ("camera", "frames", "swing", "edges", "switch times (s, rel)"))
    t0 = min((s[0][0] for s in series.values() if s), default=0.0)
    for cam, s in sorted(series.items()):
        ed = light_edges(s)
        times = " ".join(f"{(t - t0) / 1000:+.2f}{'^' if d > 0 else 'v'}" for t, d in ed[:9])
        print("%-10s %7d %7.1f %7d  %s" % (cam.replace("camera_", "").replace("_color", ""),
                                           len(s), brightness_swing(s), len(ed), times))
    print()
    try:
        res = solve(series)
    except ValueError as exc:
        print(f"solve refused: {exc}")
        return 0
    print(summary_line(res))
    for cam, q in sorted((res.get("quality") or {}).items()):
        print(f"  {cam:24s} {res['offsets_ms'].get(cam, 0.0):+9.1f} ms  "
              f"via {q.get('method')}  " + " ".join(
                  f"{k}={v}" for k, v in q.items()
                  if k not in ("method", "samples")))
    for cam, why in sorted((res.get("rejected") or {}).items()):
        print(f"  {cam:24s}   (no offset) {why}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="print the stored offsets")
    ap.add_argument("--selftest", action="store_true", help="check the correlation math")
    ap.add_argument("--replay", nargs="?", const="", metavar="FILE",
                    help="re-solve the last run's raw series (live_timing_last_run.json) "
                         "without writing anything — for diagnosing a rejected camera")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.replay is not None:
        return _replay(args.replay or None)
    path = timing_path()
    if not args.show:
        ap.print_help()
        return 0
    if not path.exists():
        print(f"no timing calibration at {path} — every camera is at 0 ms offset")
        return 0
    data = json.loads(path.read_text())
    print(f"{path}\nmeasured {data.get('measured_at')}  basis: {data.get('basis')}")
    print(summary_line(data))
    for cam, q in sorted((data.get("quality") or {}).items()):
        print(f"  {cam:24s} {data['offsets_ms'].get(cam, 0.0):+8.1f} ms  "
              f"r={q.get('correlation')}  jitter={q.get('jitter_ms')}ms  "
              f"n={q.get('samples')}")
    for cam, why in sorted((data.get("rejected") or {}).items()):
        print(f"  {cam:24s}   (no offset) {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
