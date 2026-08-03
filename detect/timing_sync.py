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
MIN_OVERLAP_MS = 3000.0
MIN_SAMPLES = 30


def timing_path() -> Path:
    return Path(os.environ.get("SMARTROOM_LIVE_TIMING")
                or (PROJECT_ROOT / "calibration" / "live_timing.json"))


def _env_key(cam_key: str) -> str:
    return "SMARTROOM_TIME_OFFSET_" + cam_key.upper()


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
        raise ValueError(f"cameras overlapped for only {overlap / 1000:.1f}s — rerun")
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
    if reference is None or reference not in usable:
        # The camera with the most samples: the steadiest, densest series makes
        # the best yardstick, and every other offset is measured against it.
        reference = max(usable, key=lambda c: len(usable[c]))

    offsets = {reference: 0.0}
    quality = {reference: {"correlation": 1.0, "samples": len(usable[reference]),
                           "jitter_ms": round(_jitter_ms(usable[reference]), 1)}}
    rejected = {}
    for cam, series in sorted(usable.items()):
        if cam == reference:
            continue
        try:
            off, corr, overlap = correlate(usable[reference], series)
        except ValueError as exc:
            rejected[cam] = str(exc)
            continue
        if corr < min_corr:
            rejected[cam] = (f"correlation too weak ({corr:.2f} < {min_corr:.2f}) — "
                             "this camera did not see the same change")
            continue
        offsets[cam] = round(off, 1)
        quality[cam] = {"correlation": round(corr, 3), "samples": len(series),
                        "overlap_s": round(overlap / 1000.0, 1),
                        "jitter_ms": round(_jitter_ms(series), 1)}
    for cam, series in series_by_cam.items():
        if cam not in usable:
            rejected[cam] = f"only {len(series)} frame(s) — not streaming?"
    return {
        "schema_version": "1",
        "reference": reference,
        "offsets_ms": offsets,     # subtract from that camera's ARRIVAL times
        "quality": quality,
        "rejected": rejected,
        "basis": "server frame-arrival time",
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="print the stored offsets")
    ap.add_argument("--selftest", action="store_true", help="check the correlation math")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
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
