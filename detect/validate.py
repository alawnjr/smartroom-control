#!/usr/bin/env python3
"""
Data validation over the saved recordings, for the smartroom-control dashboard.

For each `recordings/<day>/<rec>/streams/<cam>/camera_main.mp4` it runs a set of
integrity checks (video probeable, metadata schema + cross-field consistency,
per-frame timestamps CSV sanity) and writes one sidecar next to the clip:

  camera_main.validation.json    { status, passed, failed, checks: [{name, ok, detail}] }

Stdlib + ffprobe only — runs with the system python3, no venv needed. Mirrors
detect.py's conventions: idempotent per clip, a global flock so concurrent
triggers are safe, a `status:"analyzing"` marker first, repeatable --path for
a subset, --force to redo current results.

Usage:
  python3 validate.py                # all unvalidated clips
  python3 validate.py --path <rel>   # one clip (recordings-relative)
  python3 validate.py --force        # revalidate everything
"""

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1

# Tolerances. Recordings are requested at a whole duration but ffmpeg stops on a
# frame boundary, clocks on the two Pis differ slightly, and the variable-rate
# cameras make any exact equality meaningless — so every check is a band.
DURATION_TOL_SEC = 2.0     # |video duration - metadata duration_seconds|
FRAME_COUNT_TOL = 2        # |csv rows or metadata frame_count - actual frames|
# Cameras are driven at a nominal 30 fps; a clip whose *delivered* rate strays
# outside the band means the camera under-delivered (low light exposure, USB
# trouble, CPU-bound encode) — fail loudly so bad capture is caught same-day.
TARGET_FPS = float(os.environ.get("SMARTROOM_VALIDATE_TARGET_FPS", "30"))
FPS_TOL_FRAC = float(os.environ.get("SMARTROOM_VALIDATE_FPS_TOL", "0.10"))  # ±10%
MAX_FRAME_GAP_SEC = 1.0    # largest allowed hole between consecutive frames
FIRST_TS_TOL = 0.05        # first CSV timestamp must be ~0
LAST_TS_TOL = 2.0          # last CSV timestamp vs video duration


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


def sidecar_path(mp4: Path) -> Path:
    return mp4.with_name(f"{mp4.stem}.validation.json")


def _atomic_write_json(path: Path, data: dict):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def needs_validation(mp4: Path, force: bool) -> bool:
    if force:
        return True
    sc = sidecar_path(mp4)
    if not sc.exists():
        return True
    try:
        data = json.loads(sc.read_text())
    except Exception:
        return True
    if data.get("status") != "done":
        return True
    # 2s tolerance, same as detect.py / lib/detections.ts.
    return data.get("sourceMtimeMs", 0) + 2000 < mp4.stat().st_mtime * 1000


def ffprobe_video(mp4: Path):
    """(duration_sec, frame_count, avg_fps) from the container, or (None,)*3."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets",
             "-show_entries", "stream=nb_read_packets,avg_frame_rate:format=duration",
             "-of", "json", str(mp4)],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
        data = json.loads(out)
        stream = (data.get("streams") or [{}])[0]
        duration = float(data.get("format", {}).get("duration", 0)) or None
        frames = int(stream.get("nb_read_packets", 0)) or None
        num, _, den = (stream.get("avg_frame_rate") or "0/1").partition("/")
        fps = (float(num) / float(den)) if float(den or 1) else None
        return duration, frames, fps
    except Exception:
        return None, None, None


def validate_clip(mp4: Path) -> list:
    """All checks for one clip -> [{name, ok, detail}]. Never raises."""
    checks = []

    def check(name: str, ok: bool, detail: str):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # --- video ---
    duration, frames, _ = ffprobe_video(mp4)
    check("video_probeable", duration is not None and (frames or 0) > 0,
          f"duration={duration}s frames={frames}" if duration else "ffprobe could not read the file")
    real_fps = (frames / duration) if duration and frames else None
    if real_fps is not None:
        lo, hi = TARGET_FPS * (1 - FPS_TOL_FRAC), TARGET_FPS * (1 + FPS_TOL_FRAC)
        check("fps_matches_target", lo <= real_fps <= hi,
              f"actual {real_fps:.2f} fps vs target {TARGET_FPS:g} "
              f"(allowed {lo:.1f}-{hi:.1f}; {frames} frames / {duration:.2f}s)")

    # --- per-cam metadata.json (sibling of the mp4) ---
    meta_path = mp4.parent / "metadata.json"
    meta = None
    if not meta_path.exists():
        check("metadata_exists", False, "metadata.json missing next to the clip")
    else:
        try:
            meta = json.loads(meta_path.read_text())
            check("metadata_exists", True, meta_path.name)
        except Exception as e:
            check("metadata_exists", False, f"unparseable: {e}")

    if meta is not None:
        cam = (meta.get("streams") or {}).get("camera_main") or {}
        missing = [k for k in ("recording_id", "node", "start_time", "end_time", "duration_seconds") if k not in meta]
        missing += [f"streams.camera_main.{k}" for k in ("path", "fps", "frame_count", "timestamps_path") if k not in cam]
        check("metadata_schema", not missing, "all required keys present" if not missing else f"missing: {', '.join(missing)}")

        try:
            t0 = dt.datetime.fromisoformat(meta["start_time"])
            t1 = dt.datetime.fromisoformat(meta["end_time"])
            wall = (t1 - t0).total_seconds()
            want = float(meta["duration_seconds"])
            check("metadata_times_consistent", abs(wall - want) <= DURATION_TOL_SEC,
                  f"end-start={wall:.2f}s vs duration_seconds={want}")
        except Exception as e:
            check("metadata_times_consistent", False, f"unreadable times: {e}")

        if duration is not None:
            try:
                want = float(meta["duration_seconds"])
                check("video_duration_matches_metadata", abs(duration - want) <= DURATION_TOL_SEC,
                      f"video {duration:.2f}s vs metadata {want}s")
            except Exception:
                pass
        if frames is not None and isinstance(cam.get("frame_count"), (int, float)):
            claimed = int(cam["frame_count"])
            check("frame_count_matches_metadata", abs(claimed - frames) <= FRAME_COUNT_TOL,
                  f"metadata claims {claimed}, video has {frames}"
                  + (" (synthetic pre-fix count)" if claimed != frames else ""))

    # --- timestamps CSV (sibling of the mp4) ---
    ts_path = mp4.parent / "camera_main_timestamps.csv"
    if not ts_path.exists():
        check("timestamps_exists", False, "camera_main_timestamps.csv missing")
    else:
        check("timestamps_exists", True, ts_path.name)
        try:
            with ts_path.open() as f:
                rows = list(csv.reader(f))
            header, body = rows[0], rows[1:]
            check("timestamps_header", header[:2] == ["frame_index", "timestamp_seconds"], ",".join(header))
            times = [float(r[1]) for r in body if len(r) >= 2]
            if times:
                deltas = [b - a for a, b in zip(times, times[1:])]
                check("timestamps_monotonic", all(d > 0 for d in deltas) if deltas else True,
                      "strictly increasing" if all(d > 0 for d in deltas) else "non-increasing timestamps found")
                check("timestamps_start_at_zero", abs(times[0]) <= FIRST_TS_TOL, f"first={times[0]:.4f}s")
                if duration is not None:
                    check("timestamps_cover_video", abs(times[-1] - duration) <= LAST_TS_TOL,
                          f"last={times[-1]:.2f}s vs video {duration:.2f}s")
                if deltas:
                    worst = max(deltas)
                    check("no_large_frame_gaps", worst <= MAX_FRAME_GAP_SEC, f"largest gap {worst:.3f}s")
                if frames is not None:
                    check("timestamps_row_count", abs(len(times) - frames) <= FRAME_COUNT_TOL,
                          f"{len(times)} rows vs {frames} video frames"
                          + (" (synthetic pre-fix grid)" if abs(len(times) - frames) > FRAME_COUNT_TOL else ""))
            else:
                check("timestamps_monotonic", False, "no data rows")
        except Exception as e:
            check("timestamps_header", False, f"unreadable: {e}")

    # --- combined per-recording metadata (streams/metadata.json, written by Save All) ---
    combined = mp4.parent.parent / "metadata.json"
    check("combined_metadata_exists", combined.exists(),
          "streams/metadata.json present" if combined.exists() else "run Save All to generate the combined metadata")

    return checks


def process_clip(mp4: Path, root: Path):
    sc = sidecar_path(mp4)
    source_mtime_ms = mp4.stat().st_mtime * 1000
    _atomic_write_json(sc, {"schemaVersion": SCHEMA_VERSION, "status": "analyzing",
                            "source": mp4.name, "sourceMtimeMs": source_mtime_ms})
    try:
        checks = validate_clip(mp4)
        failed = [c["name"] for c in checks if not c["ok"]]
        _atomic_write_json(sc, {
            "schemaVersion": SCHEMA_VERSION,
            "status": "done",
            "error": None,
            "source": mp4.name,
            "sourceMtimeMs": source_mtime_ms,
            "validatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "failedChecks": failed,
            "checks": checks,
        })
        status = "OK" if not failed else f"{len(failed)} FAILED ({', '.join(failed)})"
        print(f"  [validate] {mp4.relative_to(root)}: {status}", file=sys.stderr)
        return not failed
    except Exception as e:  # never leave an 'analyzing' marker behind
        _atomic_write_json(sc, {"schemaVersion": SCHEMA_VERSION, "status": "error",
                                "source": mp4.name, "sourceMtimeMs": source_mtime_ms,
                                "error": str(e)})
        print(f"  [validate] {mp4.relative_to(root)}: ERROR {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description="Data validation over saved recordings.")
    ap.add_argument("--path", action="append", metavar="REL",
                    help="clip to validate, relative to the recordings root; repeatable for a subset")
    ap.add_argument("--force", action="store_true", help="revalidate even if results are current")
    args = ap.parse_args()

    root = saved_root()
    if not root.exists():
        print(f"no recordings dir: {root}", file=sys.stderr)
        return 0

    lock_file = open(root / ".validate.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another validation run is in progress; exiting", file=sys.stderr)
        return 0

    if args.path:
        clips = [root / p for p in args.path]
    else:
        clips = sorted((p for p in root.rglob("camera_main.mp4")),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    # undistorted/ holds lens-corrected COPIES of clips, not additional clips.
    clips = [c for c in clips if c.exists() and "undistorted" not in c.parts
             and needs_validation(c, args.force or bool(args.path))]

    ok = True
    for clip in clips:
        ok = process_clip(clip, root) and ok
    print(f"validated {len(clips)} clip(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
