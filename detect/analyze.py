#!/usr/bin/env python3
"""
Run the full analysis pipeline over the saved recordings, with a progress bar.

Terminal counterpart to the dashboard's "Analyze all" — same underlying
detect.py (person detection + pose) and action.py (per-person actions) batches,
launched with their proper venvs, but with live per-clip progress in one bar
instead of a spinner you have to trust.

  python3 detect/analyze.py                 # skip already-analyzed clips (default)
  python3 detect/analyze.py --force         # redo everything
  python3 detect/analyze.py --only action   # just the action pass
  python3 detect/analyze.py --variants ntu,hmdb
  python3 detect/analyze.py --path day_10_2026-07-09/rec_20260709_002/streams/cam2/camera_main.mp4

Stdlib only — run with any python3; the analysis venvs are used for the
children (override with SMARTROOM_DETECT_PYTHON / SMARTROOM_ACTION_PYTHON).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DETECT_DIR = Path(__file__).resolve().parent
HOME = Path.home()

DETECT_PYTHON = os.environ.get("SMARTROOM_DETECT_PYTHON") or str(
    HOME / "Code" / "yolo-bench" / ".venv" / "bin" / "python")
ACTION_PYTHON = os.environ.get("SMARTROOM_ACTION_PYTHON") or str(
    DETECT_DIR.parent / ".venv-action" / "bin" / "python")

# One "N/M clip(s) to process" line per model/variant sets the bar's total;
# each "done:"/"error:" line advances it.
RE_TODO = re.compile(r"^\s*(?:\[(?P<dkey>[\w.-]+)\]|(?P<akey>action\[[\w-]+\]):) (?P<todo>\d+)/\d+ clip\(s\) to process")
RE_DONE = re.compile(r"^\s*(?:\[[\w.-]+\] done:|action done:) (?P<rel>\S+)")
RE_ERROR = re.compile(r"^\s*(?:\[[\w.-]+\]|action) error: (?P<msg>.*)")
RE_WORKING = re.compile(r"^(?:action\[[\w-]+\]: processing|\s*\[[\w.-]+\] done:) (?P<rel>\S+)")
RE_BUSY = re.compile(r"^another .* run is in progress")


class Bar:
    def __init__(self):
        self.total = 0
        self.done = 0
        self.errors = []
        self.current = ""
        self.t0 = time.monotonic()
        self.tty = sys.stderr.isatty()

    def _elapsed(self):
        s = int(time.monotonic() - self.t0)
        return f"{s // 60}m{s % 60:02d}s"

    def render(self):
        if not self.total:
            return
        frac = min(1.0, self.done / self.total)
        if self.tty:
            width = 30
            fill = int(frac * width)
            # ".../rec_x/streams/cam2/camera_main.mp4" -> "rec_x/cam2"
            parts = Path(self.current).parts
            name = "/".join((parts[-4], parts[-2])) if len(parts) >= 4 else self.current
            line = (f"\r[{'█' * fill}{'░' * (width - fill)}] {self.done}/{self.total} "
                    f"({frac * 100:3.0f}%)  {self._elapsed()}  {name:<24}")
            sys.stderr.write(line[:120])
            sys.stderr.flush()
        else:
            print(f"{self.done}/{self.total} ({frac * 100:.0f}%) {self.current}", file=sys.stderr)

    def println(self, msg):
        if self.tty:
            sys.stderr.write("\r\033[K")  # clear the bar line first
        print(msg, file=sys.stderr)
        self.render()


def run_child(label, cmd, bar):
    """Run one batch (detect.py or action.py), feeding its stderr into the bar."""
    try:
        proc = subprocess.Popen(cmd, cwd=str(DETECT_DIR), stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
    except FileNotFoundError:
        bar.println(f"!! {label}: interpreter not found: {cmd[0]}")
        return
    for line in proc.stderr:
        line = line.rstrip("\n")
        if m := RE_TODO.match(line):
            bar.total += int(m["todo"])
            bar.render()
        elif RE_DONE.match(line):
            bar.done += 1
            bar.render()
        elif m := RE_ERROR.match(line):
            bar.done += 1
            bar.errors.append(f"{label}: {m['msg']}")
            bar.println(f"!! {label} error: {m['msg']}")
        elif RE_BUSY.match(line):
            bar.println(f"!! {label}: {line.strip()} — is the dashboard already analyzing?")
        if m := RE_WORKING.match(line):
            bar.current = m["rel"]
            bar.render()
    proc.wait()
    if proc.returncode != 0:
        bar.errors.append(f"{label}: exited with code {proc.returncode}")
        bar.println(f"!! {label} exited with code {proc.returncode}")


def main():
    ap = argparse.ArgumentParser(description="Analyze saved recordings with a progress bar.")
    ap.add_argument("--force", "-f", action="store_true",
                    help="reprocess clips that are already analyzed (default: skip them)")
    ap.add_argument("--only", choices=["detect", "action"],
                    help="run just one pass (default: both)")
    ap.add_argument("--variants", default="ntu",
                    help="action variants, comma-separated: ntu, hmdb (default ntu)")
    ap.add_argument("--path", action="append", metavar="REL",
                    help="limit to specific clip(s), recordings-relative; repeatable")
    args = ap.parse_args()

    passthrough = []
    if args.force:
        passthrough.append("--force")
    for p in args.path or []:
        passthrough += ["--path", p]

    bar = Bar()
    mode = "reprocessing everything" if args.force else "skipping already-analyzed clips"
    print(f"analyze: {mode}", file=sys.stderr)

    if args.only != "action":
        run_child("detect", [DETECT_PYTHON, str(DETECT_DIR / "detect.py"), *passthrough], bar)
    if args.only != "detect":
        run_child("action", [ACTION_PYTHON, str(DETECT_DIR / "action.py"),
                             "--variant", args.variants, *passthrough], bar)

    if bar.tty:
        sys.stderr.write("\r\033[K")
    took = f"{int(time.monotonic() - bar.t0) // 60}m{int(time.monotonic() - bar.t0) % 60:02d}s"
    if bar.total == 0:
        print(f"nothing to do — everything is already analyzed ({took})", file=sys.stderr)
    else:
        print(f"finished {bar.done}/{bar.total} in {took}"
              + (f", {len(bar.errors)} error(s)" if bar.errors else ""), file=sys.stderr)
    return 1 if bar.errors else 0


if __name__ == "__main__":
    sys.exit(main())
