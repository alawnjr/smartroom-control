#!/usr/bin/env python3
"""Sound event classification over the recordings' audio, with PaSST (AudioSet 527).

The room has ONE microphone — Reolink channel 1 — and its sound is muxed into that
camera's clip (see live_infer.py's segment recorder). So this is a ROOM-level
signal: it can say "someone is typing", never who or where. Anything per-person
has to come from pairing these events with the video.

Why PaSST: it is the strongest of the practical AudioSet taggers (published mAP
~0.47 against ~0.43 for PANNs CNN14 and ~0.31 for YAMNet), and it is trained at
32 kHz — exactly the rate the forwarder already produces, so the waveform reaches
the model without a resample.

  camera_cam1_color.audio.audio-passt.json        per-window events (the timeline)
  camera_cam1_color.detections.audio-passt.json   status + clip-level summary

The second file is what makes the model appear in the dashboard's `models` map and
in the mirror's inference endpoint; it mirrors what action.py writes.

TEMPORAL RESOLUTION, stated plainly: PaSST is trained on 10-second AudioSet clips,
so a window is 10 s wide and an event's timestamp is its window's CENTRE. A door
closing at t=5 is reported by every window that contains it. `tStart`/`tEnd` are
written per row so a consumer can see the smear rather than infer precision that
is not there. Shorter windows would sharpen the timing and move the model off the
length it was trained on; that trade is available via SMARTROOM_AUDIO_WINDOW_S.

Multi-label, like AVA and unlike the skeleton action models: AudioSet events
co-occur (speech AND typing AND air conditioning), so every class over the
threshold is reported and there is no argmax anywhere in here.

Usage (on the analysis server):
  SMARTROOM_SAVE_DIR=/mnt/data4/intern26/recordings \\
    .venv-audio/bin/python detect/audio_events.py [--path REL] [--force]
"""

from __future__ import annotations

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_KEY = os.environ.get("SMARTROOM_AUDIO_MODEL_KEY", "audio-passt")
SCHEMA_VERSION = 1
SAMPLE_RATE = 32000                       # PaSST's native rate; also the mp3's
WINDOW_S = float(os.environ.get("SMARTROOM_AUDIO_WINDOW_S", "10.0"))
HOP_S = float(os.environ.get("SMARTROOM_AUDIO_HOP_S", "2.0"))
# Report every class over this probability. AudioSet models are calibrated loosely
# and the tail is long; 0.2 keeps quiet-but-real events (a door, a cough) while
# leaving out the 0.05 noise that would make every window look busy.
THRESH = float(os.environ.get("SMARTROOM_AUDIO_THRESH", "0.2"))
TOPK = int(os.environ.get("SMARTROOM_AUDIO_TOPK", "8"))
LABELS_CSV = Path(os.environ.get("SMARTROOM_AUDIOSET_LABELS")
                  or (Path(__file__).resolve().parent / "audioset_labels.csv"))
# Classes this room cannot produce. AudioSet spans 527 classes of internet video;
# an indoor lab will never contain a helicopter, and a confident wrong label is
# worse than a missing one — the same lesson the HMDB action variant taught. The
# file is a JSON list of display names; missing file = nothing masked.
DISABLED_JSON = Path(os.environ.get("SMARTROOM_AUDIO_DISABLED")
                     or (PROJECT_ROOT / "audio-classes.json"))


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR") or (PROJECT_ROOT / "recordings"))


def load_labels() -> list[str]:
    """AudioSet display names, indexed by class id (the order PaSST's head uses)."""
    with LABELS_CSV.open() as f:
        rows = list(csv.DictReader(f))
    out = [""] * len(rows)
    for r in rows:
        out[int(r["index"])] = r["display_name"]
    return out


def load_disabled(labels) -> set[int]:
    try:
        names = set(json.loads(DISABLED_JSON.read_text()).get("disabled", []))
    except (OSError, ValueError):
        return set()
    return {i for i, n in enumerate(labels) if n in names}


def has_audio(mp4: Path) -> bool:
    """Does this clip carry an audio stream at all? Only the microphone's camera
    does, but discovering it beats hardcoding which camera that is — a hardcoded
    list of streams is exactly what kept the NVR cameras out of detection."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(mp4)],
            capture_output=True, text=True, timeout=60)
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def read_audio(mp4: Path):
    """The clip's audio as mono float32 at 32 kHz, in the clip's own timebase."""
    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:200]}")
    return np.frombuffer(proc.stdout, dtype="float32")


def sidecars(mp4: Path):
    return (mp4.with_name(f"{mp4.stem}.detections.{MODEL_KEY}.json"),
            mp4.with_name(f"{mp4.stem}.audio.{MODEL_KEY}.json"))


def _atomic_write_json(path: Path, data: dict):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def needs_work(mp4: Path, force: bool) -> bool:
    if force:
        return True
    det, tl = sidecars(mp4)
    if not det.exists() or not tl.exists():
        return True
    try:
        d = json.loads(det.read_text())
    except (OSError, ValueError):
        return True
    if d.get("status") != "done":
        return True
    # Re-run when the clip itself changed under us (a re-recorded take).
    return abs((d.get("sourceMtimeMs") or 0) - mp4.stat().st_mtime * 1000) > 1000


class Tagger:
    """PaSST, loaded once per process."""

    def __init__(self):
        import torch
        from hear21passt.base import get_basic_model

        self.torch = torch
        self.device = ("cuda:0" if (torch.cuda.is_available()
                                    and os.environ.get("SMARTROOM_AUDIO_DEVICE") != "cpu")
                       else "cpu")
        self.model = get_basic_model(mode="logits").to(self.device).eval()
        self.labels = load_labels()
        self.disabled = load_disabled(self.labels)
        print(f"[audio] PaSST on {self.device}: {len(self.labels)} classes, "
              f"{len(self.disabled)} masked, window {WINDOW_S}s hop {HOP_S}s "
              f"thresh {THRESH}", file=sys.stderr)

    def windows(self, wave):
        """(centre_time, start, end, samples) per analysis window.

        A clip shorter than one window is zero-padded and reported as one window
        rather than skipped: a 6-second take is exactly when someone said something.
        """
        import numpy as np

        n = int(WINDOW_S * SAMPLE_RATE)
        hop = max(1, int(HOP_S * SAMPLE_RATE))
        if len(wave) <= n:
            pad = np.zeros(n, dtype="float32")
            pad[:len(wave)] = wave
            yield (len(wave) / SAMPLE_RATE / 2, 0.0, len(wave) / SAMPLE_RATE, pad)
            return
        for start in range(0, len(wave) - n + 1, hop):
            seg = wave[start:start + n]
            t0 = start / SAMPLE_RATE
            yield (t0 + WINDOW_S / 2, t0, t0 + WINDOW_S, seg)

    def tag(self, wave):
        """-> [{t, tStart, tEnd, events: [[label, prob], ...]}], batched."""
        import numpy as np
        torch = self.torch

        rows = list(self.windows(wave))
        if not rows:
            return []
        out = []
        batch = int(os.environ.get("SMARTROOM_AUDIO_BATCH", "8"))
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            x = torch.from_numpy(np.stack([c[3] for c in chunk])).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
            for (t, t0, t1, _), p in zip(chunk, probs):
                if self.disabled:
                    p = p.copy()
                    for idx in self.disabled:
                        p[idx] = 0.0
                order = np.argsort(-p)[:TOPK]
                events = [[self.labels[j], round(float(p[j]), 3)]
                          for j in order if p[j] >= THRESH]
                out.append({"t": round(t, 3), "tStart": round(t0, 3), "tEnd": round(t1, 3),
                            "events": events})
        return out


def process_clip(tagger: Tagger, mp4: Path):
    det_path, tl_path = sidecars(mp4)
    source_mtime_ms = mp4.stat().st_mtime * 1000
    _atomic_write_json(det_path, {"schemaVersion": SCHEMA_VERSION, "status": "analyzing",
                                  "model": MODEL_KEY, "source": mp4.name,
                                  "sourceMtimeMs": source_mtime_ms})
    wave = read_audio(mp4)
    timeline = tagger.tag(wave)

    # Clip-level summary: how many windows each event appeared in, and how strongly.
    # `windows` is what makes "heard once, faintly" distinguishable from "constant".
    summary: dict[str, dict] = {}
    for row in timeline:
        for label, prob in row["events"]:
            e = summary.setdefault(label, {"windows": 0, "peakProb": 0.0})
            e["windows"] += 1
            e["peakProb"] = max(e["peakProb"], prob)
    ranked = dict(sorted(summary.items(), key=lambda kv: (-kv[1]["windows"], -kv[1]["peakProb"])))

    _atomic_write_json(tl_path, {
        "schemaVersion": SCHEMA_VERSION, "model": MODEL_KEY, "source": mp4.name,
        "sourceMtimeMs": source_mtime_ms, "sampleRate": SAMPLE_RATE,
        "windowSec": WINDOW_S, "hopSec": HOP_S, "threshold": THRESH,
        # Every event's `t` is its window's CENTRE; the window is `windowSec` wide.
        "timeline": timeline,
    })
    _atomic_write_json(det_path, {
        "schemaVersion": SCHEMA_VERSION, "status": "done", "error": None,
        "model": MODEL_KEY, "source": mp4.name, "sourceMtimeMs": source_mtime_ms,
        "device": tagger.device, "classifier": "passt_audioset",
        "analyzedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "durationSec": round(len(wave) / SAMPLE_RATE, 3),
        "sampleRate": SAMPLE_RATE, "windowSec": WINDOW_S, "hopSec": HOP_S,
        "threshold": THRESH, "windowsAnalyzed": len(timeline),
        "scope": "room",   # one microphone: not attributable to a person or a place
        "events": list(ranked),
        "eventStats": ranked,
    })
    print(f"  audio done: {mp4.relative_to(saved_root())} "
          f"{len(timeline)} windows {list(ranked)[:6]}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", action="append", metavar="REL",
                    help="clip to analyze, relative to the recordings root; repeatable")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = saved_root()
    if not root.exists():
        print(f"no recordings dir: {root}", file=sys.stderr)
        return 0

    sfx = os.environ.get("SMARTROOM_LOCK_SUFFIX", "")
    lock = open(root / f".audio.lock{sfx}", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another audio run is in progress; exiting", file=sys.stderr)
        return 0

    if args.path:
        clips = [root / p for p in args.path]
    else:
        clips = sorted((p for p in root.rglob("camera_*.mp4")
                        if "undistorted" not in p.parts and ".annotated." not in p.name),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    clips = [c for c in clips if c.exists()]
    audio_clips = [c for c in clips if has_audio(c)]
    todo = [c for c in audio_clips if needs_work(c, args.force)]
    print(f"audio[{MODEL_KEY}]: {len(todo)}/{len(audio_clips)} clip(s) with sound to "
          f"process ({len(clips)} clips scanned)", file=sys.stderr)
    if not todo:
        return 0

    tagger = Tagger()
    processed = errors = 0
    for mp4 in todo:
        try:
            process_clip(tagger, mp4)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  audio error on {mp4.name}: {exc}", file=sys.stderr)
            det, _ = sidecars(mp4)
            try:
                _atomic_write_json(det, {"schemaVersion": SCHEMA_VERSION, "status": "error",
                                         "model": MODEL_KEY, "error": str(exc),
                                         "source": mp4.name,
                                         "sourceMtimeMs": mp4.stat().st_mtime * 1000})
            except OSError:
                pass
    print(f"audio[{MODEL_KEY}]: {processed} done, {errors} failed", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
