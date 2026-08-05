#!/usr/bin/env python3
"""Does clothing colour actually tell these people apart? Measure, don't assume.

Reads existing action sidecars (which carry per-track keypoints in pixels, plus
each window's time), samples the garment colours out of the clip at those exact
frames, and scores two populations:

  genuine  — two samples of the SAME track at different times (same person, by
             construction: a ByteTrack id is continuous)
  impostor — two tracks visible in the SAME frame (different people, by
             construction: one camera cannot see one person twice)

Then reports the separation, which is the only thing that justifies a threshold:
an AUC (the chance a genuine pair outscores an impostor pair) and the percentiles
that matter for setting one. Run per-region as well as combined, because "put
more weight on the shirt" is a claim that should be checked rather than believed.

Usage (on the analysis server, in the detect venv):
  SMARTROOM_SAVE_DIR=/mnt/data4/intern26/recordings \
    detect/.venv-detect/bin/python detect/eval_appearance.py --clips 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import appearance as A  # noqa: E402


def saved_root() -> Path:
    return Path(os.environ.get("SMARTROOM_SAVE_DIR", "recordings"))


def load_tracks(sidecar: Path):
    """{track id: [(t, kpts, conf)]} from an action sidecar's persons block."""
    doc = json.loads(sidecar.read_text())
    persons = (doc.get("persons") or {}).get("persons") or {}
    out = {}
    for tid, p in persons.items():
        rows = []
        for w in p.get("windows") or []:
            kp = w.get("keypoints")
            if not kp or len(kp) < 17:
                continue
            kpts = [(float(k[0]), float(k[1])) for k in kp]
            conf = [float(k[2]) if len(k) > 2 else 1.0 for k in kp]
            rows.append((float(w["t"]), kpts, conf))
        if rows:
            out[tid] = sorted(rows)
    return out


def sample_clip(mp4: Path, tracks, max_samples_per_track=12):
    """{track id: [(t, descriptor)]} — decode each needed frame once."""
    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Frame index -> [(tid, kpts, conf)], so one pass over the video serves every
    # track (seeking per sample is what made the first version unusably slow).
    want = {}
    for tid, rows in tracks.items():
        step = max(1, len(rows) // max_samples_per_track)
        for t, kpts, conf in rows[::step]:
            want.setdefault(int(round(t * fps)), []).append((tid, kpts, conf))
    out = {tid: [] for tid in tracks}
    idx = 0
    todo = set(want)
    while todo:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            todo.discard(idx)
            for tid, kpts, conf in want[idx]:
                d = A.describe(frame, kpts, conf)
                if d:
                    out[tid].append((idx / fps, d))
        idx += 1
    cap.release()
    return {k: v for k, v in out.items() if v}


def auc(genuine, impostor):
    """P(genuine > impostor), by direct comparison over a capped sample."""
    if not genuine or not impostor:
        return None
    g = random.sample(genuine, min(4000, len(genuine)))
    i = random.sample(impostor, min(4000, len(impostor)))
    i = np.array(sorted(i))
    wins = sum(float(np.searchsorted(i, x, side="left") + np.searchsorted(i, x, side="right")) / 2
               for x in g)
    return wins / (len(g) * len(i))


def pct(v, q):
    if not v:
        return float("nan")
    return float(np.percentile(np.array(v), q * 100))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=int, default=6, help="how many clips to sample")
    ap.add_argument("--model", default="action-ava", help="which action sidecar to read")
    ap.add_argument("--min-tracks", type=int, default=2,
                    help="only clips with at least this many tracks (impostors need two)")
    args = ap.parse_args()
    random.seed(7)

    root = saved_root()
    cands = sorted(root.rglob(f"*.detections.{args.model}.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    scores = {"top": {"g": [], "i": []}, "bot": {"g": [], "i": []},
              "both": {"g": [], "i": []}}
    used = 0
    for det in cands:
        if used >= args.clips:
            break
        stem = det.name.split(".detections.")[0]
        side = det.with_name(f"{stem}.persons.{args.model}.json")
        mp4 = det.with_name(f"{stem}.mp4")
        if not side.exists() or not mp4.exists():
            continue
        tracks = load_tracks(side)
        if len(tracks) < args.min_tracks:
            continue
        sampled = sample_clip(mp4, tracks)
        if len(sampled) < args.min_tracks:
            continue
        used += 1
        # genuine: same track, different frames
        for tid, rows in sampled.items():
            for a in range(len(rows)):
                for b in range(a + 1, len(rows)):
                    for key, (wt, wb) in (("top", (1, 0)), ("bot", (0, 1)), ("both", (1, 1))):
                        s, _ = A.similarity(rows[a][1], rows[b][1], wt, wb)
                        if s is not None:
                            scores[key]["g"].append(s)
        # impostor: two tracks whose samples are close in time (both on screen)
        ids = list(sampled)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                for ta, da in sampled[ids[a]]:
                    for tb, db in sampled[ids[b]]:
                        if abs(ta - tb) > 0.2:
                            continue
                        for key, (wt, wb) in (("top", (1, 0)), ("bot", (0, 1)), ("both", (1, 1))):
                            s, _ = A.similarity(da, db, wt, wb)
                            if s is not None:
                                scores[key]["i"].append(s)
        print(f"  {det.parent.relative_to(root)}: {len(sampled)} tracks", file=sys.stderr)

    print(f"\nclips used: {used}   (model {args.model})")
    print(f"{'region':>7}  {'genuine n':>9} {'p05':>6} {'p50':>6}   "
          f"{'impostor n':>10} {'p50':>6} {'p95':>6} {'p99':>6}   {'AUC':>6}")
    for key in ("top", "bot", "both"):
        g, i = scores[key]["g"], scores[key]["i"]
        a = auc(g, i)
        print(f"{key:>7}  {len(g):>9} {pct(g,0.05):>6.3f} {pct(g,0.5):>6.3f}   "
              f"{len(i):>10} {pct(i,0.5):>6.3f} {pct(i,0.95):>6.3f} {pct(i,0.99):>6.3f}   "
              f"{(f'{a:.3f}' if a is not None else 'n/a'):>6}")
    # The number that decides whether this is usable: at a threshold that accepts
    # 90% of genuine pairs, how many impostor pairs does it also accept?
    print()
    for key in ("top", "bot", "both"):
        g, i = scores[key]["g"], scores[key]["i"]
        if not g or not i:
            continue
        thr = pct(g, 0.10)
        fa = float(np.mean(np.array(i) >= thr))
        print(f"  {key:>5}: threshold {thr:.3f} keeps 90% of genuine pairs "
              f"and lets through {fa * 100:.1f}% of impostor pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
