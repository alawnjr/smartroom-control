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
    # On disk the sidecar IS the persons block ({"persons": {tid: ...}}); the
    # mirror's /inference endpoint nests it one level deeper. Accept either.
    persons = doc.get("persons") or {}
    if "persons" in persons and isinstance(persons["persons"], dict):
        persons = persons["persons"]
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


def _box_of(kpts, conf, shape, pad=0.08):
    """A person box from their confident keypoints — the ReID baseline needs the
    same crop the live path would hand its encoder."""
    pts = [(x, y) for (x, y), c in zip(kpts, conf) if c >= 0.3]
    if len(pts) < 4:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w < 8 or h < 16:
        return None
    x1, y1 = max(0, min(xs) - pad * w), max(0, min(ys) - pad * h)
    x2, y2 = min(shape[1] - 1, max(xs) + pad * w), min(shape[0] - 1, max(ys) + pad * h)
    return [x1, y1, x2, y2]


def sample_clip(mp4: Path, tracks, encoder=None, max_samples_per_track=12):
    """{track id: [(t, descriptor, reid_emb|None)]} — decode each frame once."""
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
            rows = want[idx]
            embs = [None] * len(rows)
            if encoder is not None:
                boxes = [_box_of(k, c, frame.shape) for _, k, c in rows]
                keep = [i for i, b in enumerate(boxes) if b is not None]
                if keep:
                    # ultralytics' ReID wants (cx, cy, w, h) per detection.
                    dets = np.array([[(boxes[i][0] + boxes[i][2]) / 2,
                                      (boxes[i][1] + boxes[i][3]) / 2,
                                      boxes[i][2] - boxes[i][0],
                                      boxes[i][3] - boxes[i][1]] for i in keep],
                                    dtype=np.float32)
                    try:
                        got = encoder(frame, dets)
                        for j, i in enumerate(keep):
                            embs[i] = None if got[j] is None else np.asarray(got[j])
                    except Exception as exc:  # noqa: BLE001
                        print(f"  reid failed: {exc}", file=sys.stderr)
            for (tid, kpts, conf), emb in zip(rows, embs):
                d = A.describe(frame, kpts, conf)
                if d or emb is not None:
                    out[tid].append((idx / fps, d, emb))
        idx += 1
    cap.release()
    return {k: v for k, v in out.items() if v}


def cos(a, b):
    if a is None or b is None:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


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
    ap.add_argument("--reid", default=os.environ.get("SMARTROOM_REID_MODEL", "yolo26n-reid.onnx"),
                    help="ReID onnx to score as a baseline on the same pairs ('' to skip)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the chroma/lightness balance and the shirt/trouser "
                         "weighting instead of reporting one configuration")
    args = ap.parse_args()
    random.seed(7)

    encoder = None
    if args.reid and Path(args.reid).exists():
        try:
            from ultralytics.trackers.utils.reid import ReID
            encoder = ReID(args.reid, device="cpu")
            print(f"reid baseline: {args.reid}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"reid baseline unavailable ({exc})", file=sys.stderr)

    root = saved_root()
    cands = sorted(root.rglob(f"*.detections.{args.model}.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    # Collect the raw PAIRS once, then score them under any configuration —
    # sweeping weights must not mean re-decoding the video six times.
    pairs = {"g": [], "i": []}   # (desc_a, desc_b, reid_cos|None)
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
        sampled = sample_clip(mp4, tracks, encoder)
        if len(sampled) < args.min_tracks:
            continue
        used += 1
        # genuine: same track, different frames
        for tid, rows in sampled.items():
            for a in range(len(rows)):
                for b in range(a + 1, len(rows)):
                    pairs["g"].append((rows[a][1], rows[b][1], cos(rows[a][2], rows[b][2])))
        # impostor: two tracks whose samples are close in time (both on screen)
        ids = list(sampled)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                for ta, da, ea in sampled[ids[a]]:
                    for tb, db, eb in sampled[ids[b]]:
                        if abs(ta - tb) > 0.2:
                            continue
                        pairs["i"].append((da, db, cos(ea, eb)))
        print(f"  {det.parent.relative_to(root)}: {len(sampled)} tracks", file=sys.stderr)

    def score(key, wt, wb):
        """(genuine, impostor) score lists for one configuration."""
        out = {}
        for pop in ("g", "i"):
            vals = []
            for da, db, rc in pairs[pop]:
                if key == "reid":
                    if rc is not None:
                        vals.append(rc)
                    continue
                s, _ = A.similarity(da, db, wt, wb)
                if s is not None:
                    vals.append(s)
            out[pop] = vals
        return out["g"], out["i"]

    def row(label, g, i):
        a = auc(g, i)
        thr = pct(g, 0.10)
        fa = float(np.mean(np.array(i) >= thr)) * 100 if i else float("nan")
        print(f"{label:>16}  {len(g):>7} {pct(g,0.05):>6.3f} {pct(g,0.5):>6.3f}   "
              f"{len(i):>7} {pct(i,0.5):>6.3f} {pct(i,0.95):>6.3f}   "
              f"{(f'{a:.3f}' if a is not None else 'n/a'):>6}   {thr:>5.3f} {fa:>7.1f}%")

    print(f"\nclips used: {used}   (model {args.model})   "
          f"pairs: {len(pairs['g'])} genuine / {len(pairs['i'])} impostor")
    print(f"chroma_w={A.CHROMA_W} ab_bins={A.AB_BINS} ab_window=({A.AB_LO},{A.AB_HI}) "
          f"l_bins={A.L_BINS} shrink={A.SHRINK}")
    print(f"{'config':>16}  {'gen n':>7} {'p05':>6} {'p50':>6}   "
          f"{'imp n':>7} {'p50':>6} {'p95':>6}   {'AUC':>6}   {'thr':>5} {'FA@90%':>7}")
    configs = [("shirt only", 1, 0), ("trousers only", 0, 1), ("shirt+trousers", 1, 1),
               ("shirt-weighted", 2, 1), ("trouser-weighted", 1, 2)]
    if args.sweep:
        for cw in (0.0, 0.25, 0.45, 0.7, 1.0):
            A.CHROMA_W = cw
            for label, wt, wb in configs:
                g, i = score("color", wt, wb)
                row(f"cw{cw:.2f} {label[:9]}", g, i)
    else:
        for label, wt, wb in configs:
            g, i = score("color", wt, wb)
            row(label, g, i)
    if any(rc is not None for _, _, rc in pairs["g"]):
        g, i = score("reid", 0, 0)
        row("reid embedding", g, i)
    print("\nFA@90% = impostor pairs accepted at the threshold that keeps 90% of "
          "genuine pairs. Lower is better; 100% means the signal is useless.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
