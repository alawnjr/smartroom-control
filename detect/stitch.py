"""Stitch fragmented tracks back into people.

The offline pass tracks with ByteTrack, which associates by motion and box
overlap and has NO appearance model at all. So when one person walks in front of
another, the covered person's track dies; when they reappear a few seconds later
they are a NEW id, with a new colour and a new action timeline. A 40-second clip
of three people routinely came out as 12-20 "people". Nobody moved; the ids did.

This module answers one question over a finished clip: which of these track ids
are the same human? It is deliberately separate from the tracker — the tracker
runs forward frame by frame and must decide now, while this sees the whole clip
and can weigh a fragment against everything before and after it.

Rules, in the order they do work:

  1. TIME DISJOINTNESS is a hard constraint, not a score. One camera cannot see
     one person in two places at once, so two tracks that overlap in time are
     different people no matter how alike they look — this is what keeps two
     people in identical black shirts apart, and no appearance threshold can
     substitute for it.
  2. Appearance decides the rest, using the ReID embedding. Measured on 24 clips
     of this room (see eval_appearance.py): template-vs-template AUC 0.909 for the
     embedding against 0.83 for garment colour, and fusing colour IN made it
     monotonically worse (0.898 -> 0.877 -> 0.830 as colour's weight rose from 0
     to 1). Colour is therefore available as a VETO only, and off by default.
  3. MUTUAL BEST plus a MARGIN. A threshold alone is not enough: different-person
     pairs reach cosine 0.88 at the 95th percentile, so "above 0.65" admits real
     mistakes. Requiring each side to be the other's best candidate, by a clear
     margin over the runner-up, is what makes a merge mean "and nothing else came
     close".
  4. Merging is iterative. After two fragments join, the merged template is the
     weighted mean of both, and that better template can attract a third fragment
     no single pair would have justified.

A merge that does not happen costs a colour change in the UI. A merge that should
not have happened fuses two people permanently, and no downstream consumer can
detect it. The defaults are set accordingly.
"""

from __future__ import annotations

import os

import numpy as np

# Cosine floor for a merge. Genuine template pairs sit at p05 0.607 / p50 0.873;
# different people at p50 0.383 / p95 0.881. 0.72 sits above the impostor median
# by a wide margin and still keeps most genuine pairs, with the margin rule below
# doing the work on the overlap.
#
# 0.80, not the 0.72 the template-vs-template distributions suggested. Those
# distributions flattered the task: they compared two halves of ONE track, where
# scale, pose and lighting barely change. Measured on the harder thing this module
# actually does — matching ACROSS a re-detection — the long fragments of a clip
# that are plausibly one person scored only 0.55-0.60, i.e. inside the
# different-person range. So the embedding cannot be trusted to stitch in general
# here, and the threshold is set where it only fires on the clear cases (two pairs
# at 0.84-0.85 in the clip that motivated this) rather than where it would recover
# the most fragments.
STITCH_THRESH = float(os.environ.get("SMARTROOM_STITCH_THRESH", "0.80"))
# How far the winner must beat the runner-up. This is the guard against the
# impostor tail: a fragment that looks similar to TWO live identities is exactly
# the case where a merge should be declined.
STITCH_MARGIN = float(os.environ.get("SMARTROOM_STITCH_MARGIN", "0.06"))
# Longest silence a person may be absent across and still be recognised as
# themselves. Generous: an occlusion behind a colleague lasts seconds, and a clip
# is only ~30-60 s long, so the real bound on false merges is appearance plus
# disjointness rather than the clock.
STITCH_MAX_GAP_S = float(os.environ.get("SMARTROOM_STITCH_MAX_GAP_S", "20.0"))
# Tracks may overlap by this much and still be considered the same person: an id
# switch usually hands over with a frame or two of both ids alive.
STITCH_OVERLAP_TOL_S = float(os.environ.get("SMARTROOM_STITCH_OVERLAP_TOL_S", "0.35"))
# Colour veto: reject a merge whose garment colours disagree worse than this.
# 0 disables it (the default) — see rule 2 above; colour in this room is too weak
# to overrule the embedding, and a bad veto blocks genuine merges.
STITCH_COLOUR_VETO = float(os.environ.get("SMARTROOM_STITCH_COLOUR_VETO", "0"))
# A fragment with fewer appearance samples than this has no reliable template; it
# is left as its own person rather than merged on noise.
STITCH_MIN_SAMPLES = int(os.environ.get("SMARTROOM_STITCH_MIN_SAMPLES", "2"))


def _cos(a, b):
    if a is None or b is None:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def _spans_conflict(a_spans, b_spans, tol):
    """Do two clusters occupy the same time? Overlap beyond `tol` on any pair of
    spans means these are two people, whatever they look like."""
    for a0, a1 in a_spans:
        for b0, b1 in b_spans:
            if min(a1, b1) - max(a0, b0) > tol:
                return True
    return False


def _gap(a_spans, b_spans):
    """Shortest silence between two clusters (0 if they touch or overlap)."""
    best = float("inf")
    for a0, a1 in a_spans:
        for b0, b1 in b_spans:
            best = min(best, max(0.0, b0 - a1) if b0 >= a1 else max(0.0, a0 - b1))
    return best


def stitch(tracks, thresh=None, margin=None, max_gap=None, overlap_tol=None,
           colour_veto=None, colour_sim=None):
    """Group track ids into people.

    `tracks` maps track id -> {"emb": ndarray|None, "colour": dict|None,
    "t0": float, "t1": float, "samples": int}.

    `colour_sim(a, b) -> float|None` is injected (normally
    appearance.similarity) so this module stays free of cv2 and can be tested on
    plain numbers.

    Returns (person_of, detail) where person_of maps every track id to a person
    id — the LOWEST-numbered track in its cluster, so a person's id is the first
    time the clip saw them — and detail lists the merges with their scores.
    """
    thresh = STITCH_THRESH if thresh is None else thresh
    margin = STITCH_MARGIN if margin is None else margin
    max_gap = STITCH_MAX_GAP_S if max_gap is None else max_gap
    overlap_tol = STITCH_OVERLAP_TOL_S if overlap_tol is None else overlap_tol
    colour_veto = STITCH_COLOUR_VETO if colour_veto is None else colour_veto

    # One cluster per track to start. `key` is the cluster's identity; spans are
    # kept as a LIST because a stitched person is several disjoint appearances and
    # a merged min/max envelope would swallow whoever was on screen between them.
    clusters = {}
    for tid, t in tracks.items():
        clusters[tid] = {
            "ids": [tid],
            "emb": None if t.get("emb") is None else np.asarray(t["emb"], dtype="float64"),
            # A COPY: merging blends colour templates in place, and blending the
            # caller's own dict corrupts the data it passed in (found by a test
            # that ran stitch twice on the same tracks and got different answers).
            "colour": dict(t.get("colour") or {}),
            "spans": [(float(t["t0"]), float(t["t1"]))],
            "w": max(1, int(t.get("samples", 1))),
        }

    declined = set()   # pairs judged ambiguous; revisited only if a merge changes them

    def candidates():
        """Every legal merge with its score, best first.

        Declined pairs stay IN this list. They are skipped as merge candidates but
        they must still count as competition: without that, declining the better of
        two near-identical options simply promoted the worse one to unopposed and
        the ambiguous merge happened anyway (a test caught exactly that).
        """
        out = []
        keys = list(clusters)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = clusters[keys[i]], clusters[keys[j]]
                if a["emb"] is None or b["emb"] is None:
                    continue
                if _spans_conflict(a["spans"], b["spans"], overlap_tol):
                    continue
                if _gap(a["spans"], b["spans"]) > max_gap:
                    continue
                s = _cos(a["emb"], b["emb"])
                if s is None or s < thresh:
                    continue
                if colour_veto > 0 and colour_sim is not None:
                    cs, _ = colour_sim(a["colour"], b["colour"])
                    if cs is not None and cs < colour_veto:
                        continue
                out.append((s, keys[i], keys[j]))
        out.sort(reverse=True)
        return out

    detail = []
    while True:
        cands = candidates()
        live = [c for c in cands if (c[1], c[2]) not in declined]
        if not live:
            break
        best_s, ka, kb = live[0]
        # Mutual best with a margin: the runner-up involving EITHER endpoint must
        # be clearly worse, or this fragment looks similar to two identities and
        # merging it would be a guess. Decline THIS pair and move on — declining
        # is recorded, and a later merge that changes either cluster clears it.
        # ...but only a runner-up that COMPETES counts. Three fragments of one
        # person are all near-identical to each other, so "nothing else came close"
        # is false for every genuine chain — an unconditional margin merged nothing
        # at all. The competing case is narrower: this fragment also matches some
        # other cluster that CANNOT be the same person as the winner, because the
        # two of them were on screen together. Then attaching it is a coin flip.
        runner = None
        for s, x, y in cands:
            if (x, y) == (ka, kb):
                continue
            shared = ka if ka in (x, y) else (kb if kb in (x, y) else None)
            if shared is None:
                continue
            other = y if x == shared else x
            rival = kb if shared == ka else ka
            if other == rival or other not in clusters or rival not in clusters:
                continue
            if _spans_conflict(clusters[other]["spans"], clusters[rival]["spans"], overlap_tol):
                runner = s
                break
        if runner is not None and best_s - runner < margin:
            declined.add((ka, kb))
            detail.append({"declined": [ka, kb], "score": round(best_s, 3),
                           "runnerUp": round(runner, 3), "why": "ambiguous"})
            continue
        a, b = clusters[ka], clusters[kb]
        keep, drop = (ka, kb) if min(a["ids"]) < min(b["ids"]) else (kb, ka)
        k, d = clusters[keep], clusters[drop]
        wk, wd = k["w"], d["w"]
        k["emb"] = (k["emb"] * wk + d["emb"] * wd) / (wk + wd)
        k["ids"] = sorted(k["ids"] + d["ids"])
        k["spans"] = sorted(k["spans"] + d["spans"])
        k["w"] = wk + wd
        for name, vec in (d["colour"] or {}).items():
            prev = k["colour"].get(name)
            k["colour"][name] = vec if prev is None else (prev * wk + vec * wd) / (wk + wd)
        clusters.pop(drop)
        detail.append({"merged": sorted([keep, drop]), "score": round(best_s, 3),
                       "runnerUp": None if runner is None else round(runner, 3)})
        # The merged template is a new thing: pairs declined as ambiguous against
        # either endpoint deserve another look now that one candidate is gone.
        for pair in [p for p in declined if keep in p or drop in p]:
            declined.discard(pair)

    person_of = {}
    for c in clusters.values():
        pid = min(c["ids"], key=lambda x: (len(str(x)), str(x)))
        for tid in c["ids"]:
            person_of[tid] = pid
    return person_of, detail


def summarize(person_of):
    """{person id: [track ids]} — what the UI and the sidecar summary want."""
    out = {}
    for tid, pid in person_of.items():
        out.setdefault(pid, []).append(tid)
    return {k: sorted(v) for k, v in out.items()}
