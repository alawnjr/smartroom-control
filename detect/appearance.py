"""Clothing-colour appearance descriptors, for keeping one person's identity.

Both identity paths in this project fragment on occlusion:

  * offline (`action.py`) tracks with ByteTrack, which has NO appearance model at
    all — when one person walks in front of another the covered person's track
    dies and a fresh id starts, so the recorded clip shows a colour/label change
    on someone who never went anywhere.
  * live (`live_infer.py`) has a ReID embedding, but a whole-box embedding taken
    while somebody is half-hidden describes both people at once.

What survives an occlusion is what a human uses to answer "is that the same
person": the colour of their shirt and their trousers. So sample exactly that,
from the pose keypoints rather than the box (a box is full of floor, wall and
whoever is standing in front), and keep the two garments SEPARATE — two people in
white shirts are told apart by their jeans, and the discriminating garment is not
known in advance.

Design notes, in the order they matter:

  * Two regions, not one. `top` = the shoulder-to-hip quad, `bot` = hip-to-knee.
    Each is scored on its own and the results combined by the caller, so a person
    sitting behind a desk (no legs in view) still matches on their shirt.
  * The quad is shrunk toward its own centre before sampling. A shoulder-to-hip
    quad drawn through the keypoints includes background either side of the torso;
    at 60% width the sample is cloth.
  * CHROMA and LIGHTNESS are separate histograms. Room lighting and camera
    exposure move L far more than they move a/b, so the two cannot share one
    score — but lightness is the only thing that separates black trousers from
    navy ones, so it cannot be dropped either. `CHROMA_W` sets the balance.
  * Histogram INTERSECTION, not a distance on means. A striped or two-tone shirt
    has no meaningful mean colour; its histogram still matches itself.
  * Nothing here trusts a single frame. `blend` accumulates a template over time
    and the callers only feed it samples they believe (see `occlusion_frac`),
    which is the actual fix for occlusion: stop learning from the frames where
    the crop is somebody else.

Pure numpy + cv2, no torch — cheap enough to run per person per frame on the live
path (~0.2 ms), and importable by the offline pass without a GPU.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# COCO-17 indices.
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

KP_CONF = float(os.environ.get("SMARTROOM_APPEARANCE_KP_CONF", "0.4"))
# Bin counts. Coarse on purpose: a few hundred cloth pixels cannot fill a fine
# histogram, and two samples of the same shirt would then intersect at almost
# nothing.
AB_BINS = int(os.environ.get("SMARTROOM_APPEARANCE_AB_BINS", "8"))
L_BINS = int(os.environ.get("SMARTROOM_APPEARANCE_L_BINS", "8"))
# The a/b histogram covers a NARROW window around neutral, not the full 0..255.
# Measured on this room's garment pixels: a and b span p05..p95 of 112..138 and
# 114..140 — 26 of 255 levels. Binning the full range put every person's chroma in
# the same one or two bins and the descriptor could not tell anyone apart (shirt
# AUC 0.59, barely better than a coin). Values outside the window are CLIPPED into
# the end bins rather than dropped, because cv2.calcHist silently discards
# out-of-range pixels — a genuinely red shirt would have sampled as nothing.
AB_LO = float(os.environ.get("SMARTROOM_APPEARANCE_AB_LO", "104"))
AB_HI = float(os.environ.get("SMARTROOM_APPEARANCE_AB_HI", "152"))
# How much of a region's score is chroma vs lightness. Lightness carries most of
# the signal in THIS room — the clothing is largely grey/black/white/denim, whose
# a/b differences are a few levels while L ranges from 48 to 180 between people —
# so the split is deliberately close to even rather than chroma-led. Sweep it with
# eval_appearance.py --sweep if the wardrobe changes.
CHROMA_W = float(os.environ.get("SMARTROOM_APPEARANCE_CHROMA_W", "0.45"))
# Fraction of the quad's width/height kept when shrinking toward its centre.
SHRINK = float(os.environ.get("SMARTROOM_APPEARANCE_SHRINK", "0.6"))
# A region with fewer sampled pixels than this is not a colour measurement.
MIN_PIXELS = int(os.environ.get("SMARTROOM_APPEARANCE_MIN_PIXELS", "60"))

REGIONS = ("top", "bot")


def _quad(kpts, conf, a, b, c, d):
    """The polygon (a,b) -> (c,d), shrunk toward its centre, or None if any
    corner is unreliable. `a`/`b` are the upper pair, `c`/`d` the lower."""
    idx = (a, b, c, d)
    if any(conf[i] < KP_CONF for i in idx):
        return None
    pts = np.array([[kpts[a][0], kpts[a][1]], [kpts[b][0], kpts[b][1]],
                    [kpts[d][0], kpts[d][1]], [kpts[c][0], kpts[c][1]]], dtype=np.float32)
    centre = pts.mean(axis=0)
    return (centre + (pts - centre) * SHRINK).astype(np.int32)


def _region_polys(kpts, conf):
    """{region: polygon} for whichever regions this pose actually shows.

    The legs fall back from knees to ankles: a person walking away has clean
    ankles and knees hidden by their own coat as often as the reverse.
    """
    out = {}
    top = _quad(kpts, conf, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if top is not None:
        out["top"] = top
    bot = _quad(kpts, conf, L_HIP, R_HIP, L_KNEE, R_KNEE)
    if bot is None:
        bot = _quad(kpts, conf, L_HIP, R_HIP, L_ANKLE, R_ANKLE)
    if bot is not None:
        out["bot"] = bot
    return out


def _hists(lab, mask):
    """(chroma 2D hist, lightness 1D hist), L1-normalized, or None if too few
    pixels. Both are float32 and flat, so a descriptor is plain array data."""
    n = int(mask.sum())
    if n < MIN_PIXELS:
        return None
    # Clip a/b into the window (see AB_LO/AB_HI) so out-of-window chroma lands in
    # an end bin instead of being dropped by calcHist.
    ab_src = np.clip(lab[:, :, 1:3], AB_LO, AB_HI - 1e-3)
    ab = cv2.calcHist([ab_src.astype(np.float32)], [0, 1], mask,
                      [AB_BINS, AB_BINS], [AB_LO, AB_HI, AB_LO, AB_HI])
    li = cv2.calcHist([lab], [0], mask, [L_BINS], [0, 256])
    ab = (ab / max(1.0, ab.sum())).astype(np.float32).ravel()
    li = (li / max(1.0, li.sum())).astype(np.float32).ravel()
    return np.concatenate([ab, li])


def describe(frame_bgr, kpts, conf):
    """Sample this person's garment colours from one frame.

    Returns {"top": vec, "bot": vec} with only the regions that were measurable
    — an empty dict when the pose shows neither, which is a real answer ("no
    appearance evidence in this frame"), not a failure.
    """
    if frame_bgr is None or len(kpts) < 17:
        return {}
    polys = _region_polys(kpts, conf)
    if not polys:
        return {}
    h, w = frame_bgr.shape[:2]
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    out = {}
    for name, poly in polys.items():
        # Clip to the image before rasterizing: a person at the frame edge has
        # keypoints outside it, and fillConvexPoly would silently wrap.
        p = poly.copy()
        p[:, 0] = np.clip(p[:, 0], 0, w - 1)
        p[:, 1] = np.clip(p[:, 1], 0, h - 1)
        x0, y0 = p[:, 0].min(), p[:, 1].min()
        x1, y1 = p[:, 0].max(), p[:, 1].max()
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillConvexPoly(mask, p - [x0, y0], 255)
        hv = _hists(lab[y0:y1 + 1, x0:x1 + 1], mask)
        if hv is not None:
            out[name] = hv
    return out


def _inter(a, b):
    """Histogram intersection of two same-length L1-normalized vectors."""
    return float(np.minimum(a, b).sum())


def region_similarity(a, b):
    """0..1 for one region: chroma intersection and lightness intersection,
    blended by CHROMA_W. Both histograms live in one vector (chroma first)."""
    n_ab = AB_BINS * AB_BINS
    return CHROMA_W * _inter(a[:n_ab], b[:n_ab]) + (1.0 - CHROMA_W) * _inter(a[n_ab:], b[n_ab:])


def similarity(a, b, w_top=1.0, w_bot=1.0):
    """Colour similarity between two descriptors, and which regions decided it.

    Returns (sim, parts) where parts is {region: sim} for the regions BOTH sides
    measured. sim is None when they have no region in common — "unknown", which a
    caller must not read as "different": one person seen standing and then seated
    legitimately shares only their shirt.
    """
    if not a or not b:
        return None, {}
    parts, wsum, acc = {}, 0.0, 0.0
    for name, w in (("top", w_top), ("bot", w_bot)):
        if name in a and name in b:
            s = region_similarity(a[name], b[name])
            parts[name] = round(s, 4)
            acc += w * s
            wsum += w
    if wsum == 0:
        return None, {}
    return acc / wsum, parts


def blend(template, sample, momentum):
    """Fold a fresh sample into a per-region template (EMA per region).

    Regions are folded independently, so a frame that saw only the shirt updates
    the shirt and leaves the trousers as they were rather than dropping them.
    """
    if not sample:
        return template
    out = dict(template or {})
    for name, vec in sample.items():
        prev = out.get(name)
        out[name] = vec.copy() if prev is None else (momentum * prev + (1.0 - momentum) * vec)
    return out


def occlusion_frac(box, others):
    """Largest fraction of `box` covered by any other person's box.

    This is the gate that matters. A descriptor sampled while someone stands in
    front of this person describes both of them, and folding it into the template
    is how an identity gets poisoned precisely when it is about to be needed. The
    callers refuse to LEARN from such a frame; they still MATCH, because the
    person coming back out of the occlusion is clean and is being compared against
    a template that was never corrupted.

    Boxes are (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = box
    area = max(1.0, (x2 - x1) * (y2 - y1))
    worst = 0.0
    for o in others:
        ix = max(0.0, min(x2, o[2]) - max(x1, o[0]))
        iy = max(0.0, min(y2, o[3]) - max(y1, o[1]))
        worst = max(worst, ix * iy / area)
    return worst
