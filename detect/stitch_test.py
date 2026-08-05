#!/usr/bin/env python3
"""Tests for stitch.py — the rules that decide when two track fragments are one
person. Run: detect/.venv-detect/bin/python detect/stitch_test.py

These are the cases that cost something when they go wrong, so they are pinned:
a merge that should happen (the whole point), a merge that must NEVER happen
(two people on screen at once), and the ambiguous middle where guessing is worse
than leaving the ids split.
"""

import unittest

import numpy as np

import stitch as S


def emb(*v):
    """A unit-ish embedding from a few numbers — cosine only cares about direction."""
    a = np.array(v, dtype="float64")
    return a / np.linalg.norm(a)


def track(t0, t1, e, samples=8, colour=None):
    return {"emb": e, "colour": colour, "t0": t0, "t1": t1, "samples": samples}


ALICE = emb(1.0, 0.05, 0.02)
BOB = emb(0.05, 1.0, 0.03)


class Stitching(unittest.TestCase):
    def test_disjoint_fragments_of_one_person_merge(self):
        # The symptom this module exists for: one person, occluded mid-clip, comes
        # back as a new ByteTrack id.
        person_of, _ = S.stitch({1: track(0, 4, ALICE), 7: track(6, 10, ALICE)})
        self.assertEqual(person_of[1], person_of[7])

    def test_person_id_is_the_first_sighting(self):
        person_of, _ = S.stitch({9: track(6, 10, ALICE), 2: track(0, 4, ALICE)})
        self.assertEqual(set(person_of.values()), {2})

    def test_two_people_on_screen_together_never_merge(self):
        # Identical embeddings AND overlapping time: disjointness must win, or two
        # people in the same black shirt become one person forever.
        person_of, _ = S.stitch({1: track(0, 10, ALICE), 2: track(1, 9, ALICE)})
        self.assertNotEqual(person_of[1], person_of[2])

    def test_a_frame_of_handover_overlap_is_tolerated(self):
        # An id switch usually has both ids alive for a frame or two.
        person_of, _ = S.stitch({1: track(0, 5.1, ALICE), 2: track(5.0, 9, ALICE)})
        self.assertEqual(person_of[1], person_of[2])

    def test_different_people_stay_apart(self):
        person_of, _ = S.stitch({1: track(0, 4, ALICE), 2: track(6, 10, BOB)})
        self.assertNotEqual(person_of[1], person_of[2])

    def test_a_long_absence_is_not_stitched(self):
        person_of, _ = S.stitch({1: track(0, 4, ALICE), 2: track(400, 404, ALICE)})
        self.assertNotEqual(person_of[1], person_of[2])

    def test_ambiguity_is_declined_not_guessed(self):
        # Two candidates that look nearly the same: merging picks one at random in
        # effect, and a wrong merge is unrecoverable. Leave the ids split.
        near = emb(1.0, 0.05, 0.02)
        also = emb(1.0, 0.06, 0.02)
        person_of, detail = S.stitch({1: track(0, 3, near), 2: track(0, 3, also),
                                      3: track(5, 8, ALICE)})
        self.assertNotEqual(person_of[3], person_of[1])
        self.assertNotEqual(person_of[3], person_of[2])
        self.assertTrue(any("declined" in d for d in detail))

    def test_three_fragments_chain_into_one_person(self):
        # Iterative merging: the template improves as fragments join.
        person_of, _ = S.stitch({1: track(0, 3, ALICE), 2: track(4, 6, ALICE),
                                 3: track(7, 9, ALICE)})
        self.assertEqual(len(set(person_of.values())), 1)

    def test_a_fragment_with_no_appearance_is_left_alone(self):
        person_of, _ = S.stitch({1: track(0, 4, ALICE), 2: track(6, 10, None)})
        self.assertNotEqual(person_of[1], person_of[2])

    def test_colour_veto_is_off_by_default_and_works_when_asked(self):
        red = {"top": np.array([1.0, 0.0], dtype="float32")}
        blue = {"top": np.array([0.0, 1.0], dtype="float32")}
        tracks = {1: track(0, 4, ALICE, colour=red), 2: track(6, 10, ALICE, colour=blue)}
        sim = lambda a, b: (float(np.minimum(a["top"], b["top"]).sum()) if a and b else None, {})  # noqa: E731
        merged, _ = S.stitch(tracks, colour_sim=sim)
        self.assertEqual(merged[1], merged[2], "default must not veto on colour")
        split, _ = S.stitch(tracks, colour_veto=0.5, colour_sim=sim)
        self.assertNotEqual(split[1], split[2], "an enabled veto must reject a clash")

    def test_summarize_groups_tracks_under_their_person(self):
        person_of, _ = S.stitch({1: track(0, 3, ALICE), 5: track(4, 6, ALICE),
                                 9: track(0, 6, BOB)})
        groups = S.summarize(person_of)
        self.assertEqual(groups[1], [1, 5])
        self.assertEqual(groups[9], [9])

    def test_a_person_keeps_a_span_LIST_not_an_envelope(self):
        # 1 and 3 are one person either side of a gap; 7 sits INSIDE that gap's
        # envelope but overlaps fragment 1, so it cannot be the same person. A
        # merged min/max envelope (0..11) would have compared 7 against the whole
        # range and hidden the overlap.
        # 7 looks close enough to merge on appearance alone (cos 0.90, well over the
        # 0.72 floor) but far enough not to make the 1-3 merge ambiguous, so ONLY the
        # span logic can keep it out.
        lookalike = emb(1.0, 0.48, 0.02)
        person_of, _ = S.stitch({1: track(0, 3, ALICE), 3: track(8, 11, ALICE),
                                 7: track(1.5, 2.0, lookalike)})
        self.assertEqual(person_of[1], person_of[3], "the gap is bridged")
        self.assertNotEqual(person_of[7], person_of[1], "but not by someone standing there")

    def test_no_stitched_person_is_ever_in_two_places_at_once(self):
        # The structural guarantee, checked over a messy mix rather than one case.
        tracks = {1: track(0, 3, ALICE), 2: track(0.5, 2, BOB), 3: track(4, 6, ALICE),
                  4: track(4.2, 5, BOB), 5: track(7, 9, ALICE), 6: track(7, 9.5, BOB)}
        person_of, _ = S.stitch(tracks)
        spans = {}
        for tid, pid in person_of.items():
            spans.setdefault(pid, []).append((tracks[tid]["t0"], tracks[tid]["t1"]))
        for pid, sp in spans.items():
            sp.sort()
            for a, b in zip(sp, sp[1:]):
                self.assertLessEqual(min(a[1], b[1]) - max(a[0], b[0]), S.STITCH_OVERLAP_TOL_S,
                                     f"person {pid} occupies two times at once: {sp}")
        # And the two people really were recovered from six fragments.
        self.assertEqual(len(spans), 2, f"expected 2 people, got {len(spans)}: {spans}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
