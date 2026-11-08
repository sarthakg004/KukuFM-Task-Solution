"""Generic timestamp rules tested without any series-specific fixtures."""

import copy
import unittest

from scripts.deduplication import PromoDeduplicator


def candidate(title, segments, valid=True):
    return {
        "title": title,
        "valid": valid,
        "resolved_segments": [
            {"episode": episode, "start_seconds": start, "end_seconds": end}
            for episode, start, end in segments
        ],
    }


def report(*candidates):
    return {
        "model": "test-model",
        "windows": [{
            "window_id": "W001",
            "status": "success",
            "result": {"candidates": list(candidates)},
        }],
    }


class DeduplicationTests(unittest.TestCase):
    def setUp(self):
        self.deduplicator = PromoDeduplicator(overlap_threshold=0.8)

    def test_identical_footage_matches_despite_different_titles(self):
        first = candidate("Any title", [(104, 10, 40)])
        second = candidate("Completely different title", [(104, 10, 40)])
        result = self.deduplicator.deduplicate(report(first, second))
        self.assertEqual(result["summary"]["unique_candidates"], 1)
        self.assertEqual(result["duplicates"][0]["match_type"], "exact")

    def test_shifted_boundaries_match_at_threshold(self):
        first = candidate("A", [(104, 10, 40)])
        second = candidate("B", [(104, 16, 46)])
        self.assertAlmostEqual(self.deduplicator.overlap_score(first, second), 0.8)
        self.assertEqual(
            self.deduplicator.deduplicate(report(first, second))["summary"]["duplicates"], 1
        )

    def test_same_times_in_different_episodes_do_not_match(self):
        first = candidate("Same title", [(7, 10, 40)])
        second = candidate("Same title", [(8, 10, 40)])
        self.assertEqual(self.deduplicator.overlap_score(first, second), 0)

    def test_contained_cut_matches_but_partial_or_touching_cut_does_not(self):
        first = candidate("A", [(1, 10, 60)])
        contained = candidate("B", [(1, 20, 40)])
        partial = candidate("C", [(1, 50, 80)])
        touching = candidate("D", [(1, 60, 90)])
        self.assertEqual(self.deduplicator.overlap_score(first, contained), 1)
        self.assertLess(self.deduplicator.overlap_score(first, partial), 0.8)
        self.assertEqual(self.deduplicator.overlap_score(first, touching), 0)

    def test_montage_order_and_all_segments_must_match(self):
        first = candidate("A", [(10, 0, 20), (20, 10, 30)])
        near = candidate("B", [(10, 1, 21), (20, 11, 31)])
        reversed_order = candidate("C", [(20, 10, 30), (10, 0, 20)])
        extra = candidate("D", [(10, 0, 20), (20, 10, 30), (30, 0, 10)])
        different_end = candidate("E", [(10, 0, 20), (20, 40, 60)])
        self.assertGreaterEqual(self.deduplicator.overlap_score(first, near), 0.8)
        for other in [reversed_order, extra, different_end]:
            self.assertEqual(self.deduplicator.overlap_score(first, other), 0)

    def test_matches_only_kept_representatives_to_avoid_chained_merges(self):
        result = self.deduplicator.deduplicate(report(
            candidate("A", [(1, 0, 30)]),
            candidate("B", [(1, 6, 36)]),
            candidate("C", [(1, 12, 42)]),
        ))
        self.assertEqual(result["summary"]["unique_candidates"], 2)

    def test_originals_and_invalid_candidates_are_preserved(self):
        original = report(
            candidate("A", [(1, 10, 40)]),
            candidate("B", [(1, 10, 40)]),
            candidate("Invalid", [(2, 0, 10)], valid=False),
        )
        before = copy.deepcopy(original)
        result = self.deduplicator.deduplicate(original)
        self.assertEqual(original, before)
        self.assertEqual(result["summary"], {
            "input_candidates": 3,
            "unique_candidates": 1,
            "duplicates": 1,
            "invalid_candidates": 1,
        })
        self.assertEqual(result["duplicates"][0]["candidate"]["title"], "B")

    def test_threshold_is_configurable(self):
        first = candidate("A", [(1, 0, 30)])
        second = candidate("B", [(1, 6, 36)])
        result = PromoDeduplicator(0.9).deduplicate(report(first, second))
        self.assertEqual(result["summary"]["unique_candidates"], 2)
        with self.assertRaises(ValueError):
            PromoDeduplicator(0)


if __name__ == "__main__":
    unittest.main()
