"""Deduplicate promo footage using episode-relative timestamp overlap."""

import copy
import hashlib
import json
from pathlib import Path


class PromoDeduplicator:
    """Group duplicate cuts without relying on titles, dialogue or an LLM.

    Each segment must match the corresponding segment's episode and overlap by
    at least the configured fraction of the shorter segment. Segment count and
    playback order must match. The first occurrence represents the group; this
    is a deterministic choice, not an editorial ranking. Alternative cuts remain
    in the output for review.
    """

    def __init__(self, overlap_threshold: float = 0.8):
        if not 0 < overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be greater than 0 and at most 1")

        self.overlap_threshold = overlap_threshold

    def deduplicate(self, report: dict) -> dict:
        """Deduplicate one batch report, preserving source provenance."""
        unique = []
        duplicates = []
        invalid = []
        total = 0

        for window in report["windows"]:
            if window["status"] != "success":
                continue

            for index, original in enumerate(window["result"]["candidates"], 1):
                total += 1
                candidate = copy.deepcopy(original)
                candidate["candidate_id"] = f"{window['window_id']}-C{index:03d}"
                candidate["source_window"] = window["window_id"]

                if not candidate.get("valid", False):
                    invalid.append(candidate)
                    continue

                match, score = self._find_match(candidate, unique)
                if match is None:
                    unique.append(candidate)
                    continue

                duplicates.append({
                    "kept_candidate_id": match["candidate_id"],
                    "overlap_score": round(score, 4),
                    "match_type": self._match_type(candidate, match),
                    "candidate": candidate,
                })

        return {
            "model": report["model"],
            "source_report": report.get("report_path"),
            "overlap_threshold": self.overlap_threshold,
            "selection_rule": "First occurrence; alternative cuts retained, no ranking",
            "summary": {
                "input_candidates": total,
                "unique_candidates": len(unique),
                "duplicates": len(duplicates),
                "invalid_candidates": len(invalid),
            },
            "candidates": unique,
            "duplicates": duplicates,
            "invalid_candidates": invalid,
        }

    def save(self, result: dict, output_dir: str | Path) -> Path:
        """Save a report identified by its content, keeping different runs separate."""
        text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"deduplicated-{digest}.json"
        path.write_text(text)
        return path

    def _find_match(self, candidate: dict, kept: list[dict]) -> tuple:
        for representative in kept:
            score = self.overlap_score(candidate, representative)
            if score >= self.overlap_threshold:
                return representative, score

        return None, 0.0

    @staticmethod
    def overlap_score(left: dict, right: dict) -> float:
        """Return the weakest corresponding-segment overlap, from zero to one.

        Example: 00:10–00:40 and 00:12–00:42 share 28 of their 30 seconds:
        overlap = 28 / 30 = 0.933. Contained clips have a score of 1.
        Different episodes, segment counts or playback order never match.
        """
        left_segments = left["resolved_segments"]
        right_segments = right["resolved_segments"]

        if not left_segments or len(left_segments) != len(right_segments):
            return 0.0

        scores = []
        for first, second in zip(left_segments, right_segments):
            if first["episode"] != second["episode"]:
                return 0.0

            first_duration = first["end_seconds"] - first["start_seconds"]
            second_duration = second["end_seconds"] - second["start_seconds"]
            shortest_duration = min(first_duration, second_duration)
            if shortest_duration <= 0:
                return 0.0

            overlap_start = max(first["start_seconds"], second["start_seconds"])
            overlap_end = min(first["end_seconds"], second["end_seconds"])
            shared_seconds = max(0.0, overlap_end - overlap_start)
            scores.append(shared_seconds / shortest_duration)

        return min(scores)

    @staticmethod
    def _match_type(left: dict, right: dict) -> str:
        pairs = zip(left["resolved_segments"], right["resolved_segments"])
        exact = all(
            abs(first[boundary] - second[boundary]) < 0.000001
            for first, second in pairs
            for boundary in ("start_seconds", "end_seconds")
        )
        return "exact" if exact else "overlap"
