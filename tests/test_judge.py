"""Judge integration tests with synthetic footage and mocked Gemini responses."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from google.genai import types

from scripts.preprocessing import Entry, Episode, Scene
from scripts.promo_generator import GenerationConfig, GeminiJSONClient
from scripts.promo_judge import (
    AssessmentBatch,
    JudgeConfig,
    PostingOrder,
    PromoExcerptBuilder,
    PromoJudge,
)


def entry(source_id, start, end, text):
    return Entry(
        source_id, 1, "dialogue", "PERSON", "PERSON",
        f"00:{start:05.2f}", f"00:{end:05.2f}", start, end, text,
    )


def episode(number):
    entries = [
        entry(f"E{number}-0", 0, 12, "Partly visible event"),
        entry(f"E{number}-1", 10, 13, "Opening conflict"),
        entry(f"E{number}-2", 12, 15, "Alternate opening"),
        entry(f"E{number}-3", 15, 30, "Rising stakes"),
        entry(f"E{number}-4", 37, 40, "Unanswered threat"),
        entry(f"E{number}-5", 55, 60, "Outside the clip"),
    ]
    scene = Scene(1, "ROOM", "00:00.00", "01:00.00", 0, 60, entries)
    return Episode(number, "fixture.txt", "HIDDEN LOGLINE", "HIDDEN NOTES", [scene], [])


def candidate(candidate_id, ep=501, start=10):
    start_id = f"E{ep}-1" if start == 10 else f"E{ep}-2"
    return {
        "candidate_id": candidate_id,
        "title": "HIDDEN TITLE",
        "hook": "HIDDEN GENERATOR PRAISE",
        "cliffhanger": "HIDDEN GENERATOR ENDING",
        "why_it_works": "HIDDEN GENERATOR WHY",
        "valid": True,
        "duration_seconds": 40 - start,
        "resolved_segments": [{
            "episode": ep, "start_id": start_id, "end_id": f"E{ep}-4",
            "start": f"00:{start}.00", "end": "00:40.00",
            "start_seconds": start, "end_seconds": 40,
        }],
    }


def assessment(excerpt, score=3):
    entries = [event for segment in excerpt["segments"] for event in segment["entries"]]
    opening = next(event["source_id"] for event in entries if event["clip_start_seconds"] < 3)
    ending = next(event["source_id"] for event in entries if event["clip_end_seconds"] >= excerpt["duration_seconds"] - 5)
    result = {
        "candidate_id": excerpt["candidate_id"],
        "hook_summary": "Observed opening",
        "cliffhanger_question": "What happens next?",
        "why_it_works": "The conflict escalates into an unresolved threat.",
        "concerns": [],
    }
    for name in ("hook", "cliffhanger", "clarity", "escalation"):
        result[name] = {
            "score": score,
            "evidence_ids": [ending if name == "cliffhanger" else opening],
            "reason": "Evidence-based fixture",
        }
    return result


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [episode(501), episode(502)]
        self.builder = PromoExcerptBuilder(self.episodes)
        self.excerpt = self.builder.build(candidate("A"))

    def test_excerpt_hides_generator_context_and_clips_overlapping_events(self):
        serialized = json.dumps(self.excerpt)
        self.assertNotIn("HIDDEN", serialized)
        self.assertNotIn("Outside the clip", serialized)
        first = self.excerpt["segments"][0]["entries"][0]
        self.assertTrue(first["partial_boundary"])
        self.assertEqual(first["clip_start_seconds"], 0)
        self.assertEqual(first["clip_end_seconds"], 2)

    def test_multisegment_offsets_follow_playback_order(self):
        promo = candidate("Montage")
        promo["resolved_segments"] += candidate("Other", ep=502)["resolved_segments"]
        excerpt = self.builder.build(promo)
        self.assertEqual(excerpt["duration_seconds"], 60)
        self.assertEqual(excerpt["segments"][1]["entries"][0]["clip_start_seconds"], 30)

    def test_evidence_must_come_from_selected_footage(self):
        scored = assessment(self.excerpt)
        scored["hook"]["evidence_ids"] = ["E501-5"]
        with self.assertRaisesRegex(ValueError, "outside"):
            PromoJudge._validate_assessments(AssessmentBatch(assessments=[scored]), [self.excerpt])

    def test_hook_and_cliffhanger_evidence_must_match_playback_timing(self):
        scored = assessment(self.excerpt)
        scored["hook"]["evidence_ids"] = ["E501-4"]
        with self.assertRaisesRegex(ValueError, "first 3s"):
            PromoJudge._validate_assessments(AssessmentBatch(assessments=[scored]), [self.excerpt])
        scored = assessment(self.excerpt)
        scored["cliffhanger"]["evidence_ids"] = ["E501-1"]
        with self.assertRaisesRegex(ValueError, "ending"):
            PromoJudge._validate_assessments(AssessmentBatch(assessments=[scored]), [self.excerpt])

    def test_judge_cannot_omit_or_repeat_candidates(self):
        for items in [[], [assessment(self.excerpt)] * 2]:
            with self.assertRaisesRegex(ValueError, "exactly once"):
                PromoJudge._validate_assessments(AssessmentBatch(assessments=items), [self.excerpt])

    def test_core_quality_gate_prevents_weak_endings_from_ranking_first(self):
        judge = PromoJudge(GenerationConfig(), JudgeConfig(top_compare_count=0))
        strong = assessment(self.excerpt, 3)
        weak = assessment(self.excerpt, 4)
        weak["candidate_id"] = "B"
        weak["cliffhanger"]["score"] = 0
        winners, _ = judge._choose_cuts(
            [[candidate("A")], [candidate("B")]], {"A": strong, "B": weak}
        )
        self.assertEqual(winners[0]["candidate_id"], "A")
        self.assertFalse(winners[1]["recommended"])

    def test_full_ranking_selects_better_cut_and_retains_all_required_fields(self):
        dedup = {
            "model": "source-model",
            "candidates": [candidate("A"), candidate("C", ep=502)],
            "duplicates": [{"kept_candidate_id": "A", "candidate": candidate("B", start=12)}],
        }
        original = copy.deepcopy(dedup)

        def complete(prompt, response_model, key_names, job_name, force=False, validate=None):
            marker = "CANDIDATES\n" if response_model is AssessmentBatch else "SHORTLIST\n"
            items = json.loads(prompt.split(marker)[-1])
            if response_model is AssessmentBatch:
                data = {"assessments": [assessment(item, {"A": 2, "B": 4, "C": 3}[item["candidate_id"]]) for item in items]}
            else:
                data = {"order": [
                    {"candidate_id": item["candidate_id"], "posting_rationale": "Fixture posting rationale"}
                    for item in reversed(items)
                ]}
            parsed = response_model.model_validate(data)
            validate(parsed)
            return {"data": data, "model_used": "mock", "usage": {}, "output_directory": "mock"}

        with tempfile.TemporaryDirectory() as directory:
            judge = PromoJudge(GenerationConfig(), JudgeConfig(batch_size=2), directory)
            judge.client.complete = MagicMock(side_effect=complete)
            with contextlib.redirect_stdout(io.StringIO()):
                ranked = judge.rank(dedup, self.episodes, ["TEST_KEY"], request_delay_seconds=0)

            self.assertEqual(judge.client.complete.call_count, 3)
            self.assertEqual(ranked["summary"]["cuts_judged"], 3)
            self.assertEqual(ranked["summary"]["alternative_cuts_selected"], 1)
            self.assertEqual([item["candidate_id"] for item in ranked["candidates"]], ["C", "B"])
            self.assertEqual(ranked["candidates"][0]["score_rank"], 2)
            for item in ranked["candidates"]:
                for field in ("rank", "resolved_segments", "hook", "cliffhanger", "why_it_works", "posting_rationale"):
                    self.assertIn(field, item)
            path = Path(ranked["output_path"])
            self.assertTrue(path.exists())
            self.assertIn("Episode 502", path.with_suffix(".md").read_text())
            self.assertEqual(dedup, original)

    @patch("scripts.promo_generator.genai.Client")
    @patch.dict("os.environ", {"TEST_JUDGE_KEY": "judge-test-only"})
    def test_judge_calls_share_quota_and_cache_validated_responses(self, constructor):
        data = {"assessments": [assessment(self.excerpt)]}
        response = types.GenerateContentResponse(candidates=[types.Candidate(
            finish_reason="STOP",
            content=types.Content(parts=[types.Part(text=json.dumps(data))]),
        )])
        client = MagicMock()
        constructor.return_value.__enter__.return_value = client
        client.models.generate_content.return_value = response

        with tempfile.TemporaryDirectory() as directory:
            transport = GeminiJSONClient(
                GenerationConfig(), directory, Path(directory) / "usage.json"
            )
            def validate(parsed):
                PromoJudge._validate_assessments(parsed, [self.excerpt])
            with contextlib.redirect_stdout(io.StringIO()):
                first = transport.complete("fixture", AssessmentBatch, ["TEST_JUDGE_KEY"], "scores", validate=validate)
                cached = transport.complete("fixture", AssessmentBatch, ["TEST_JUDGE_KEY"], "scores", validate=validate)
            self.assertEqual(first, cached)
            self.assertEqual(transport.budget.used("judge-test-only"), 1)
            self.assertEqual(client.models.generate_content.call_count, 1)


if __name__ == "__main__":
    unittest.main()
