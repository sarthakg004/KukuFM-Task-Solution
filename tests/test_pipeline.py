"""Local regression checks; all model calls are mocked."""

import contextlib
import io
import json
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from google.genai import errors, types

from scripts.preprocessing import ScreenplayParser, ScreenplayPreprocessor
from scripts.request_budget import DailyBudgetExceeded, DailyRequestBudget
from scripts.promo_generator import (
    CandidateBatch,
    CandidateValidator,
    GenerationConfig,
    GeminiConnectionError,
    PromoGenerator,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "I Married a Broken Soldier. He was a Duke"


@unittest.skipUnless(INPUT.is_dir(), "Private input pack is not included in the public repository")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocessor = ScreenplayPreprocessor(INPUT)
        cls.episodes = cls.preprocessor.episodes
        cls.window = cls.preprocessor.create_windows()[0]
        entries = cls.episodes[0].entries
        end = next(entry for entry in entries if entry.end_seconds == 29.8)
        cls.candidate = {
            "title": "Test fixture",
            "segments": [{"start_id": entries[0].source_id, "end_id": end.source_id}],
            "hook": "Fixture",
            "cliffhanger": "Fixture",
            "standalone_context": "Fixture",
            "why_it_works": "Fixture",
            "concerns": "Fixture",
        }

    def response(self):
        text = json.dumps({"candidates": [self.candidate]})
        return types.GenerateContentResponse(candidates=[
            types.Candidate(content=types.Content(parts=[types.Part(text=text)]))
        ])

    def test_source_coverage(self):
        text = self.preprocessor.full_text()
        self.assertEqual(len(self.episodes), 50)
        self.assertEqual(sum(len(e.entries) for e in self.episodes), 3727)

        for episode in self.episodes:
            lines = Path(episode.source_file).read_text().splitlines()
            for entry in episode.entries:
                self.assertEqual(text.count(f"[{entry.source_id} |"), 1)
                self.assertEqual(lines[entry.source_line - 1], f"**{entry.label}:**")

    def test_window_coverage(self):
        windows = self.preprocessor.create_windows(10, 5)
        self.assertEqual(len(windows), 9)
        self.assertEqual(windows[-1].episode_numbers, list(range(41, 51)))
        covered = {number for window in windows for number in window.episode_numbers}
        self.assertEqual(covered, set(range(1, 51)))

        with self.assertRaises(ValueError):
            self.preprocessor.create_windows(5, 10)

    def test_malformed_source_is_rejected(self):
        source = Path(self.episodes[0].source_file).read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text(source.replace("00:04.80", "00:99.80", 1))
            with self.assertRaises(ValueError):
                ScreenplayParser().parse_file(path)

    def test_candidate_validation(self):
        validator = CandidateValidator(self.episodes, self.window)
        batch = CandidateBatch(candidates=[self.candidate])
        self.assertTrue(validator.validate(batch)[0]["valid"])

        invalid_segments = [
            [],
            [{"start_id": "missing", "end_id": "missing"}],
            self.candidate["segments"] * 2,
            [{
                "start_id": self.episodes[0].entries[0].source_id,
                "end_id": self.episodes[1].entries[-1].source_id,
            }],
            [{
                "start_id": self.episodes[0].entries[0].source_id,
                "end_id": self.episodes[0].entries[-1].source_id,
            }],
        ]
        for segments in invalid_segments:
            batch = CandidateBatch(candidates=[{**self.candidate, "segments": segments}])
            self.assertFalse(validator.validate(batch)[0]["valid"])

    @patch("scripts.promo_generator.time.sleep")
    @patch("scripts.promo_generator.genai.Client")
    @patch.dict("os.environ", {"TEST_GEMINI_KEY": "test-only"})
    def test_retry_fallback_and_cache(self, constructor, sleep):
        client = MagicMock()
        constructor.return_value.__enter__.return_value = client
        client.models.generate_content.side_effect = [
            errors.ServerError(503, {"error": {"message": "Unavailable"}}),
            errors.ServerError(504, {"error": {"message": "Deadline"}}),
            self.response(),
        ]
        config = GenerationConfig(
            model="test-primary",
            fallback_model="test-backup",
            key_name="TEST_GEMINI_KEY",
            retries=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            generator = PromoGenerator(config, directory, Path(directory) / "usage.json")
            with contextlib.redirect_stdout(io.StringIO()):
                result = generator.generate(self.window, self.episodes)
                cached = generator.generate(self.window, self.episodes)

            self.assertEqual(result["model_used"], config.fallback_model)
            self.assertEqual(result, cached)
            self.assertEqual(client.models.generate_content.call_count, 3)
            self.assertEqual(generator.budget.used("test-only"), 3)
            request_config = client.models.generate_content.call_args.kwargs["config"]
            self.assertNotIn("temperature", request_config.model_dump(exclude_none=True))
            sleep.assert_called_once_with(5)
            self.assertTrue(Path(result["output_directory"], "response.json").exists())
            self.assertEqual(len(result["attempts"]), 3)

    @patch("scripts.promo_generator.time.sleep")
    def test_nonretryable_error_stops(self, sleep):
        client = MagicMock()
        client.models.generate_content.side_effect = errors.ClientError(
            403, {"error": {"message": "Denied"}}
        )
        generator = PromoGenerator(GenerationConfig())
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "403"):
                    generator.budget = DailyRequestBudget(Path(directory) / "usage.json")
                    generator._request(client, "fixture", Path(directory), "test-only")

        self.assertEqual(client.models.generate_content.call_count, 1)
        sleep.assert_not_called()

    def test_truncated_response_is_saved_and_rejected(self):
        response = types.GenerateContentResponse(candidates=[
            types.Candidate(
                finish_reason="MAX_TOKENS",
                content=types.Content(parts=[types.Part(text='{"candidates": [{"title": "cut')]),
            )
        ])
        generator = PromoGenerator(GenerationConfig())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "token limit"):
                generator._parse_response(response, path)

            self.assertTrue((path / "response.json").exists())
            self.assertEqual((path / "response.txt").read_text(), response.text)
            self.assertFalse((path / "candidates.json").exists())

    def test_high_thinking_request_uses_maximum_output_budget(self):
        config = GenerationConfig(
            thinking_level="high",
            max_output_tokens=65536,
            timeout_seconds=300,
        )
        client = MagicMock()
        client.models.generate_content.return_value = self.response()

        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                PromoGenerator(config, budget_path=Path(directory) / "usage.json")._request(
                    client, "fixture", Path(directory), "test-only"
                )

        request = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(request.thinking_config.thinking_level, types.ThinkingLevel.HIGH)
        self.assertEqual(request.max_output_tokens, 65536)
        self.assertNotIn("temperature", request.model_dump(exclude_none=True))

    def test_malformed_json_has_readable_error(self):
        response = types.GenerateContentResponse(candidates=[
            types.Candidate(
                finish_reason="STOP",
                content=types.Content(parts=[types.Part(text='{"candidates":')]),
            )
        ])
        generator = PromoGenerator(GenerationConfig())

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "invalid or incomplete"):
                generator._parse_response(response, Path(directory))

    @patch.dict("os.environ", {"TEST_KEY_A": "a", "TEST_KEY_B": "b"})
    @patch("scripts.promo_generator.time.sleep")
    @patch.object(PromoGenerator, "generate")
    def test_key_trial_stops_on_success(self, generate, sleep):
        generate.side_effect = [RuntimeError("HTTP 503"), {"candidates": []}]
        generator = PromoGenerator(GenerationConfig(retry_delay_seconds=0))

        with contextlib.redirect_stdout(io.StringIO()):
            result = generator.generate_with_keys(
                self.window, self.episodes, ["TEST_KEY_A", "TEST_KEY_B"]
            )

        self.assertEqual(result, {"candidates": []})
        self.assertEqual(generate.call_count, 2)

    @patch.dict("os.environ", {"TEST_KEY_A": "a", "TEST_KEY_B": "b"})
    @patch.object(PromoGenerator, "generate")
    def test_network_failure_does_not_cycle_keys(self, generate):
        generate.side_effect = GeminiConnectionError("Network unavailable")
        generator = PromoGenerator(GenerationConfig())

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(GeminiConnectionError):
                generator.generate_with_keys(
                    self.window, self.episodes, ["TEST_KEY_A", "TEST_KEY_B"]
                )

        self.assertEqual(generate.call_count, 1)

    @patch("scripts.promo_generator.time.sleep")
    @patch.object(PromoGenerator, "generate_with_keys")
    def test_batch_preserves_successes_and_records_failures(self, generate, sleep):
        generate.side_effect = [
            {"candidates": []},
            RuntimeError("HTTP 503"),
            {"candidates": []},
        ]
        windows = self.preprocessor.create_windows()[:3]

        with tempfile.TemporaryDirectory() as directory:
            generator = PromoGenerator(GenerationConfig(), directory)
            with contextlib.redirect_stdout(io.StringIO()):
                report = generator.generate_windows(
                    windows, self.episodes, ["TEST_KEY"], request_delay_seconds=0
                )

            self.assertEqual(
                [item["status"] for item in report["windows"]],
                ["success", "failed", "success"],
            )
            self.assertEqual(json.loads(Path(report["report_path"]).read_text()), report)
            self.assertEqual(generate.call_count, 3)

    def test_daily_budget_persists_and_resets_on_new_day(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.json"
            with patch.object(DailyRequestBudget, "_today", return_value="2026-11-08"):
                budget = DailyRequestBudget(path, limit=5)
                for _ in range(5):
                    budget.reserve("secret-test-value", "TEST_KEY")

                restarted = DailyRequestBudget(path, limit=5)
                self.assertEqual(restarted.used("secret-test-value"), 5)
                with self.assertRaises(DailyBudgetExceeded):
                    restarted.reserve("secret-test-value", "ALIAS_OF_SAME_KEY")
                self.assertNotIn("secret-test-value", path.read_text())

            with patch.object(DailyRequestBudget, "_today", return_value="2026-11-09"):
                self.assertEqual(budget.used("secret-test-value"), 0)
                budget.reserve("secret-test-value", "TEST_KEY")
                self.assertEqual(budget.used("secret-test-value"), 1)

    @patch.dict("os.environ", {"TEST_KEY_A": "a", "TEST_KEY_B": "b"})
    def test_rotation_chooses_the_least_used_key(self):
        selected = []

        def fake_generate(generator, window, episodes):
            key_name = generator.config.key_name
            generator.budget.reserve(os.environ[key_name], key_name)
            selected.append(key_name)
            return {"candidates": []}

        with tempfile.TemporaryDirectory() as directory:
            generator = PromoGenerator(
                GenerationConfig(retry_delay_seconds=0),
                directory,
                Path(directory) / "usage.json",
            )
            with patch.object(PromoGenerator, "generate", fake_generate):
                with contextlib.redirect_stdout(io.StringIO()):
                    for _ in range(4):
                        generator.generate_with_keys(
                            self.window, self.episodes, ["TEST_KEY_A", "TEST_KEY_B"]
                        )

        self.assertEqual(selected, ["TEST_KEY_A", "TEST_KEY_B"] * 2)

    def test_budget_prevents_a_sixth_api_call(self):
        client = MagicMock()
        client.models.generate_content.return_value = self.response()

        with tempfile.TemporaryDirectory() as directory:
            generator = PromoGenerator(
                GenerationConfig(daily_requests_per_key=5),
                directory,
                Path(directory) / "usage.json",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(5):
                    generator._request(client, "fixture", Path(directory), "test-only")
                with self.assertRaises(DailyBudgetExceeded):
                    generator._request(client, "fixture", Path(directory), "test-only")

        self.assertEqual(client.models.generate_content.call_count, 5)

    def test_cache_does_not_depend_on_selected_key(self):
        first = PromoGenerator(GenerationConfig(key_name="TEST_KEY_A"))
        second = PromoGenerator(GenerationConfig(key_name="TEST_KEY_B"))
        self.assertEqual(
            first._run_directory(self.window, "same prompt"),
            second._run_directory(self.window, "same prompt"),
        )

    @patch.object(PromoGenerator, "generate_with_keys")
    def test_batch_stops_when_all_daily_budgets_are_exhausted(self, generate):
        generate.side_effect = DailyBudgetExceeded("All budgets exhausted")
        with tempfile.TemporaryDirectory() as directory:
            generator = PromoGenerator(GenerationConfig(), directory)
            with contextlib.redirect_stdout(io.StringIO()):
                report = generator.generate_windows(
                    self.preprocessor.create_windows(), self.episodes, ["TEST_KEY"]
                )
            self.assertEqual(len(report["windows"]), 1)
            self.assertEqual(report["windows"][0]["status"], "failed")
            self.assertTrue(Path(report["report_path"]).exists())

        self.assertEqual(generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
