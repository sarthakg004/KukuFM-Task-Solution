"""Generate, validate and save promo candidates for one screenplay window."""

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict, ValidationError

from .preprocessing import Episode, ScreenplayWindow
from .api_keys import discover_gemini_keys
from .request_budget import DailyBudgetExceeded, DailyRequestBudget


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "candidate_prompt.txt"
RETRYABLE_CODES = {429, 500, 502, 503, 504}
FALLBACK_CODES = RETRYABLE_CODES | {404}
ERROR_DESCRIPTIONS = {
    404: "Model not found or unavailable to this project",
    429: "Rate limit or quota exhausted",
    503: "Service temporarily overloaded or unavailable",
    504: "Request did not finish within the server deadline",
}


def save_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


class OutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Segment(OutputModel):
    start_id: str
    end_id: str


class Candidate(OutputModel):
    title: str
    segments: list[Segment]
    hook: str
    cliffhanger: str
    standalone_context: str
    why_it_works: str
    concerns: str


class CandidateBatch(OutputModel):
    candidates: list[Candidate]


class GeminiConnectionError(RuntimeError):
    """A connection failure that cannot be fixed by changing API keys."""


@dataclass(frozen=True)
class GenerationConfig:
    """Model settings and bounded retry policy. Timeout is in seconds."""

    model: str = "gemini-3.6-flash"
    fallback_model: str | None = None
    key_name: str = "GEMINI_API_KEY1"
    max_candidates: int = 8
    max_output_tokens: int = 24000
    thinking_level: str = "low"
    timeout_seconds: int = 180
    retries: int = 2
    retry_delay_seconds: float = 5
    daily_requests_per_key: int = 5

    def __post_init__(self) -> None:
        positive_integers = (
            self.max_candidates,
            self.max_output_tokens,
            self.timeout_seconds,
            self.daily_requests_per_key,
        )
        if any(type(value) is not int or value < 1 for value in positive_integers):
            raise ValueError("Candidate, output-token and timeout limits must be positive integers")

        if type(self.retries) is not int or self.retries < 0:
            raise ValueError("retries must be a nonnegative integer")

        if not 0 <= self.retry_delay_seconds <= 30:
            raise ValueError("retry_delay_seconds must be between 0 and 30")

        if self.thinking_level not in {"low", "medium", "high"}:
            raise ValueError("thinking_level must be low, medium or high")

    @property
    def models(self) -> list[str]:
        names = [self.model, self.fallback_model]
        return list(dict.fromkeys(name for name in names if name))


class CandidateValidator:
    """Resolve source IDs and check timings, without judging creative quality."""

    def __init__(self, episodes: list[Episode], window: ScreenplayWindow):
        self.sources = {
            entry.source_id: (episode, entry)
            for episode in episodes
            if episode.number in window.episode_numbers
            for entry in episode.entries
        }

    def validate(self, batch: CandidateBatch) -> list[dict]:
        return [self._validate_candidate(candidate) for candidate in batch.candidates]

    def _validate_candidate(self, candidate: Candidate) -> dict:
        issues = []
        resolved = []

        if not candidate.segments:
            issues.append("No segments")

        for segment in candidate.segments:
            try:
                resolved.append(self._resolve_segment(segment))
            except ValueError as exc:
                issues.append(str(exc))

        if any(self._overlap(left, right) for left, right in combinations(resolved, 2)):
            issues.append("Segments reuse overlapping footage")

        duration = round(sum(segment["duration_seconds"] for segment in resolved), 3)
        if not 20 <= duration <= 60:
            issues.append(f"Duration {duration}s outside 20–60s")

        return {
            **candidate.model_dump(),
            "resolved_segments": resolved,
            "duration_seconds": duration,
            "valid": not issues,
            "issues": issues,
        }

    def _resolve_segment(self, segment: Segment) -> dict:
        if segment.start_id not in self.sources or segment.end_id not in self.sources:
            raise ValueError(
                f"Unknown or out-of-window ID: {segment.start_id}, {segment.end_id}"
            )

        episode, start = self.sources[segment.start_id]
        end_episode, end = self.sources[segment.end_id]

        if episode.number != end_episode.number:
            raise ValueError("A segment crosses episode boundaries")

        if end.end_seconds <= start.start_seconds:
            raise ValueError("Nonpositive segment duration")

        if start.start_seconds < 0 or end.end_seconds > episode.duration_seconds:
            raise ValueError("Segment outside episode bounds")

        for boundary in (start, end):
            if any(boundary.source_id in warning for warning in episode.warnings):
                raise ValueError(f"Source timing warning: {boundary.source_id}")

        return {
            **segment.model_dump(),
            "episode": episode.number,
            "start": start.start,
            "end": end.end,
            "start_seconds": start.start_seconds,
            "end_seconds": end.end_seconds,
            "duration_seconds": round(end.end_seconds - start.start_seconds, 3),
            "start_text": start.text,
            "end_text": end.text,
        }

    @staticmethod
    def _overlap(left: dict, right: dict) -> bool:
        same_episode = left["episode"] == right["episode"]
        latest_start = max(left["start_seconds"], right["start_seconds"])
        earliest_end = min(left["end_seconds"], right["end_seconds"])
        return same_episode and latest_start < earliest_end


class GeminiJSONClient:
    """Shared Gemini transport, quota accounting and structured-response parsing."""

    def __init__(
        self,
        config: GenerationConfig,
        output_dir: str | Path = "outputs/candidates",
        budget_path: str | Path = "outputs/request_usage.json",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.budget = DailyRequestBudget(budget_path, config.daily_requests_per_key)

    def complete(
        self,
        prompt: str,
        response_model: type[BaseModel],
        key_names: list[str] | None = None,
        job_name: str = "request",
        force: bool = False,
        validate: Callable[[BaseModel], None] | None = None,
    ) -> dict:
        """Return cached structured JSON or request it using available keys."""
        settings = asdict(self.config)
        settings.pop("key_name")
        settings.pop("daily_requests_per_key")
        signature = json.dumps(
            [prompt, response_model.model_json_schema(), settings], sort_keys=True
        )
        digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
        directory = self.output_dir / f"{job_name}-{digest}"
        cache_path = directory / "result.json"

        if cache_path.exists() and not force:
            cached = json.loads(cache_path.read_text())
            if validate:
                validate(response_model.model_validate(cached["data"]))
            return cached

        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prompt.txt").write_text(prompt)
        available = discover_gemini_keys(key_names)
        available.sort(key=lambda name: self.budget.used(os.environ[name]))
        if not available:
            raise ValueError(
                "No Gemini keys found. Set GEMINI_API_KEY or numbered keys such as "
                "GEMINI_API_KEY1 in .env or the environment."
            )

        failures = []
        exhausted = 0
        for index, name in enumerate(available):
            if index:
                time.sleep(self.config.retry_delay_seconds)

            client_config = replace(self.config, key_name=name)
            transport = GeminiJSONClient(client_config, self.output_dir, self.budget.path)
            attempt_dir = directory / f"key-{index + 1}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            http_options = types.HttpOptions(
                timeout=self.config.timeout_seconds * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            )

            try:
                with genai.Client(api_key=os.environ[name], http_options=http_options) as client:
                    response, model, attempts = transport._request(
                        client, prompt, attempt_dir, os.environ[name], response_model
                    )
                parsed = transport._parse_response(response, attempt_dir, response_model)
                if validate:
                    validate(parsed)
                usage = response.usage_metadata
                result = {
                    "data": parsed.model_dump(mode="json"),
                    "model_used": model,
                    "key_name": name,
                    "usage": usage.model_dump(mode="json") if usage else {},
                    "attempts": attempts,
                    "output_directory": str(attempt_dir),
                }
                save_json(cache_path, result)
                return result
            except GeminiConnectionError:
                raise
            except RuntimeError as exc:
                exhausted += isinstance(exc, DailyBudgetExceeded)
                failures.append({"key_name": name, "error": str(exc)})
                save_json(directory / "failures.json", failures)

        if exhausted == len(available):
            raise DailyBudgetExceeded("All local daily budgets are exhausted; progress is cached.")

        raise RuntimeError(f"No key completed {job_name}; inspect {directory / 'failures.json'}")

    def _request(
        self, client, prompt: str, directory: Path, api_key: str,
        response_model: type[BaseModel] = CandidateBatch,
    ) -> tuple:
        attempts = []
        request_config = types.GenerateContentConfig(
            max_output_tokens=self.config.max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=response_model.model_json_schema(),
            thinking_config=types.ThinkingConfig(
                thinking_level=self.config.thinking_level,
            ),
        )

        for model in self.config.models:
            for retry in range(self.config.retries + 1):
                self.budget.reserve(api_key, self.config.key_name)
                attempt = {
                    "model": model,
                    "key_name": self.config.key_name,
                    "attempt": retry + 1,
                }
                print(f"Generating with {model} (attempt {retry + 1})…", flush=True)

                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=request_config,
                    )
                    attempt["status"] = "success"
                    attempts.append(attempt)
                    save_json(directory / "attempts.json", attempts)
                    return response, model, attempts

                except errors.APIError as exc:
                    # Exception messages and URLs can contain credentials.
                    code = exc.code
                    reason = ERROR_DESCRIPTIONS.get(code, "Gemini API error")
                    attempt.update(status_code=code, reason=reason)
                    attempts.append(attempt)
                    save_json(directory / "attempts.json", attempts)
                    print(f"HTTP {code}: {reason}.", flush=True)

                    if code not in FALLBACK_CODES:
                        raise RuntimeError(f"Gemini request failed: HTTP {code}") from None

                    if code not in RETRYABLE_CODES or retry == self.config.retries:
                        break

                    if code == 429:
                        # Try another configured model/project instead of spending
                        # more attempts on a key that has just hit a quota limit.
                        break

                    self._wait_before_retry(retry)

                except httpx.TimeoutException:
                    attempt["reason"] = "Client request timed out"
                    attempts.append(attempt)
                    save_json(directory / "attempts.json", attempts)

                    if retry == self.config.retries:
                        break

                    self._wait_before_retry(retry)

                except httpx.TransportError:
                    attempt["reason"] = "Network connection failed"
                    attempts.append(attempt)
                    save_json(directory / "attempts.json", attempts)
                    raise GeminiConnectionError(
                        "Cannot connect to Gemini; check network/DNS access."
                    ) from None

            print(f"Attempts exhausted for {model}.", flush=True)

        last_attempt = attempts[-1]
        reason = last_attempt.get("reason", "Unknown error")
        status = last_attempt.get("status_code")
        status_text = f"HTTP {status}: " if status else ""
        raise RuntimeError(
            f"No model completed the request. Last failure: {status_text}{reason}. "
            "Try a smaller episode window or retry when the service is available. "
            f"Details: {directory / 'attempts.json'}"
        )

    def _wait_before_retry(self, retry: int) -> None:
        delay = min(self.config.retry_delay_seconds * (2 ** retry), 60)
        print(f"Retrying in {delay:g}s…", flush=True)
        time.sleep(delay)

    def _parse_response(
        self, response, directory: Path,
        response_model: type[BaseModel] = CandidateBatch,
    ) -> BaseModel:
        (directory / "response.json").write_text(
            response.model_dump_json(indent=2, exclude_none=True)
        )
        text = response.text
        if text:
            (directory / "response.txt").write_text(text)

        candidates = response.candidates or []
        reason = candidates[0].finish_reason if candidates else None
        finish_reason = getattr(reason, "value", reason)

        if finish_reason == "MAX_TOKENS":
            raise RuntimeError(
                "Gemini reached its token limit before completing the response. "
                "Thinking and answer tokens share this limit. Use thinking_level='low', "
                "increase max_output_tokens, or reduce max_candidates/window size. "
                f"Partial response saved in {directory}."
            )

        if finish_reason not in (None, "STOP"):
            raise RuntimeError(
                f"Gemini stopped with {finish_reason}; inspect {directory / 'response.json'}"
            )

        if not text:
            raise RuntimeError(f"No text returned; inspect {directory / 'response.json'}")

        try:
            batch = response_model.model_validate_json(text)
        except ValidationError:
            raise RuntimeError(
                "Gemini returned invalid or incomplete candidate JSON. "
                f"Raw response saved in {directory}; no candidates were accepted."
            ) from None

        if isinstance(batch, CandidateBatch) and len(batch.candidates) > self.config.max_candidates:
            raise ValueError(f"Model exceeded candidate ceiling; inspect {directory}")

        return batch



class PromoGenerator(GeminiJSONClient):
    """Generate source-linked promos and save per-window progress."""

    def generate_with_keys(
        self,
        window: ScreenplayWindow,
        episodes: list[Episode],
        key_names: list[str] | None = None,
    ) -> dict:
        """Try least-used keys first; stop on success or a network failure.

        Keys may share project quotas. This is a bounded availability trial,
        not a guarantee of separate capacity for each key.
        """
        available = discover_gemini_keys(key_names)
        if not available:
            raise ValueError(
                "No Gemini keys found. Set GEMINI_API_KEY or numbered keys such as "
                "GEMINI_API_KEY1 in .env or the environment."
            )

        available.sort(key=lambda name: self.budget.used(os.environ[name]))

        failures = []
        exhausted_keys = 0
        for index, key_name in enumerate(available):
            if index:
                time.sleep(self.config.retry_delay_seconds)

            print(f"Trying {key_name}…", flush=True)
            config = replace(self.config, key_name=key_name)
            generator = PromoGenerator(config, self.output_dir, self.budget.path)

            try:
                return generator.generate(window, episodes)
            except GeminiConnectionError:
                raise
            except RuntimeError as exc:
                if isinstance(exc, DailyBudgetExceeded):
                    exhausted_keys += 1
                failures.append(f"{key_name}: {exc}")
                print(str(exc), flush=True)

        if exhausted_keys == len(available):
            raise DailyBudgetExceeded(
                "All configured keys have used their local daily request budget. "
                "Successful windows remain cached; rerun after midnight Pacific."
            )

        raise RuntimeError(
            "No configured key/model combination completed the trial.\n"
            + "\n".join(failures)
        )

    def generate_windows(
        self,
        windows: list[ScreenplayWindow],
        episodes: list[Episode],
        key_names: list[str] | None = None,
        request_delay_seconds: float = 15,
    ) -> dict:
        """Process all windows, saving progress and retaining individual failures.

        Successful windows use the existing response cache on rerun. API failures
        do not discard other windows; connection failures stop the run immediately.
        """
        if not 0 <= request_delay_seconds <= 60:
            raise ValueError("request_delay_seconds must be between 0 and 60")

        if key_names is None:
            key_names = discover_gemini_keys(key_names)
        else:
            key_names = list(dict.fromkeys(key_names))
        if not key_names:
            raise ValueError("No Gemini API keys found in .env or the environment")
        print(f"Detected {len(key_names)} configured Gemini keys.", flush=True)

        signature = json.dumps({
            "config": asdict(self.config),
            "keys": key_names,
            "inputs": [window.text for window in windows],
        }, sort_keys=True)
        digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
        report_path = self.output_dir / f"batch-{digest}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "model": self.config.model,
            "total_windows": len(windows),
            "windows": [],
            "report_path": str(report_path),
        }
        for index, window in enumerate(windows):
            if index:
                time.sleep(request_delay_seconds)

            print(f"\nWindow {index + 1}/{len(windows)}: {window.episode_numbers}", flush=True)
            record = {"window_id": window.window_id, "episodes": window.episode_numbers}

            try:
                result = self.generate_with_keys(window, episodes, key_names)
                record.update(status="success", result=result)
            except RuntimeError as exc:
                record.update(status="failed", error=str(exc))
                report["windows"].append(record)
                save_json(report_path, report)

                if isinstance(exc, GeminiConnectionError):
                    raise

                if isinstance(exc, DailyBudgetExceeded):
                    print("Daily budget exhausted; stopping with progress saved.", flush=True)
                    break

                print(f"{window.window_id} failed; moving to the next window.", flush=True)
                continue

            report["windows"].append(record)
            save_json(report_path, report)

        return report

    def generate(
        self,
        window: ScreenplayWindow,
        episodes: list[Episode],
        force: bool = False,
    ) -> dict:
        prompt = PROMPT_PATH.read_text().format(
            max_candidates=self.config.max_candidates,
            screenplay=window.text,
        )
        directory = self._run_directory(window, prompt)
        result_path = directory / "candidates.json"

        if result_path.exists() and not force:
            return json.loads(result_path.read_text())

        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prompt.txt").write_text(prompt)

        # Credentials are only needed on a cache miss and are never saved.
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        api_key = os.getenv(self.config.key_name)
        if not api_key:
            raise ValueError(f"Missing environment variable: {self.config.key_name}")

        http_options = types.HttpOptions(
            timeout=self.config.timeout_seconds * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        with genai.Client(api_key=api_key, http_options=http_options) as client:
            response, model, attempts = self._request(client, prompt, directory, api_key)

        batch = self._parse_response(response, directory)
        usage = response.usage_metadata
        result = {
            "window_id": window.window_id,
            "episodes": window.episode_numbers,
            "requested_model": self.config.model,
            "model_used": model,
            "key_name": self.config.key_name,
            "usage": usage.model_dump(mode="json") if usage else {},
            "attempts": attempts,
            "output_directory": str(directory),
            "candidates": CandidateValidator(episodes, window).validate(batch),
        }
        save_json(result_path, result)
        return result

    def _run_directory(self, window: ScreenplayWindow, prompt: str) -> Path:
        settings = asdict(self.config)
        # Credential selection and budget changes do not alter generated content.
        settings.pop("key_name")
        settings.pop("daily_requests_per_key")
        signature = json.dumps({
            "version": 2,
            "prompt": prompt,
            "schema": CandidateBatch.model_json_schema(),
            "config": settings,
        }, sort_keys=True)
        digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
        return self.output_dir / f"{window.window_id}-{digest}"



def display_candidates(result: dict) -> None:
    """Print a compact review without putting display logic in the notebook."""
    print("Model:", result["model_used"])
    print("Token usage:", result["usage"])
    print("Saved to:", result["output_directory"])

    for number, candidate in enumerate(result["candidates"], 1):
        print(f"\n{number}. {candidate['title']}")
        print(f"Duration: {candidate['duration_seconds']}s | Valid: {candidate['valid']}")

        for segment in candidate["resolved_segments"]:
            print(f"  Episode {segment['episode']}: {segment['start']}–{segment['end']}")

        fields = [
            ("Hook", "hook"),
            ("Cliffhanger", "cliffhanger"),
            ("Context", "standalone_context"),
            ("Why it works", "why_it_works"),
            ("Concerns", "concerns"),
            ("Validation issues", "issues"),
        ]
        for label, field in fields:
            print(f"{label}: {candidate[field]}")
