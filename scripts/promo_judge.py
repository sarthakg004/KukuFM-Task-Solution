"""Score promo footage, choose alternative cuts, and produce a posting order."""

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field

from .preprocessing import Episode
from .promo_generator import GenerationConfig, GeminiJSONClient, OutputModel, save_json


CRITERIA = ("hook", "cliffhanger", "clarity", "escalation")
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


class CriterionScore(OutputModel):
    score: int = Field(ge=0, le=4, strict=True)
    evidence_ids: list[str] = Field(min_length=1)
    reason: str


class PromoAssessment(OutputModel):
    candidate_id: str
    hook: CriterionScore
    cliffhanger: CriterionScore
    clarity: CriterionScore
    escalation: CriterionScore
    hook_summary: str
    cliffhanger_question: str
    why_it_works: str
    concerns: list[str]


class AssessmentBatch(OutputModel):
    assessments: list[PromoAssessment]


class PostingChoice(OutputModel):
    candidate_id: str
    posting_rationale: str


class PostingOrder(OutputModel):
    order: list[PostingChoice]


@dataclass(frozen=True)
class JudgeConfig:
    batch_size: int = 8
    top_compare_count: int = 10
    minimum_core_score: int = 2
    weights: dict[str, float] = field(default_factory=lambda: {
        "hook": 0.30,
        "cliffhanger": 0.30,
        "clarity": 0.25,
        "escalation": 0.15,
    })

    def __post_init__(self):
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(self.top_compare_count) is not int or self.top_compare_count < 0:
            raise ValueError("top_compare_count must be a nonnegative integer")
        if type(self.minimum_core_score) is not int or not 0 <= self.minimum_core_score <= 4:
            raise ValueError("minimum_core_score must be an integer from 0 to 4")
        if set(self.weights) != set(CRITERIA):
            raise ValueError(f"weights must contain exactly {CRITERIA}")
        if any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("Weights must be finite and nonnegative")
        if sum(self.weights.values()) == 0:
            raise ValueError("At least one weight must be positive")


class PromoExcerptBuilder:
    """Extract exactly the selected intervals without generator explanations."""

    def __init__(self, episodes: list[Episode]):
        self.episodes = {episode.number: episode for episode in episodes}

    def build(self, candidate: dict) -> dict:
        segments = []
        playback_offset = 0.0

        for segment in candidate["resolved_segments"]:
            episode = self.episodes[segment["episode"]]
            start = segment["start_seconds"]
            end = segment["end_seconds"]
            sources = {entry.source_id: entry for entry in episode.entries}
            left = sources[segment["start_id"]]
            right = sources[segment["end_id"]]

            if not math.isclose(start, left.start_seconds) or not math.isclose(end, right.end_seconds):
                raise ValueError(f"Source boundaries disagree for {candidate['candidate_id']}")
            if end <= start or start < 0 or end > episode.duration_seconds:
                raise ValueError(f"Invalid segment bounds for {candidate['candidate_id']}")

            entries = []
            for entry in episode.entries:
                if entry.start_seconds >= end or entry.end_seconds <= start:
                    continue

                entries.append({
                    "source_id": entry.source_id,
                    "kind": entry.kind,
                    "label": entry.label,
                    "text": entry.text,
                    "clip_start_seconds": round(
                        playback_offset + max(entry.start_seconds, start) - start, 3
                    ),
                    "clip_end_seconds": round(
                        playback_offset + min(entry.end_seconds, end) - start, 3
                    ),
                    "partial_boundary": entry.start_seconds < start or entry.end_seconds > end,
                })

            segments.append({
                "episode": episode.number,
                "start": segment["start"],
                "end": segment["end"],
                "entries": entries,
            })
            playback_offset += end - start

        if not 20 <= round(playback_offset, 3) <= 60:
            raise ValueError(f"Invalid promo duration for {candidate['candidate_id']}")

        return {
            "candidate_id": candidate["candidate_id"],
            "duration_seconds": round(playback_offset, 3),
            "segments": segments,
        }


class PromoJudge:
    """Judge all cuts, select group winners and compare the leading shortlist."""

    def __init__(
        self,
        api_config: GenerationConfig,
        judge_config: JudgeConfig,
        output_dir: str | Path = "outputs/ranking",
        budget_path: str | Path = "outputs/request_usage.json",
    ):
        self.config = judge_config
        self.client = GeminiJSONClient(api_config, output_dir, budget_path)
        self.output_dir = Path(output_dir)

    def rank(
        self,
        deduplicated: dict,
        episodes: list[Episode],
        key_names: list[str] | None = None,
        request_delay_seconds: float = 15,
        force: bool = False,
    ) -> dict:
        if not 0 <= request_delay_seconds <= 60:
            raise ValueError("request_delay_seconds must be between 0 and 60")

        groups = self._groups(deduplicated)
        if not groups:
            raise ValueError("No valid candidates to judge")

        builder = PromoExcerptBuilder(episodes)
        candidates = [candidate for group in groups for candidate in group]
        excerpts = {candidate["candidate_id"]: builder.build(candidate) for candidate in candidates}
        assessments, calls = self._score_batches(
            groups, excerpts, key_names, request_delay_seconds, force
        )
        winners, alternatives = self._choose_cuts(groups, assessments)
        posting_choices = []
        shortlist = [winner for winner in winners if winner["recommended"]]
        shortlist = shortlist[:self.config.top_compare_count]

        if len(shortlist) > 1:
            time.sleep(request_delay_seconds)
            posting_choices, metadata = self._compare_shortlist(shortlist, excerpts, key_names, force)
            calls.append(metadata)

        ranked = self._assemble_ranking(winners, posting_choices)
        result = {
            "source_model": deduplicated["model"],
            "judge_model": self.client.config.model,
            "weights": self.config.weights,
            "minimum_core_score": self.config.minimum_core_score,
            "summary": {
                "cuts_judged": len(candidates),
                "groups_ranked": len(ranked),
                "recommended": sum(item["recommended"] for item in ranked),
                "alternative_cuts_selected": sum(
                    winner["candidate_id"] != winner["group_id"] for winner in ranked
                ),
            },
            "ranking_method": "Core-quality gate, weighted rubric, then editorial order within top shortlist",
            "candidates": ranked,
            "alternative_assessments": alternatives,
            "judge_calls": calls,
            "limitations": [
                "LLM editorial judgments, not measured advertising performance.",
                "Scores across batches may vary despite a shared rubric.",
                "Screenplay timestamps cannot establish word timing within a line.",
                "Human review is required before posting.",
            ],
        }
        signature = json.dumps(result, sort_keys=True)
        digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
        path = self.output_dir / f"ranked-{digest}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(path, result)
        self._save_markdown(result, path.with_suffix(".md"))
        result["output_path"] = str(path)
        return result

    @staticmethod
    def _groups(deduplicated: dict) -> list[list[dict]]:
        groups = {
            candidate["candidate_id"]: [copy.deepcopy(candidate)]
            for candidate in deduplicated["candidates"]
        }
        for duplicate in deduplicated["duplicates"]:
            groups[duplicate["kept_candidate_id"]].append(copy.deepcopy(duplicate["candidate"]))

        identifiers = [candidate["candidate_id"] for group in groups.values() for candidate in group]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate candidate IDs in deduplicated report")
        return list(groups.values())

    def _batches(self, groups: list[list[dict]]) -> list[list[dict]]:
        batches = []
        current = []
        for group in groups:
            if len(group) > self.config.batch_size:
                raise ValueError("A duplicate group exceeds judge batch_size; increase it")
            if current and len(current) + len(group) > self.config.batch_size:
                batches.append(current)
                current = []
            current.extend(group)
        if current:
            batches.append(current)
        return batches

    def _score_batches(self, groups, excerpts, key_names, delay, force):
        template = (PROMPT_DIR / "judge_prompt.txt").read_text()
        assessments = {}
        calls = []
        batches = self._batches(groups)

        for index, batch in enumerate(batches):
            if index:
                time.sleep(delay)
            items = [excerpts[candidate["candidate_id"]] for candidate in batch]
            prompt = template.format(candidates=json.dumps(items, ensure_ascii=False))
            print(f"Judging batch {index + 1}/{len(batches)}: {len(items)} cuts", flush=True)
            reply = self.client.complete(
                prompt=prompt,
                response_model=AssessmentBatch,
                key_names=key_names,
                job_name=f"scores-{index + 1:03d}",
                force=force,
                validate=lambda response: self._validate_assessments(response, items),
            )
            for assessment in reply["data"]["assessments"]:
                assessments[assessment["candidate_id"]] = assessment
            calls.append({key: value for key, value in reply.items() if key != "data"})

        return assessments, calls

    @staticmethod
    def _validate_assessments(response: AssessmentBatch, excerpts: list[dict]) -> None:
        expected = {item["candidate_id"]: item for item in excerpts}
        returned = [item.candidate_id for item in response.assessments]
        if len(returned) != len(set(returned)) or set(returned) != set(expected):
            raise ValueError("Judge must score each requested candidate exactly once; no extras")

        for assessment in response.assessments:
            excerpt = expected[assessment.candidate_id]
            entries = {
                entry["source_id"]: entry
                for segment in excerpt["segments"]
                for entry in segment["entries"]
            }
            for criterion in CRITERIA:
                score = getattr(assessment, criterion)
                if not set(score.evidence_ids) <= set(entries):
                    raise ValueError(f"Judge cited footage outside {assessment.candidate_id}")

                evidence = [entries[source_id] for source_id in score.evidence_ids]
                if criterion == "hook" and score.score > 0:
                    if not any(entry["clip_start_seconds"] < 3 for entry in evidence):
                        raise ValueError(f"Hook evidence is outside first 3s: {assessment.candidate_id}")
                if criterion == "cliffhanger" and score.score >= 3:
                    if not any(entry["clip_end_seconds"] >= excerpt["duration_seconds"] - 5 for entry in evidence):
                        raise ValueError(f"Cliffhanger evidence is not near the ending: {assessment.candidate_id}")

    def _score(self, assessment: dict) -> float:
        total_weight = sum(self.config.weights.values())
        weighted = sum(
            assessment[name]["score"] * weight
            for name, weight in self.config.weights.items()
        )
        return round(100 * weighted / (4 * total_weight), 2)

    def _choose_cuts(self, groups, assessments):
        winners = []
        alternatives = []
        for group in groups:
            scored = []
            for candidate in group:
                assessment = assessments[candidate["candidate_id"]]
                item = {
                    **candidate,
                    "group_id": group[0]["candidate_id"],
                    "assessment": assessment,
                    "score": self._score(assessment),
                    "recommended": all(
                        assessment[name]["score"] >= self.config.minimum_core_score
                        for name in ("hook", "cliffhanger", "clarity")
                    ),
                }
                scored.append(item)
            scored.sort(key=self._sort_key)
            winners.append(scored[0])
            alternatives.extend(scored[1:])

        winners.sort(key=self._sort_key)
        return winners, alternatives

    @staticmethod
    def _sort_key(item):
        assessment = item["assessment"]
        return (
            not item["recommended"],
            -item["score"],
            -assessment["cliffhanger"]["score"],
            -assessment["hook"]["score"],
            item["candidate_id"],
        )

    def _compare_shortlist(self, shortlist, excerpts, key_names, force):
        items = [{
            **excerpts[candidate["candidate_id"]],
            "score": candidate["score"],
            "assessment": candidate["assessment"],
        } for candidate in shortlist]
        prompt = (PROMPT_DIR / "posting_prompt.txt").read_text().format(
            candidates=json.dumps(items, ensure_ascii=False)
        )
        expected = {item["candidate_id"] for item in items}

        def validate(response):
            returned = [item.candidate_id for item in response.order]
            if len(returned) != len(expected) or set(returned) != expected:
                raise ValueError("Posting order must contain every shortlisted ID exactly once")

        print(f"Comparing top {len(shortlist)} promos for posting order", flush=True)
        reply = self.client.complete(
            prompt, PostingOrder, key_names, "posting-order", force, validate
        )
        metadata = {key: value for key, value in reply.items() if key != "data"}
        return reply["data"]["order"], metadata

    def _assemble_ranking(self, winners, choices):
        lookup = {item["candidate_id"]: item for item in winners}
        selected_ids = [item["candidate_id"] for item in choices]
        reasons = {item["candidate_id"]: item["posting_rationale"] for item in choices}
        score_ranks = {item["candidate_id"]: index for index, item in enumerate(winners, 1)}
        order = selected_ids + [item["candidate_id"] for item in winners if item["candidate_id"] not in selected_ids]
        result = []

        for rank, candidate_id in enumerate(order, 1):
            item = copy.deepcopy(lookup[candidate_id])
            assessment = item["assessment"]
            item["generator_explanations"] = {
                name: item[name] for name in ("hook", "cliffhanger", "why_it_works")
            }
            item.update(
                rank=rank,
                score_rank=score_ranks[candidate_id],
                hook=assessment["hook_summary"],
                cliffhanger=assessment["cliffhanger_question"],
                why_it_works=assessment["why_it_works"],
            )
            default_reason = (
                f"Weighted score {item['score']}/100; meets the core-quality thresholds."
                if item["recommended"] else
                "Below the minimum hook, cliffhanger or clarity threshold; review before posting."
            )
            item["posting_rationale"] = reasons.get(candidate_id, default_reason)
            result.append(item)

        return result

    @staticmethod
    def _save_markdown(result: dict, path: Path) -> None:
        lines = ["# Ranked promos", "", "Editorial judgments; human review required.", ""]
        for candidate in result["candidates"]:
            lines.extend([
                f"## {candidate['rank']}. {candidate['title']}",
                "",
                f"Score: {candidate['score']}/100 | Recommended: {candidate['recommended']}",
                "",
            ])
            for index, segment in enumerate(candidate["resolved_segments"], 1):
                lines.append(
                    f"{index}. Episode {segment['episode']}: {segment['start']}–{segment['end']}"
                )
            lines.extend([
                "",
                f"**Hook:** {candidate['hook']}",
                "",
                f"**Cliffhanger:** {candidate['cliffhanger']}",
                "",
                f"**Why it works:** {candidate['why_it_works']}",
                "",
                f"**Posting rationale:** {candidate['posting_rationale']}",
                "",
            ])
        path.write_text("\n".join(lines))


def display_ranked_promos(result: dict, limit: int = 5) -> None:
    """Show the required assignment fields and the judge's posting rationale."""
    print("Summary:", result["summary"])
    print("Saved to:", result["output_path"])

    for candidate in result["candidates"][:limit]:
        print(f"\n#{candidate['rank']} {candidate['title']} | {candidate['score']}/100")
        print("Recommended:", candidate["recommended"])
        for segment in candidate["resolved_segments"]:
            print(f"  Episode {segment['episode']}: {segment['start']}–{segment['end']}")
        print("Hook:", candidate["hook"])
        print("Cliffhanger:", candidate["cliffhanger"])
        print("Why it works:", candidate["why_it_works"])
        print("Post here because:", candidate["posting_rationale"])
        print("Concerns:", candidate["assessment"]["concerns"])
