"""Blinded human review artifacts and provider-neutral experiment reports."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from evaluation.inference.datasets import canonical_json
from evaluation.inference.models import EvaluationAttempt, Workload

_RUBRIC_DIMENSIONS: dict[Workload, frozenset[str]] = {
    "general": frozenset(
        {
            "correctness",
            "instruction_following",
            "concision_usefulness",
            "format_validity",
        }
    ),
    "research": frozenset(
        {
            "supported_claim_coverage",
            "unsupported_claims",
            "conflict_handling",
            "uncertainty",
            "claim_source_attribution",
            "citation_fidelity",
            "concision_usefulness",
        }
    ),
}


@dataclass(frozen=True)
class HumanScore:
    attempt_id: str
    dimensions: dict[str, int]
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, str) or not self.attempt_id.strip():
            raise ValueError("attempt_id must be non-empty")
        for dimension, score in self.dimensions.items():
            if not isinstance(dimension, str) or not dimension.strip():
                raise ValueError("human score dimension names must be non-empty strings")
            if type(score) is not int or not 1 <= score <= 5:
                raise ValueError("human scores must be integers from 1 to 5")


@dataclass(frozen=True)
class PricingMetadata:
    """Optional experiment-only list-rate metadata; never treated as actual billing."""

    provider: str
    model: str
    input_per_million: float
    output_per_million: float
    source: str
    published_date: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.provider, self.model, self.source, self.published_date)
        ):
            raise ValueError("pricing identity, source, and published_date are required")
        for rate in (self.input_per_million, self.output_per_million):
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(rate)
                or rate < 0
            ):
                raise ValueError("pricing rates must be finite non-negative numbers")


def write_blinded_review(attempts: list[EvaluationAttempt], destination: Path) -> None:
    """Write review rows without candidate/provider/model identity."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as review_file:
        for attempt in attempts:
            review_row = {
                "attempt_id": attempt.attempt_id,
                "case_id": attempt.case_id,
                "workload": attempt.workload,
                "status": attempt.status,
                "output": attempt.output,
                "checks": [
                    {"check_id": check.check_id, "passed": check.passed} for check in attempt.checks
                ],
            }
            review_file.write(canonical_json(review_row))
            review_file.write("\n")


def join_human_scores(
    attempts: list[EvaluationAttempt], scores: list[HumanScore]
) -> dict[str, HumanScore]:
    """Join blinded score sheets through attempt ids, rejecting ambiguous rows."""
    attempts_by_id = {attempt.attempt_id: attempt for attempt in attempts}
    joined: dict[str, HumanScore] = {}
    for score in scores:
        attempt = attempts_by_id.get(score.attempt_id)
        if attempt is None:
            raise ValueError(f"Unknown attempt_id in human scores: {score.attempt_id}")
        if score.attempt_id in joined:
            raise ValueError(f"Duplicate human score for attempt_id: {score.attempt_id}")
        allowed = _RUBRIC_DIMENSIONS[attempt.workload]
        invalid = set(score.dimensions) - allowed
        if invalid:
            raise ValueError(f"Invalid {attempt.workload} score dimensions: {sorted(invalid)}")
        joined[score.attempt_id] = score
    return joined


def _format_latency(values: list[float]) -> str:
    if not values:
        return "unknown (no measured values)"
    return (
        f"n={len(values)}, mean={statistics.fmean(values):.1f} ms, "
        f"median={statistics.median(values):.1f} ms"
    )


def _format_usage(attempts: list[EvaluationAttempt]) -> str:
    successful = [item for item in attempts if item.status == "succeeded"]
    input_values = [item.input_tokens for item in successful if item.input_tokens is not None]
    output_values = [item.output_tokens for item in successful if item.output_tokens is not None]
    input_text = (
        f"{sum(input_values)} reported across {len(input_values)}/{len(attempts)} attempts"
        if input_values
        else "unknown"
    )
    output_text = (
        f"{sum(output_values)} reported across {len(output_values)}/{len(attempts)} attempts"
        if output_values
        else "unknown"
    )
    return f"input: {input_text}; output: {output_text}"


def _format_cost(
    attempts: list[EvaluationAttempt],
    pricing: PricingMetadata | None,
) -> str:
    if pricing is None:
        return "unknown (no pricing metadata)"
    metered = [
        item
        for item in attempts
        if item.status == "succeeded"
        and item.input_tokens is not None
        and item.output_tokens is not None
    ]
    if not metered:
        return "unknown (no successful attempt reported both token counts)"
    estimated = sum(
        (item.input_tokens or 0) * pricing.input_per_million / 1_000_000
        + (item.output_tokens or 0) * pricing.output_per_million / 1_000_000
        for item in metered
    )
    coverage = f"{len(metered)}/{len(attempts)} attempts"
    completeness: Literal["complete", "partial"] = (
        "complete" if len(metered) == len(attempts) else "partial"
    )
    return (
        f"${estimated:.6f} estimated list-rate ({completeness} token coverage: {coverage}; "
        f"not actual billed cost; source: {pricing.source}, {pricing.published_date})"
    )


def generate_markdown_report(
    attempts: list[EvaluationAttempt],
    *,
    scores: list[HumanScore] | None = None,
    pricing: list[PricingMetadata] | None = None,
) -> str:
    """Summarize observed attempts without imputing missing metrics."""
    score_by_id = join_human_scores(attempts, scores or [])
    price_by_identity = {(item.provider.lower(), item.model): item for item in pricing or []}
    groups: dict[tuple[str, str, str], list[EvaluationAttempt]] = defaultdict(list)
    for attempt in attempts:
        groups[(attempt.workload, attempt.requested_provider, attempt.requested_model)].append(
            attempt
        )

    lines = [
        "# Inference Evaluation Report",
        "",
        "List-rate costs are estimates from supplied pricing metadata, not billed amounts.",
        "Missing provider-reported usage remains unknown; no values are imputed.",
        "",
    ]
    for workload, provider, model in sorted(groups):
        group = groups[(workload, provider, model)]
        successful = [item for item in group if item.status == "succeeded"]
        checks = [check for item in successful for check in item.checks]
        checks_by_id: dict[str, list[bool]] = defaultdict(list)
        for check in checks:
            checks_by_id[check.check_id].append(check.passed)
        measured_latency = [item.latency_ms for item in group if item.latency_ms >= 0]
        datasets = sorted(
            {(item.dataset_name, item.dataset_version, item.dataset_sha256) for item in group}
        )
        commits = sorted({item.repository_commit for item in group})
        truncation_known = [item.truncated for item in group if item.truncated is not None]
        adapter_configurations = sorted(
            {
                f"{item.invocation.candidate.adapter}, "
                f"output_limit={item.invocation.effective_output_limit_mode}:"
                f"{item.invocation.effective_output_limit}, "
                f"temperature={item.invocation.effective_temperature_mode}:"
                f"{item.invocation.effective_temperature}, "
                f"metadata={item.invocation.metadata.metadata_source}/"
                f"{item.invocation.metadata.metadata_method}@"
                f"{item.invocation.metadata.retrieved_at}"
                for item in group
            }
        )
        lines.extend(
            [
                f"## {workload}: {provider} / `{model}`",
                "",
                "- Adapter configurations: " + "; ".join(adapter_configurations),
                "- Datasets: "
                + "; ".join(
                    f"{name} v{version} (sha256 `{sha256}`)" for name, version, sha256 in datasets
                ),
                "- Repository commits: " + ", ".join(commits),
                f"- Scheduled attempts: {len(group)}",
                f"- Successes / failures: {len(successful)} / {len(group) - len(successful)}",
                (
                    f"- Truncation reported: {sum(value is True for value in truncation_known)}"
                    f" true / {len(truncation_known)} known"
                    if truncation_known
                    else "- Truncation reported: unknown"
                ),
                (
                    f"- Deterministic checks: {sum(item.passed for item in checks)}"
                    f"/{len(checks)} passed across successful attempts"
                    if checks
                    else "- Deterministic checks: unavailable"
                ),
                *(
                    [
                        "- Check outcomes: "
                        + "; ".join(
                            f"{check_id} {sum(values)}/{len(values)}"
                            for check_id, values in sorted(checks_by_id.items())
                        )
                    ]
                    if checks_by_id
                    else []
                ),
                f"- Latency: {_format_latency(measured_latency)}",
                f"- Reported token usage: {_format_usage(group)}",
                (
                    "- Estimated cost: "
                    + _format_cost(
                        group,
                        price_by_identity.get((provider, model)),
                    )
                ),
            ]
        )
        group_scores = [
            score_by_id[item.attempt_id] for item in group if item.attempt_id in score_by_id
        ]
        if group_scores:
            dimensions = sorted(
                {dimension for score in group_scores for dimension in score.dimensions}
            )
            for dimension in dimensions:
                values = [
                    score.dimensions[dimension]
                    for score in group_scores
                    if dimension in score.dimensions
                ]
                if values:
                    lines.append(
                        f"- Human {dimension} score: {statistics.fmean(values):.2f}/5 "
                        f"({len(values)} scores)"
                    )
        else:
            lines.append("- Human scores: unavailable")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def read_attempts(path: Path) -> list[EvaluationAttempt]:
    """Read result JSONL records written by the Stage 0 runner."""
    from evaluation.inference.models import attempt_from_dict

    attempts: list[EvaluationAttempt] = []
    with path.open(encoding="utf-8") as result_file:
        for line_number, line in enumerate(result_file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Result line {line_number} must contain a JSON object")
            attempts.append(attempt_from_dict(value))
    return attempts
