from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from evaluation.inference.models import (
    Candidate,
    CheckResult,
    EffectiveInvocationDescriptor,
    EvaluationAttempt,
    InvocationPolicy,
    ProviderModelMetadataSnapshot,
)
from evaluation.inference.reporting import (
    HumanScore,
    PricingMetadata,
    generate_markdown_report,
    join_human_scores,
    write_blinded_review,
)


def _attempt(
    attempt_id: str,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    status: Literal["succeeded", "failed"] = "succeeded",
    workload: Literal["general", "research"] = "general",
) -> EvaluationAttempt:
    candidate = Candidate(
        id=f"{workload}-cloudflare-test",
        workload=workload,
        provider="cloudflare",
        model="@cf/test/model",
        adapter="test-adapter",
    )
    descriptor = EffectiveInvocationDescriptor(
        candidate=candidate,
        metadata=ProviderModelMetadataSnapshot(
            provider="cloudflare",
            model="@cf/test/model",
            retrieved_at="2025-01-01T00:00:00Z",
            metadata_source="offline_fixture",
            metadata_method="fixture",
        ),
        policy=InvocationPolicy(),
    )
    return EvaluationAttempt(
        schema_version=2,
        run_id="report-run",
        attempt_id=attempt_id,
        repository_commit="3ee0bc92d33a98ac72fedb84bb1f929df6301a26",
        dataset_name=workload,
        dataset_version="1.0.1",
        dataset_sha256="0" * 64,
        case_id=("general-json-06" if workload == "general" else "research-support-01"),
        repetition=0,
        invocation=descriptor,
        timestamp="2025-01-01T00:00:00Z",
        status=status,
        latency_ms=10.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output='{"status":"completed","count":4}',
        checks=(CheckResult("json-contract", True),),
    )


def test_blinded_review_hides_candidate_identity_and_scores_join_by_attempt_id(
    tmp_path: Path,
) -> None:
    attempt = _attempt("attempt-opaque-001")
    destination = tmp_path / "blind.jsonl"

    write_blinded_review([attempt], destination)
    review_row = json.loads(destination.read_text(encoding="utf-8"))
    score = HumanScore(
        attempt_id="attempt-opaque-001",
        dimensions={"correctness": 4, "instruction_following": 5},
    )
    joined = join_human_scores([attempt], [score])
    report = generate_markdown_report([attempt], scores=[score])

    assert review_row["attempt_id"] == attempt.attempt_id
    assert review_row["output"] == attempt.output
    assert not {
        "candidate_id",
        "provider",
        "model",
        "requested_provider",
        "requested_model",
        "observed_provider",
        "observed_model",
        "run_id",
    }.intersection(review_row)
    assert joined[attempt.attempt_id] == score
    assert "Human correctness score: 4.00/5" in report


def test_report_preserves_unknown_metrics_and_labels_partial_cost(
    tmp_path: Path,
) -> None:
    metered = _attempt("attempt-metered", input_tokens=100, output_tokens=50)
    partial = _attempt("attempt-partial", input_tokens=70, output_tokens=None)
    unknown_cost = generate_markdown_report([partial])
    priced_report = generate_markdown_report(
        [metered, partial],
        pricing=[
            PricingMetadata(
                provider="cloudflare",
                model="@cf/test/model",
                input_per_million=0.2,
                output_per_million=0.3,
                source="official test price sheet",
                published_date="2025-01-01",
            )
        ],
    )

    assert "Estimated cost: unknown (no pricing metadata)" in unknown_cost
    assert "input: 70 reported across 1/1 attempts; output: unknown" in unknown_cost
    assert "partial token coverage: 1/2 attempts" in priced_report
    assert "not actual billed cost" in priced_report
    assert "Latency: n=1, mean=10.0 ms" in unknown_cost
    assert "general v1.0.1" in unknown_cost
    assert "Repository commits:" in unknown_cost


def test_report_cost_is_unknown_without_both_provider_reported_token_counts() -> None:
    attempt = _attempt("attempt-missing-output", input_tokens=100)
    price = PricingMetadata(
        provider="cloudflare",
        model="@cf/test/model",
        input_per_million=0.2,
        output_per_million=0.3,
        source="official test price sheet",
        published_date="2025-01-01",
    )

    report = generate_markdown_report([attempt], pricing=[price])

    assert ("Estimated cost: unknown (no successful attempt reported both token counts)") in report


def test_human_score_dimensions_are_validated_against_attempt_workload() -> None:
    general = _attempt("general-score", workload="general")
    research = _attempt("research-score", workload="research")
    general_score = HumanScore(
        attempt_id=general.attempt_id,
        dimensions={
            "correctness": 4,
            "instruction_following": 4,
            "concision_usefulness": 4,
            "format_validity": 4,
        },
    )
    research_score = HumanScore(
        attempt_id=research.attempt_id,
        dimensions={
            "supported_claim_coverage": 4,
            "unsupported_claims": 4,
            "conflict_handling": 4,
            "uncertainty": 4,
            "claim_source_attribution": 4,
            "citation_fidelity": 4,
            "concision_usefulness": 4,
        },
    )

    joined = join_human_scores([general, research], [general_score, research_score])
    assert joined[general.attempt_id] == general_score
    assert joined[research.attempt_id] == research_score


def test_human_scores_allow_omitted_dimensions_but_reject_invalid_workload_dimensions() -> None:
    general = _attempt("general-score")
    omitted = HumanScore(attempt_id=general.attempt_id, dimensions={})
    assert join_human_scores([general], [omitted])[general.attempt_id] == omitted

    unknown = HumanScore(
        attempt_id=general.attempt_id,
        dimensions={"instruction_followng": 4},
    )
    other_workload = HumanScore(
        attempt_id=general.attempt_id,
        dimensions={"conflict_handling": 4},
    )
    with pytest.raises(ValueError, match="instruction_followng"):
        join_human_scores([general], [unknown])
    with pytest.raises(ValueError, match="conflict_handling"):
        join_human_scores([general], [other_workload])


@pytest.mark.parametrize("score", [0, 6, True, 3.5])
def test_human_score_values_remain_integer_scale_one_to_five(score: object) -> None:
    with pytest.raises(ValueError, match="integers from 1 to 5"):
        HumanScore(attempt_id="invalid-score", dimensions={"correctness": score})  # type: ignore[dict-item]
