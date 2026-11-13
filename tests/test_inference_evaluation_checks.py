from __future__ import annotations

import json

import pytest

from evaluation.inference.checks import evaluate_checks, validate_check_specs
from evaluation.inference.datasets import load_dataset
from evaluation.inference.models import EvaluationCase


def test_structured_output_requires_exact_json_keys_and_types() -> None:
    case = load_dataset("general").case("general-json-06")

    valid = evaluate_checks(case, '{"status":"completed","count":4}')
    invalid_markdown = evaluate_checks(case, '```json\n{"status":"completed","count":4}\n```')
    invalid_extra_key = evaluate_checks(case, '{"status":"completed","count":4,"extra":true}')
    invalid_boolean_count = evaluate_checks(case, '{"status":"completed","count":true}')

    assert all(result.passed for result in valid)
    assert not next(
        item for item in invalid_markdown if item.check_id == "exact-json-object-contract"
    ).passed
    assert not next(
        item for item in invalid_extra_key if item.check_id == "exact-json-object-contract"
    ).passed
    assert not next(
        item for item in invalid_boolean_count if item.check_id == "exact-json-object-contract"
    ).passed


def test_research_citations_are_limited_to_fixture_sources() -> None:
    case = load_dataset("research").case("research-attribution-03")

    allowed = evaluate_checks(case, "The field report measured 12 [S1]; drift was 0.4% [S2].")
    unknown = evaluate_checks(case, "The field report measured 12 [S1]; drift was 0.4% [S9].")

    assert next(item for item in allowed if item.check_id == "uses-citation-allowlist").passed
    assert not next(item for item in unknown if item.check_id == "uses-citation-allowlist").passed


def test_general_synthesis_checks_do_not_detect_semantic_absence_of_claims() -> None:
    case = load_dataset("general").case("general-synthesis-03")
    response = (
        "Northbridge library reopened in 2022 and serves 240 residents. "
        "Its rooftop solar panels supplied 65% of electricity. "
        "The source does not mention a mobile app."
    )

    results = evaluate_checks(case, response)

    assert all(result.passed for result in results)
    assert {spec["id"] for spec in case.checks} == {
        "includes-all-supplied-facts",
        "concise-summary",
    }
    assert any("Adds no claims" in item for item in case.human_rubric_items)


def test_known_research_distractor_is_prohibited_separately_from_allowlist() -> None:
    case = load_dataset("research").case("research-long-context-05")
    relevant = evaluate_checks(
        case, "The pilot ran 28 days and supplied 17% [S1]. Limits were reported [S2]."
    )
    cites_distractor = evaluate_checks(
        case, "The pilot ran 28 days and supplied 17% [S1]. The festival was relevant [S4]."
    )

    allowlist = next(item for item in case.checks if item["id"] == "citation-ids-are-allowlisted")
    assert allowlist["values"] == ["S1", "S2", "S3", "S4"]
    assert next(item for item in relevant if item.check_id == "does-not-cite-distractor").passed
    assert not next(
        item for item in cites_distractor if item.check_id == "does-not-cite-distractor"
    ).passed
    assert next(
        item for item in cites_distractor if item.check_id == "citation-ids-are-allowlisted"
    ).passed
    assert any("unrelated distractor" in item for item in case.human_rubric_items)


def test_fixture_checks_do_not_claim_deterministic_entailment_scoring() -> None:
    research = load_dataset("research")

    supported_kinds = {
        "nonempty",
        "contains_all",
        "contains_any",
        "forbids_any",
        "max_words",
        "bullet_count",
        "bullet_word_limit",
        "bullet_citation",
        "json_object",
        "json_values",
        "required_citations",
        "citation_allowlist",
    }
    for case in research.cases:
        validate_check_specs(case)
        assert all(spec["kind"] in supported_kinds for spec in case.checks)
        assert case.human_rubric_items

    unsupported = EvaluationCase(
        id="invalid",
        workload="research",
        category="test",
        instruction="Test",
        question="Question",
        sources=(),
        checks=({"id": "semantic", "kind": "entailment"},),
        human_rubric_items=(),
    )
    with pytest.raises(ValueError, match="unsupported check kind"):
        validate_check_specs(unsupported)


def test_citation_check_does_not_change_raw_candidate_text() -> None:
    text = "Unsupported claim [S9]."
    case = load_dataset("research").case("research-attribution-03")

    result = evaluate_checks(case, text)

    assert json.dumps(text) == '"Unsupported claim [S9]."'
    assert not next(item for item in result if item.check_id == "uses-citation-allowlist").passed
