"""Tests for routing evaluation dataset, metrics, and benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from orchestrator.domain.capability import AgentCapability

from evaluation.routing.benchmark import (
    DEFAULT_DATASET_PATH,
    VALIDATION_DATASET_PATH,
    _compute_metrics,
    _compute_percentile,
    build_parser,
    evaluate_correctness,
    format_benchmark_report,
    load_dataset,
    main,
    measure_latency,
    normalize_capability,
    run_benchmark,
)
from evaluation.routing.models import (
    BenchmarkResult,
    CaseResult,
    ConfusionMatrix,
    EvaluationCase,
    LatencyStats,
)

REQUIRED_CATEGORIES = {
    "clear_general",
    "clear_research",
    "ambiguous_borderline",
    "freshness_current_info",
    "explicit_sources_evidence",
    "explanatory_general",
    "code_transformation",
    "multi_clause_mixed",
    "research_keywords_negative",
}


class TestDatasetIntegrity:
    """Validates structural integrity, schema, and coverage of dataset.json."""

    def test_dataset_file_exists_and_is_valid_json(self) -> None:
        assert DEFAULT_DATASET_PATH.exists(), f"Missing dataset at {DEFAULT_DATASET_PATH}"
        with DEFAULT_DATASET_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 30, f"Expected at least 30 cases, found {len(data)}"

    def test_dataset_schema_and_value_domains(self) -> None:
        cases = load_dataset()
        ids = set()

        for case in cases:
            # Unique ID check
            assert case.id not in ids, f"Duplicate case ID: {case.id}"
            ids.add(case.id)

            # Field presence and types
            assert case.prompt.strip(), f"Empty prompt in case {case.id}"
            assert case.rationale.strip(), f"Empty rationale in case {case.id}"
            assert case.category.strip(), f"Empty category in case {case.id}"
            assert isinstance(case.tags, list), f"Tags must be a list in case {case.id}"

            # Strict domain checks
            assert case.expected_capability in {"general", "research"}, (
                f"Invalid expected_capability '{case.expected_capability}' in {case.id}"
            )
            assert case.clarity in {"clear", "ambiguous"}, (
                f"Invalid clarity '{case.clarity}' in {case.id}"
            )

    def test_dataset_covers_all_required_categories(self) -> None:
        cases = load_dataset()
        dataset_categories = {c.category for c in cases}
        missing = REQUIRED_CATEGORIES - dataset_categories
        assert not missing, f"Dataset is missing required categories: {missing}"

    def test_dataset_contains_both_clear_and_ambiguous_prompts(self) -> None:
        cases = load_dataset()
        clear_count = sum(1 for c in cases if c.clarity == "clear")
        ambiguous_count = sum(1 for c in cases if c.clarity == "ambiguous")

        assert clear_count > 0, "Dataset must contain clear prompts"
        assert ambiguous_count > 0, "Dataset must contain ambiguous prompts"
        assert clear_count > ambiguous_count, "Clear prompts should outnumber ambiguous edge cases"


class TestNormalizationAndValidation:
    """Validates normalization of router output and strict error handling."""

    def test_normalize_capability_valid(self) -> None:
        assert normalize_capability("general") == "general"
        assert normalize_capability("General ") == "general"
        assert normalize_capability("RESEARCH") == "research"
        assert normalize_capability(AgentCapability.GENERAL) == "general"
        assert normalize_capability(AgentCapability.RESEARCH) == "research"

    def test_normalize_capability_unsupported_raises_clear_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported router capability output: 'invalid_cap'"):
            normalize_capability("invalid_cap")

        with pytest.raises(ValueError, match="Unsupported router capability output: 42"):
            normalize_capability(42)

    def test_unsupported_router_output_fails_benchmark_cleanly(self) -> None:
        mock_router = MagicMock()
        mock_router.route.return_value = "unknown_capability"

        cases = [
            EvaluationCase(
                id="syn-1",
                prompt="Test prompt",
                expected_capability="general",
                clarity="clear",
                rationale="Test",
                category="clear_general",
            )
        ]

        with pytest.raises(ValueError, match="Unsupported router capability output: 'unknown_capability'"):
            run_benchmark(router=mock_router, dataset=cases, warmup=0, latency_iterations=1)


class TestBenchmarkPassSeparation:
    """Validates that correctness and latency passes are semantically and operationally decoupled."""

    def test_correctness_pass_evaluates_each_case_once(self) -> None:
        mock_router = MagicMock()
        mock_router.route.return_value = "general"
        mock_router.detect_intent.return_value = None

        cases = [
            EvaluationCase(
                id=f"syn-{i}",
                prompt=f"Prompt {i}",
                expected_capability="general",
                clarity="clear",
                rationale="Test",
                category="clear_general",
            )
            for i in range(5)
        ]

        results, matrix, cat_acc = evaluate_correctness(mock_router, cases)

        assert len(results) == 5
        assert mock_router.route.call_count == 5
        assert mock_router.detect_intent.call_count == 5
        assert matrix.true_general == 5
        assert cat_acc["clear_general"] == 1.0

    def test_latency_pass_repeated_routing_excludes_intent_detection(self) -> None:
        mock_router = MagicMock()
        mock_router.route.return_value = "general"

        prompts = ["Prompt A", "Prompt B"]
        warmup = 3
        latency_iterations = 5

        stats = measure_latency(
            router=mock_router,
            prompts=prompts,
            warmup=warmup,
            latency_iterations=latency_iterations,
        )

        # Total route calls = (warmup + iterations) * len(prompts) = (3 + 5) * 2 = 16
        assert mock_router.route.call_count == 16
        # detect_intent must not be called during latency pass
        assert not hasattr(mock_router, "detect_intent") or mock_router.detect_intent.call_count == 0

        # Sample count corresponds strictly to measured iterations
        assert stats.sample_count == 5 * 2
        assert stats.warmup_iterations == 3
        assert stats.measured_iterations == 5

    def test_configurable_latency_iterations_produce_expected_sample_count(self) -> None:
        cases = [
            EvaluationCase(
                id=f"c-{i}",
                prompt=f"P {i}",
                expected_capability="general",
                clarity="clear",
                rationale="R",
                category="cat",
            )
            for i in range(4)
        ]

        mock_router = MagicMock()
        mock_router.route.return_value = AgentCapability.GENERAL

        result = run_benchmark(
            router=mock_router,
            dataset=cases,
            warmup=2,
            latency_iterations=7,
        )

        assert result.latency.warmup_iterations == 2
        assert result.latency.measured_iterations == 7
        assert result.latency.sample_count == 4 * 7


class TestBenchmarkCalculations:
    """Validates statistical and confusion matrix computation routines."""

    def test_compute_metrics_normal(self) -> None:
        precision, recall, f1 = _compute_metrics(tp=8, fp=2, tn=18, fn=2)
        assert precision == pytest.approx(0.8)
        assert recall == pytest.approx(0.8)
        assert f1 == pytest.approx(0.8)

    def test_compute_metrics_zero_division(self) -> None:
        precision, recall, f1 = _compute_metrics(tp=0, fp=0, tn=10, fn=0)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_compute_percentile_empty_and_single(self) -> None:
        assert _compute_percentile([], 50.0) == 0.0
        assert _compute_percentile([42.0], 50.0) == 42.0
        assert _compute_percentile([42.0], 99.0) == 42.0

    def test_compute_percentile_interpolated(self) -> None:
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _compute_percentile(data, 0.0) == 10.0
        assert _compute_percentile(data, 50.0) == 30.0
        assert _compute_percentile(data, 100.0) == 50.0

    def test_report_includes_latency_measurement_metadata(self) -> None:
        matrix = ConfusionMatrix(true_research=5, false_research=1, true_general=10, false_general=2)
        latency = LatencyStats(
            warmup_iterations=5,
            measured_iterations=15,
            sample_count=60,
            mean_us=12.5,
            p50_us=10.0,
            p90_us=20.0,
            p95_us=25.0,
            p99_us=30.0,
            min_us=5.0,
            max_us=35.0,
            total_ns=750_000,
        )
        dummy_result = BenchmarkResult(
            router_name="TestRouter",
            total_cases=4,
            clear_cases=3,
            ambiguous_cases=1,
            overall_correct=3,
            clear_correct=2,
            ambiguous_correct=1,
            overall_accuracy=0.75,
            clear_accuracy=2 / 3,
            ambiguous_accuracy=1.0,
            research_precision=0.8,
            research_recall=0.7,
            research_f1=0.75,
            general_precision=0.85,
            general_recall=0.9,
            general_f1=0.87,
            confusion_matrix=matrix,
            latency=latency,
            category_accuracies={"cat1": 1.0},
            case_results=[],
        )

        report = format_benchmark_report(dummy_result)
        assert "Warmup Passes:" in report
        assert "Measured Iterations:" in report
        assert "Total Latency Samples: 60" in report


class TestCliBehavior:
    """Validates CLI arguments, --help flag, and dataset override options."""

    def test_cli_help_does_not_run_benchmark(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_cli_dataset_override(self, tmp_path: Path) -> None:
        custom_dataset = tmp_path / "custom.json"
        custom_data = [
            {
                "id": "c-001",
                "prompt": "Hello test prompt",
                "expected_capability": "general",
                "clarity": "clear",
                "rationale": "Custom test",
                "category": "clear_general",
                "tags": ["test"],
            }
        ]
        custom_dataset.write_text(json.dumps(custom_data))

        with patch("evaluation.routing.benchmark.run_benchmark") as mock_run:
            mock_run.return_value = BenchmarkResult(
                router_name="MockRouter",
                total_cases=1,
                clear_cases=1,
                ambiguous_cases=0,
                overall_correct=1,
                clear_correct=1,
                ambiguous_correct=0,
                overall_accuracy=1.0,
                clear_accuracy=1.0,
                ambiguous_accuracy=0.0,
                research_precision=0.0,
                research_recall=0.0,
                research_f1=0.0,
                general_precision=1.0,
                general_recall=1.0,
                general_f1=1.0,
                confusion_matrix=ConfusionMatrix(0, 0, 1, 0),
                latency=LatencyStats(0, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000),
                category_accuracies={"clear_general": 1.0},
                case_results=[],
            )
            exit_code = main(["--dataset", str(custom_dataset), "--latency-iterations", "1", "--warmup", "0"])

            assert exit_code == 0
            assert mock_run.call_count == 1
            passed_cases = mock_run.call_args.kwargs["dataset"]
            assert len(passed_cases) == 1
            assert passed_cases[0].id == "c-001"

    def test_cli_baseline_v2_flag(self) -> None:
        with patch("evaluation.routing.benchmark.run_benchmark") as mock_run:
            mock_run.return_value = BenchmarkResult(
                router_name="LegacyRoutingV2Router",
                total_cases=1,
                clear_cases=1,
                ambiguous_cases=0,
                overall_correct=1,
                clear_correct=1,
                ambiguous_correct=0,
                overall_accuracy=1.0,
                clear_accuracy=1.0,
                ambiguous_accuracy=0.0,
                research_precision=0.0,
                research_recall=0.0,
                research_f1=0.0,
                general_precision=1.0,
                general_recall=1.0,
                general_f1=1.0,
                confusion_matrix=ConfusionMatrix(0, 0, 1, 0),
                latency=LatencyStats(0, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000),
                category_accuracies={"clear_general": 1.0},
                case_results=[],
            )
            exit_code = main(["--baseline-v2", "--latency-iterations", "1", "--warmup", "0"])

            assert exit_code == 0
            assert mock_run.call_count == 1
            assert mock_run.call_args.kwargs["legacy_v2"] is True


class TestFullDatasetBenchmarkExecution:
    """Verifies end-to-end benchmark execution on the full dataset without fragile timing asserts."""

    def test_full_dataset_execution_completes_deterministically(self) -> None:
        result = run_benchmark(warmup=2, latency_iterations=3)

        assert result.total_cases == 44
        assert result.clear_cases + result.ambiguous_cases == result.total_cases
        assert result.overall_correct <= result.total_cases
        assert 0.0 <= result.overall_accuracy <= 1.0

        # Phase B Assessment Metrics
        assert result.clarity_confusion_matrix is not None
        assert result.clarity_confusion_matrix.total == result.total_cases
        assert 0.0 <= result.clarity_accuracy <= 1.0
        assert 0.0 <= result.advisor_delegation_rate <= 1.0
        assert result.clear_assessment_accuracy > 0.0

        # Confusion matrix sum must equal total cases
        assert result.confusion_matrix.total == result.total_cases

        # Latency statistics must reflect the configured passes: 44 * 3 = 132 samples
        assert result.latency.warmup_iterations == 2
        assert result.latency.measured_iterations == 3
        assert result.latency.sample_count == 132
        assert result.latency.mean_us > 0.0
        assert result.latency.p50_us > 0.0
        assert result.latency.min_us >= 0.0
        assert result.latency.max_us >= result.latency.min_us
        assert result.latency.total_ns > 0

        # All case results populated from correctness pass
        assert len(result.case_results) == result.total_cases

    def test_full_dataset_execution_with_legacy_v2_baseline(self) -> None:
        result = run_benchmark(warmup=1, latency_iterations=1, legacy_v2=True)
        assert result.router_name == "LegacyRoutingV2Router"
        assert result.total_cases == 44
        assert result.overall_correct == 28
        assert round(result.overall_accuracy * 100, 2) == 63.64


class TestValidationDatasetIntegrity:
    """Validates structural integrity, schema, and coverage of validation_dataset.json."""

    def test_validation_file_exists_and_is_valid_json(self) -> None:
        assert VALIDATION_DATASET_PATH.exists(), f"Missing dataset at {VALIDATION_DATASET_PATH}"
        with VALIDATION_DATASET_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 20, f"Expected at least 20 validation cases, found {len(data)}"

    def test_validation_schema_and_value_domains(self) -> None:
        cases = load_dataset(VALIDATION_DATASET_PATH)
        ids = set()

        for case in cases:
            assert case.id not in ids, f"Duplicate validation case ID: {case.id}"
            ids.add(case.id)
            assert case.prompt.strip(), f"Empty prompt in case {case.id}"
            assert case.rationale.strip(), f"Empty rationale in case {case.id}"
            assert case.category.strip(), f"Empty category in case {case.id}"
            assert isinstance(case.tags, list), f"Tags must be a list in case {case.id}"
            assert case.expected_capability in {"general", "research"}
            assert case.clarity in {"clear", "ambiguous"}

    def test_validation_benchmark_execution(self) -> None:
        cases = load_dataset(VALIDATION_DATASET_PATH)
        result = run_benchmark(
            dataset=cases,
            warmup=1,
            latency_iterations=1,
            dataset_name="validation",
        )
        assert result.total_cases == len(cases)
        assert result.dataset_name == "validation"
        assert result.overall_accuracy > 0.90
        assert result.clarity_accuracy > 0.90
        assert 0.0 <= result.advisor_delegation_rate <= 1.0


class TestCliValidationAndAllModes:
    """Verifies that the CLI correctly parses and dispatches --validation and --all."""

    @patch("evaluation.routing.benchmark.run_benchmark")
    @patch("evaluation.routing.benchmark.load_dataset")
    def test_cli_validation_flag_loads_validation_dataset(
        self, mock_load: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_load.return_value = []
        mock_run.return_value = BenchmarkResult(
            router_name="TestRouter",
            total_cases=1,
            clear_cases=1,
            ambiguous_cases=0,
            overall_correct=1,
            clear_correct=1,
            ambiguous_correct=0,
            overall_accuracy=1.0,
            clear_accuracy=1.0,
            ambiguous_accuracy=0.0,
            research_precision=0.0,
            research_recall=0.0,
            research_f1=0.0,
            general_precision=1.0,
            general_recall=1.0,
            general_f1=1.0,
            confusion_matrix=ConfusionMatrix(0, 0, 1, 0),
            latency=LatencyStats(0, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000),
            category_accuracies={"clear_general": 1.0},
            dataset_name="validation",
            case_results=[],
        )
        exit_code = main(["--validation", "--latency-iterations", "1", "--warmup", "0"])
        assert exit_code == 0
        mock_load.assert_called_with(VALIDATION_DATASET_PATH)
        assert mock_run.call_args.kwargs["dataset_name"] == "validation"


class TestBenchmarkAdvisorSemantics:
    """Verifies that provisional capability and final post-advisor capability remain decoupled."""

    def test_case_result_distinguishes_provisional_and_final_capability(self) -> None:
        case = EvaluationCase(
            id="test-001",
            prompt="Test prompt",
            expected_capability="general",
            clarity="ambiguous",
            rationale="Rationale",
            category="ambiguous_borderline",
        )
        result = CaseResult(
            case=case,
            actual_capability="research",  # Final capability after advisor override
            provisional_capability="general",  # Deterministic provisional capability
        )
        assert result.provisional_capability == "general"
        assert result.effective_provisional_capability == "general"
        assert result.final_capability == "research"
        assert result.actual_capability == "research"
        assert result.is_provisional_correct is True
        assert result.is_final_correct is False
        assert result.is_correct is False

    def test_simulated_advisor_decouples_provisional_and_final_accuracy(self) -> None:
        result = run_benchmark(warmup=0, latency_iterations=1, simulate_advisor=True)

        assert result.total_cases == 44
        assert result.clear_cases == 39
        assert result.ambiguous_cases == 5
        assert result.advisor_consultation_count == 5
        assert result.simulated_advisor_enabled is True

        # Deterministic provisional accuracy must be 100% (44/44)
        assert result.provisional_overall_correct == 44
        assert result.provisional_overall_accuracy == 1.0
        assert result.provisional_clear_correct == 39
        assert result.provisional_clear_accuracy == 1.0
        assert result.provisional_ambiguous_correct == 5
        assert result.provisional_ambiguous_accuracy == 1.0

        # Final post-simulation capability accuracy reflects synthetic override on eval-012
        assert result.overall_correct == 43
        assert round(result.overall_accuracy * 100, 2) == 97.73
        assert result.clear_correct == 39
        assert result.clear_accuracy == 1.0
        assert result.ambiguous_correct == 4
        assert result.ambiguous_accuracy == 0.8

    def test_eval_012_override_behavior_and_report_attribution(self) -> None:
        result = run_benchmark(warmup=0, latency_iterations=1, simulate_advisor=True)
        eval_012_results = [r for r in result.case_results if r.case.id == "eval-012"]
        assert len(eval_012_results) == 1
        r = eval_012_results[0]

        assert r.case.expected_capability == "general"
        assert r.effective_provisional_capability == "general"
        assert r.is_provisional_correct is True
        assert r.advisor_recommendation is not None
        assert r.advisor_recommendation.recommended_capability == AgentCapability.RESEARCH
        assert r.final_capability == "research"
        assert r.is_final_correct is False

        report = format_benchmark_report(result)
        assert "Provisional Capability Acc:  100.00% (44/44)" in report
        assert "Final Capability Acc:         97.73% (43/44)" in report
        assert "Disagreement Source: Deterministic assessment was CORRECT; advisor recommendation changed final routing." in report
        assert "synthetic advisor override" not in report
        assert "[eval-012]" in report
        assert "Advisor Action:    Changed to RESEARCH" in report

    def test_clear_assessments_never_invoke_advisor(self) -> None:
        result = run_benchmark(warmup=0, latency_iterations=1, simulate_advisor=True)
        clear_results = [r for r in result.case_results if r.case.clarity == "clear"]
        assert len(clear_results) == 39

        for r in clear_results:
            assert r.advisor_recommendation is None
            assert r.fallback_used is False
            assert r.effective_provisional_capability == r.final_capability

    def test_effective_provisional_capability_derivation_from_assessment(self) -> None:
        from orchestrator.domain.assessment import RoutingAssessment, RoutingClarity

        case_gen = EvaluationCase(
            id="t-gen",
            prompt="Gen prompt",
            expected_capability="general",
            clarity="clear",
            rationale="Test",
            category="clear_general",
        )
        assessment_gen = RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.CLEAR,
            signals=(),
            reason="Clear general",
        )
        res_gen = CaseResult(
            case=case_gen,
            actual_capability="general",
            provisional_capability=None,  # Omitted/None
            assessment=assessment_gen,
        )
        assert res_gen.effective_provisional_capability == "general"
        assert res_gen.is_provisional_correct is True

        case_res = EvaluationCase(
            id="t-res",
            prompt="Res prompt",
            expected_capability="research",
            clarity="clear",
            rationale="Test",
            category="clear_research",
        )
        assessment_res = RoutingAssessment(
            provisional_capability=AgentCapability.RESEARCH,
            clarity=RoutingClarity.CLEAR,
            signals=(),
            reason="Clear research",
        )
        res_res = CaseResult(
            case=case_res,
            actual_capability="research",
            provisional_capability=None,  # Omitted/None
            assessment=assessment_res,
        )
        assert res_res.effective_provisional_capability == "research"
        assert res_res.is_provisional_correct is True

    def test_report_distinguishes_advisor_agreement_and_change_actions(self) -> None:
        from orchestrator.domain.advisor import AdvisorRecommendation

        case = EvaluationCase(
            id="t-disagree",
            prompt="Prompt",
            expected_capability="research",
            clarity="ambiguous",
            rationale="R",
            category="cat",
        )
        # Agreement case: both provisional and advisor were GENERAL, expected RESEARCH
        rec_agree = AdvisorRecommendation(
            recommended_capability=AgentCapability.GENERAL,
            reason="Advisor concurred with provisional capability",
        )
        res_agree = CaseResult(
            case=case,
            actual_capability="general",
            provisional_capability="general",
            advisor_recommendation=rec_agree,
        )
        matrix = ConfusionMatrix(0, 0, 0, 1)
        latency = LatencyStats(0, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000)
        benchmark_agree = BenchmarkResult(
            router_name="TestRouter",
            total_cases=1,
            clear_cases=0,
            ambiguous_cases=1,
            overall_correct=0,
            clear_correct=0,
            ambiguous_correct=0,
            overall_accuracy=0.0,
            clear_accuracy=0.0,
            ambiguous_accuracy=0.0,
            research_precision=0.0,
            research_recall=0.0,
            research_f1=0.0,
            general_precision=0.0,
            general_recall=0.0,
            general_f1=0.0,
            confusion_matrix=matrix,
            latency=latency,
            category_accuracies={"cat": 0.0},
            case_results=[res_agree],
            simulated_advisor_enabled=True,
            advisor_consultation_count=1,
        )
        report_agree = format_benchmark_report(benchmark_agree)
        assert "Advisor Action:    Agreed with GENERAL" in report_agree
        assert "Changed to" not in report_agree

        # Change case: advisor changed general -> research, expected general
        case_gen = EvaluationCase(
            id="t-change",
            prompt="Prompt",
            expected_capability="general",
            clarity="ambiguous",
            rationale="R",
            category="cat",
        )
        rec_change = AdvisorRecommendation(
            recommended_capability=AgentCapability.RESEARCH,
            reason="Advisor decided research was appropriate",
        )
        res_change = CaseResult(
            case=case_gen,
            actual_capability="research",
            provisional_capability="general",
            advisor_recommendation=rec_change,
        )
        benchmark_change = BenchmarkResult(
            router_name="TestRouter",
            total_cases=1,
            clear_cases=0,
            ambiguous_cases=1,
            overall_correct=0,
            clear_correct=0,
            ambiguous_correct=0,
            overall_accuracy=0.0,
            clear_accuracy=0.0,
            ambiguous_accuracy=0.0,
            research_precision=0.0,
            research_recall=0.0,
            research_f1=0.0,
            general_precision=0.0,
            general_recall=0.0,
            general_f1=0.0,
            confusion_matrix=matrix,
            latency=latency,
            category_accuracies={"cat": 0.0},
            case_results=[res_change],
            simulated_advisor_enabled=True,
            advisor_consultation_count=1,
        )
        report_change = format_benchmark_report(benchmark_change)
        assert "Advisor Action:    Changed to RESEARCH" in report_change
        assert "Agreed with" not in report_change

    def test_benchmark_run_enforces_populated_provisional_capability_invariant(self) -> None:
        result_dev = run_benchmark(warmup=0, latency_iterations=1, simulate_advisor=True)
        assert len(result_dev.case_results) == 44
        assert all(r.provisional_capability is not None for r in result_dev.case_results)

        result_val = run_benchmark(
            dataset_name="validation",
            dataset=load_dataset(VALIDATION_DATASET_PATH),
            warmup=0,
            latency_iterations=1,
            simulate_advisor=True,
        )
        assert len(result_val.case_results) == 34
        assert all(r.provisional_capability is not None for r in result_val.case_results)
