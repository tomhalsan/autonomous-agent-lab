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
