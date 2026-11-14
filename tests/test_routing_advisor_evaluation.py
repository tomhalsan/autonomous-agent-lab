"""Deterministic offline unit tests for model-backed routing advisor evaluation (Increment 3)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from orchestrator.domain.advisor import (
    AdvisorExecutionError,
    AdvisorRecommendation,
    AdvisorTimeoutError,
    AdvisorUnavailableError,
)
from orchestrator.domain.assessment import RoutingAssessment, RoutingClarity
from orchestrator.domain.capability import AgentCapability

from evaluation.routing.advisor_evaluation import (
    _compute_distribution,
    compute_advisor_quality_metrics,
    execute_advisor_consultation_async,
    execute_advisor_consultation_sync,
    execute_cold_probe,
)
from evaluation.routing.benchmark import (
    DEFAULT_DATASET_PATH,
    _build_benchmark_result,
    build_parser,
    compute_advisor_latency_stats,
    format_benchmark_report,
    load_dataset,
    main,
    run_benchmark_async,
    validate_cli_args,
)
from evaluation.routing.models import (
    CaseResult,
    ColdProbeOutcome,
    ConfusionMatrix,
    EvaluationCase,
    LatencyStats,
    OllamaProviderTiming,
    parse_provider_duration_ns,
)
from evaluation.routing.ollama_advisor import OllamaRoutingAdvisor


def make_mock_client(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    text_content: str | None = None,
) -> httpx.AsyncClient:
    """Create a mock httpx.AsyncClient with deterministic responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        if text_content is not None:
            return httpx.Response(status_code=status_code, text=text_content)
        return httpx.Response(
            status_code=status_code,
            json=json_data
            if json_data is not None
            else {
                "response": '{"capability": "general", "reason": "Default test reason"}',
                "total_duration": 1_500_000_000,
                "load_duration": 200_000_000,
                "prompt_eval_duration": 300_000_000,
                "eval_duration": 1_000_000_000,
            },
        )

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


def make_assessment(
    capability: AgentCapability = AgentCapability.GENERAL,
    clarity: RoutingClarity = RoutingClarity.AMBIGUOUS,
) -> RoutingAssessment:
    return RoutingAssessment(
        provisional_capability=capability,
        clarity=clarity,
        signals=(),
        reason="Test assessment reason",
    )


class TestCliIntegrationAndValidation:
    """Tests 1, 2, 18, 19: CLI backward compatibility and flag validation."""

    def test_1_deterministic_cli_still_works_unchanged(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Existing default deterministic benchmark execution works with no advisor flags."""
        ret = main([])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "ROUTING BENCHMARK REPORT: DeterministicCapabilityRouter" in captured
        assert "Provisional Capability Acc:  100.00% (45/45)" in captured
        assert "Final Capability Acc:        100.00% (45/45)" in captured
        assert "SIMULATED ADVISOR PLUMBING VALIDATION" not in captured
        assert "MODEL ADVISOR EXPERIMENTAL CONFIGURATION" not in captured

    def test_2_simulate_advisor_cli_still_works_unchanged(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Existing --simulate-advisor flag remains fully compatible."""
        ret = main(["--simulate-advisor"])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "SIMULATED ADVISOR PLUMBING VALIDATION" in captured
        assert "Total Ambiguous Eligible:          5" in captured
        assert "Clear Assessment Invocations:      0" in captured
        assert "Provisional Capability Acc:  100.00% (45/45)" in captured
        assert "Final Capability Acc:         97.78% (44/45)" in captured

    def test_18_cli_rejects_missing_ollama_model(self) -> None:
        """Selecting --advisor-provider ollama without --ollama-model raises ValueError."""
        args = build_parser().parse_args(["--advisor-provider", "ollama"])
        with pytest.raises(ValueError, match="--ollama-model is required"):
            validate_cli_args(args)

    def test_19_cli_does_not_silently_enable_ollama(self) -> None:
        """Supplying Ollama-specific configuration without --advisor-provider ollama is rejected."""
        args1 = build_parser().parse_args(["--ollama-model", "llama3.2:3b"])
        with pytest.raises(ValueError, match="Ollama options .* require --advisor-provider ollama"):
            validate_cli_args(args1)

        args2 = build_parser().parse_args(["--cold-probe"])
        with pytest.raises(ValueError, match="require --advisor-provider ollama"):
            validate_cli_args(args2)

        args3 = build_parser().parse_args(["--unload-after-run"])
        with pytest.raises(ValueError, match="require --advisor-provider ollama"):
            validate_cli_args(args3)

        args4 = build_parser().parse_args(["--simulate-advisor", "--advisor-provider", "ollama", "--ollama-model", "m"])
        with pytest.raises(ValueError, match="Cannot combine --simulate-advisor and --advisor-provider"):
            validate_cli_args(args4)


class TestAdvisorScoringAndAuthorityInvariants:
    """Tests 3, 4, 5, 6, 7, 8, 22, 23: Advisor consultation boundaries and metrics."""

    @pytest.mark.anyio
    async def test_3_and_4_real_advisor_only_invokes_ambiguous_and_zero_clear(self) -> None:
        """Advisor is invoked ONLY on AMBIGUOUS cases; CLEAR invocation count is strictly 0."""
        invoked_prompts: list[str] = []

        class RecordingAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                if assessment.clarity == RoutingClarity.CLEAR:
                    raise RuntimeError("Protocol violation: Advisor invoked on CLEAR case!")
                invoked_prompts.append(prompt)
                return AdvisorRecommendation(
                    recommended_capability=AgentCapability.GENERAL,
                    reason="Classified as general",
                )

        cases = load_dataset(DEFAULT_DATASET_PATH)
        result = await run_benchmark_async(
            dataset=cases,
            advisor=RecordingAdvisor(),
            warmup=0,
            latency_iterations=1,
        )

        ambiguous_cases = [c for c in cases if c.clarity == "ambiguous"]
        assert len(ambiguous_cases) == 5
        assert len(invoked_prompts) == 5
        assert result.advisor_eligible_count == 5
        assert result.advisor_consultation_count == 5
        assert result.advisor_clear_invocation_count == 0

    @pytest.mark.anyio
    async def test_5_advisor_agreement_counted_correctly(self) -> None:
        """When advisor confirms provisional capability, agreement is incremented."""
        class AgreeingAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                return AdvisorRecommendation(
                    recommended_capability=assessment.provisional_capability,
                    reason="Agreed with provisional",
                )

        cases = load_dataset(DEFAULT_DATASET_PATH)
        result = await run_benchmark_async(
            dataset=cases,
            advisor=AgreeingAdvisor(),
            warmup=0,
            latency_iterations=1,
        )

        assert result.advisor_agreement_count == 5
        assert result.advisor_override_count == 0
        assert result.advisor_successful_count == 5

    @pytest.mark.anyio
    async def test_6_correct_change_counted_correctly(self) -> None:
        """Advisor changes provisional incorrect into final correct capability."""
        case = EvaluationCase(
            id="test-corr-change",
            prompt="Analyze recent CVE disclosures for OpenSSL",
            expected_capability="research",
            clarity="ambiguous",
            rationale="Ambiguous prompt where expected is research",
            category="ambiguous_borderline",
        )

        class ChangingAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                return AdvisorRecommendation(
                    recommended_capability=AgentCapability.RESEARCH,
                    reason="Correctly recommends research",
                )

        # Mock router that provisionally produces GENERAL on ambiguous case
        mock_router = MagicMock()
        mock_router.assess.return_value = RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.AMBIGUOUS,
            signals=(),
            reason="Provisional general",
        )
        mock_router.route.return_value = AgentCapability.GENERAL

        result = await run_benchmark_async(
            router=mock_router,
            dataset=[case],
            advisor=ChangingAdvisor(),
            warmup=0,
            latency_iterations=1,
        )

        assert result.advisor_consultation_count == 1
        assert result.advisor_successful_count == 1
        assert result.advisor_override_count == 1
        assert result.advisor_correct_change_count == 1
        assert result.advisor_regression_count == 0
        assert result.provisional_overall_accuracy == 0.0
        assert result.overall_accuracy == 1.0

    @pytest.mark.anyio
    async def test_7_regression_counted_correctly(self) -> None:
        """Advisor changes provisional correct into final incorrect capability."""
        case = EvaluationCase(
            id="test-regression",
            prompt="What are the best practices for securing a REST API?",
            expected_capability="general",
            clarity="ambiguous",
            rationale="Expected general",
            category="ambiguous_borderline",
        )

        class RegressingAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                return AdvisorRecommendation(
                    recommended_capability=AgentCapability.RESEARCH,
                    reason="Regresses to research",
                )

        mock_router = MagicMock()
        mock_router.assess.return_value = RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.AMBIGUOUS,
            signals=(),
            reason="Provisional general",
        )
        mock_router.route.return_value = AgentCapability.GENERAL

        result = await run_benchmark_async(
            router=mock_router,
            dataset=[case],
            advisor=RegressingAdvisor(),
            warmup=0,
            latency_iterations=1,
        )

        assert result.advisor_consultation_count == 1
        assert result.advisor_override_count == 1
        assert result.advisor_regression_count == 1
        assert result.advisor_correct_change_count == 0
        assert result.provisional_overall_accuracy == 1.0
        assert result.overall_accuracy == 0.0

    @pytest.mark.anyio
    async def test_8_preserved_correct_and_preserved_incorrect_outcomes_counted(self) -> None:
        """Preserved-correct and preserved-incorrect metrics are recorded accurately."""
        cases = [
            EvaluationCase(
                id="case-pres-correct",
                prompt="Prompt 1",
                expected_capability="general",
                clarity="ambiguous",
                rationale="r1",
                category="ambiguous_borderline",
            ),
            EvaluationCase(
                id="case-pres-incorrect",
                prompt="Prompt 2",
                expected_capability="research",
                clarity="ambiguous",
                rationale="r2",
                category="ambiguous_borderline",
            ),
        ]

        class ConfirmingAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                return AdvisorRecommendation(
                    recommended_capability=AgentCapability.GENERAL,
                    reason="Recommends general for both",
                )

        mock_router = MagicMock()
        mock_router.assess.return_value = RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.AMBIGUOUS,
            signals=(),
            reason="Provisional general",
        )
        mock_router.route.return_value = AgentCapability.GENERAL

        result = await run_benchmark_async(
            router=mock_router,
            dataset=cases,
            advisor=ConfirmingAdvisor(),
            warmup=0,
            latency_iterations=1,
        )

        assert result.advisor_consultation_count == 2
        assert result.advisor_agreement_count == 2
        assert result.advisor_preserved_correct_count == 1
        assert result.advisor_preserved_incorrect_count == 1

    @pytest.mark.anyio
    async def test_22_and_23_provisional_final_semantics_and_clear_cases_unmodified(self) -> None:
        """CLEAR cases are completely unmodified by advisor path; provisional accuracy is distinct."""
        cases = load_dataset(DEFAULT_DATASET_PATH)

        class ChangingAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                # Deliberately change everything ambiguous to RESEARCH
                return AdvisorRecommendation(
                    recommended_capability=AgentCapability.RESEARCH,
                    reason="Recommend research",
                )

        result = await run_benchmark_async(
            dataset=cases,
            advisor=ChangingAdvisor(),
            warmup=0,
            latency_iterations=1,
        )

        # All 40 clear cases remain 100% provisional and final correct
        assert result.clear_cases == 40
        assert result.provisional_clear_accuracy == 1.0
        assert result.final_clear_accuracy == 1.0
        assert result.advisor_clear_invocation_count == 0


class TestAdvisorFailureAndFallbackHandling:
    """Tests 9, 10, 11: Failures produce fallback without aborting the run."""

    @pytest.mark.anyio
    async def test_9_advisor_failure_produces_fallback_without_aborting_run(self) -> None:
        """Advisor failures trigger deterministic fallback and run completes cleanly."""
        class FailingAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                raise AdvisorUnavailableError("Ollama daemon down")

        cases = load_dataset(DEFAULT_DATASET_PATH)
        result = await run_benchmark_async(
            dataset=cases,
            advisor=FailingAdvisor(),
            warmup=0,
            latency_iterations=1,
        )

        assert result.advisor_consultation_count == 5
        assert result.advisor_fallback_count == 5
        assert result.advisor_unavailable_count == 5
        assert result.advisor_successful_count == 0
        # All ambiguous cases fell back to provisional capability (which is 100% on dev dataset)
        assert result.provisional_overall_accuracy == 1.0
        assert result.overall_accuracy == 1.0

    @pytest.mark.anyio
    async def test_10_timeout_categorized_separately(self) -> None:
        """AdvisorTimeoutError is categorized as timeout fallback."""
        class TimeoutAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                raise AdvisorTimeoutError("Request deadline exceeded")

        cases = load_dataset(DEFAULT_DATASET_PATH)
        result = await run_benchmark_async(
            dataset=cases,
            advisor=TimeoutAdvisor(),
            warmup=0,
            latency_iterations=1,
        )

        assert result.advisor_fallback_count == 5
        assert result.advisor_timeout_count == 5
        assert result.advisor_unavailable_count == 0
        assert result.advisor_execution_error_count == 0

    @pytest.mark.anyio
    async def test_11_malformed_advisor_response_categorized_as_execution_error(self) -> None:
        """AdvisorExecutionError is categorized as execution error fallback."""
        class MalformedAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                raise AdvisorExecutionError("Invalid JSON structure")

        cases = load_dataset(DEFAULT_DATASET_PATH)
        result = await run_benchmark_async(
            dataset=cases,
            advisor=MalformedAdvisor(),
            warmup=0,
            latency_iterations=1,
        )

        assert result.advisor_fallback_count == 5
        assert result.advisor_execution_error_count == 5
        assert result.advisor_timeout_count == 0
        assert result.advisor_unavailable_count == 0


class TestTelemetryAndLatencySeparation:
    """Tests 12, 13, 14, 15, 16, 17: Wall clock, provider telemetry, cold vs warm."""

    @pytest.mark.anyio
    async def test_12_client_wall_clock_kept_separate_from_deterministic_latency(self) -> None:
        """Client-observed advisor wall-clock latency is separated from deterministic router latency."""
        async with make_mock_client() as client:
            advisor = OllamaRoutingAdvisor(model="test-model", client=client)
            cases = load_dataset(DEFAULT_DATASET_PATH)
            result = await run_benchmark_async(
                dataset=cases,
                advisor=advisor,
                warmup=2,
                latency_iterations=5,
            )

            # Deterministic latency is in microseconds (µs)
            assert result.latency.mean_us > 0.0
            assert result.latency.sample_count == 45 * 5

            # Advisor latency is in nanoseconds (ns) and calculated only over ambiguous cases (5 cases)
            assert result.advisor_latency is not None
            assert result.advisor_latency.sample_count == 5
            assert result.advisor_latency.client_mean_ns > 0.0
            assert result.advisor_latency.client_min_ns > 0

    @pytest.mark.anyio
    async def test_13_provider_timing_telemetry_recorded_separately(self) -> None:
        """Provider duration telemetry from Ollama /api/generate is recorded accurately in nanoseconds."""
        async with make_mock_client(
            json_data={
                "response": '{"capability": "general", "reason": "Valid reason"}',
                "total_duration": 1_200_000_000,
                "load_duration": 150_000_000,
                "prompt_eval_duration": 250_000_000,
                "eval_duration": 800_000_000,
            }
        ) as client:
            advisor = OllamaRoutingAdvisor(model="test-model", client=client)
            cases = load_dataset(DEFAULT_DATASET_PATH)
            result = await run_benchmark_async(
                dataset=cases,
                advisor=advisor,
                warmup=0,
                latency_iterations=1,
            )

            assert result.advisor_latency is not None
            assert result.advisor_latency.provider_total_duration_mean_ns == 1_200_000_000.0
            assert result.advisor_latency.provider_load_duration_mean_ns == 150_000_000.0
            assert result.advisor_latency.provider_prompt_eval_duration_mean_ns == 250_000_000.0
            assert result.advisor_latency.provider_eval_duration_mean_ns == 800_000_000.0

    @pytest.mark.anyio
    async def test_14_missing_provider_telemetry_does_not_invalidate_recommendation(self) -> None:
        """Missing provider timing durations do not invalidate an otherwise valid advisory response."""
        async with make_mock_client(
            json_data={"response": '{"capability": "general", "reason": "No timing telemetry"}'}
        ) as client:
            advisor = OllamaRoutingAdvisor(model="test-model", client=client)
            cases = load_dataset(DEFAULT_DATASET_PATH)
            result = await run_benchmark_async(
                dataset=cases,
                advisor=advisor,
                warmup=0,
                latency_iterations=1,
            )

            assert result.advisor_successful_count == 5
            assert result.advisor_fallback_count == 0
            assert result.advisor_latency is not None
            assert result.advisor_latency.provider_total_duration_mean_ns is None
            assert result.advisor_latency.provider_load_duration_mean_ns is None

    def test_15_invalid_provider_timing_values_rejected_or_ignored(self) -> None:
        """Invalid timing values (boolean, negative, string, float) are rejected safely."""
        assert parse_provider_duration_ns(123456) == 123456
        assert parse_provider_duration_ns(0) == 0
        assert parse_provider_duration_ns(-1) is None
        assert parse_provider_duration_ns(True) is None
        assert parse_provider_duration_ns(False) is None
        assert parse_provider_duration_ns("1000") is None
        assert parse_provider_duration_ns(12.34) is None
        assert parse_provider_duration_ns(None) is None

    @pytest.mark.anyio
    async def test_16_and_17_cold_probe_excluded_from_accuracy_and_warm_latency(self) -> None:
        """Cold probe request is executed, recorded separately, and excluded from scored dataset metrics."""
        async with make_mock_client() as client:
            advisor = OllamaRoutingAdvisor(model="test-model", client=client)
            cases = load_dataset(DEFAULT_DATASET_PATH)
            result = await run_benchmark_async(
                dataset=cases,
                advisor=advisor,
                advisor_provider="ollama",
                cold_probe=True,
                warmup=0,
                latency_iterations=1,
            )

            # Accuracy still scored across exactly 45 dataset cases
            assert result.total_cases == 45
            assert len(result.case_results) == 45

            # Warm latency sample count is exactly 5 (the 5 ambiguous cases), excluding the cold probe
            assert result.advisor_latency is not None
            assert result.advisor_latency.sample_count == 5
            assert result.advisor_latency.cold_probe_client_ns is not None
            assert result.advisor_latency.cold_probe_client_ns > 0
            assert result.advisor_latency.cold_probe_label is not None
            assert result.advisor_metadata is not None
            assert result.advisor_metadata.cold_probe_performed is True


class TestReportingAndSanitization:
    """Tests 20, 21: Model metadata in report and raw model sanitization."""

    @pytest.mark.anyio
    async def test_20_reports_identify_model_provider_and_experimental_conditions(self) -> None:
        """Report contains explicit model, provider, URL, timeout, schema, and latency profile."""
        async with make_mock_client() as client:
            advisor = OllamaRoutingAdvisor(
                model="llama3.2:3b",
                base_url="http://test-host:11434",
                timeout=25.0,
                keep_alive="10m",
                client=client,
            )
            cases = load_dataset(DEFAULT_DATASET_PATH)
            result = await run_benchmark_async(
                dataset=cases,
                advisor=advisor,
                advisor_provider="ollama",
                cold_probe=True,
                warmup=0,
                latency_iterations=1,
            )

            report = format_benchmark_report(result)
            assert "MODEL ADVISOR EXPERIMENTAL CONFIGURATION" in report
            assert "Provider:                    ollama" in report
            assert "Model:                       llama3.2:3b" in report
            assert "Base URL:                    http://test-host:11434" in report
            assert "Timeout:                     25.0 s" in report
            assert "Keep Alive:                  10m" in report
            assert "Cold Probe Performed:        True" in report
            assert "ADVISOR QUALITY & DISAMBIGUATION PERFORMANCE" in report
            assert "ADVISOR LATENCY PROFILE" in report

    @pytest.mark.anyio
    async def test_21_no_raw_model_output_emitted_in_reports_or_errors(self) -> None:
        """Malformed or unexpected model outputs do not leak into report strings or exceptions."""
        secret_leak_marker = "SUPER_SECRET_PAYLOAD_TOKEN_LEAK_999"
        async with make_mock_client(
            json_data={
                "response": f'{{"capability": "general", "reason": "ok", "unexpected": "{secret_leak_marker}"}}'
            }
        ) as client:
            advisor = OllamaRoutingAdvisor(model="test-model", client=client)
            case = EvaluationCase(
                id="test-sanitization",
                prompt="Research consensus algorithms",
                expected_capability="general",
                clarity="ambiguous",
                rationale="r",
                category="ambiguous_borderline",
            )
            result = await run_benchmark_async(
                dataset=[case],
                advisor=advisor,
                warmup=0,
                latency_iterations=1,
            )

            report = format_benchmark_report(result)
            assert secret_leak_marker not in report
            assert result.advisor_fallback_count == 1
            assert result.advisor_execution_error_count == 1


class TestIncrement3ReviewBlockers:
    """Targeted regression tests for Increment 3 review blockers (items 1, 3, 5, 7, 9)."""

    def test_24_clear_case_reaching_advisor_raises_invariant_violation(self) -> None:
        """If a CLEAR assessment case records an advisor invocation, RuntimeError is raised."""
        case = EvaluationCase(
            id="clear-inv-case",
            prompt="Hello world",
            expected_capability="general",
            clarity="clear",
            rationale="Clear general",
            category="factual_recall",
        )
        clear_assessment = RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.CLEAR,
            signals=(),
            reason="Clear general",
        )
        # Construct an invalid CaseResult where a clear case invoked the advisor
        violating_result = CaseResult(
            case=case,
            actual_capability="general",
            matched_rules=(),
            assessment=clear_assessment,
            advisor_recommendation=AdvisorRecommendation(
                recommended_capability=AgentCapability.GENERAL,
                reason="Should not have been called",
            ),
            fallback_used=False,
            provisional_capability="general",
            advisor_wall_clock_ns=50_000,
        )

        mock_router = MagicMock()
        mock_latency = LatencyStats(
            warmup_iterations=0,
            measured_iterations=1,
            sample_count=1,
            mean_us=1.0,
            p50_us=1.0,
            p90_us=1.0,
            p95_us=1.0,
            p99_us=1.0,
            min_us=1.0,
            max_us=1.0,
            total_ns=1000,
        )
        matrix = ConfusionMatrix(true_research=0, false_research=0, true_general=1, false_general=0)

        with pytest.raises(
            RuntimeError,
            match="Routing invariant violation: Advisor was invoked on 1 CLEAR assessment",
        ):
            _build_benchmark_result(
                active_router=mock_router,
                cases=[case],
                case_results=[violating_result],
                matrix=matrix,
                category_accuracies={"factual_recall": 1.0},
                latency_stats=mock_latency,
                dataset_name="test",
                simulate_advisor=False,
                advisor=None,
                advisor_provider=None,
                advisor_metadata=None,
                advisor_latency=None,
            )

    @pytest.mark.anyio
    async def test_25_cold_probe_failed_outcome_does_not_report_latency(self) -> None:
        """A failed cold probe reports failure status and category, but leaves latency_ns as None."""
        def handler(request: httpx.Request) -> httpx.Response:
            if "unload" in request.url.path or "generate" in request.url.path:
                return httpx.Response(503, json={"error": "Ollama service unavailable"})
            return httpx.Response(200, json={"response": '{"capability": "general", "reason": "ok"}'})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            advisor = OllamaRoutingAdvisor(model="test-model", client=client)
            case = EvaluationCase(
                id="test-probe-fail",
                prompt="Research consensus protocols",
                expected_capability="research",
                clarity="ambiguous",
                rationale="r",
                category="ambiguous_borderline",
            )
            result = await run_benchmark_async(
                dataset=[case],
                advisor=advisor,
                advisor_provider="ollama",
                cold_probe=True,
                warmup=0,
                latency_iterations=1,
            )

            assert result.advisor_metadata is not None
            probe_outcome = result.advisor_metadata.cold_probe_outcome
            assert probe_outcome is not None
            assert probe_outcome.status == "failed"
            assert probe_outcome.latency_ns is None
            assert probe_outcome.failure_category == "unavailable"

            # Check that report displays failure cleanly without reporting a latency duration
            report = format_benchmark_report(result)
            assert "Cold Probe Measurement:             FAILED (unavailable)" in report

    @pytest.mark.anyio
    async def test_26_custom_advisor_without_provider_flag_reports_custom_provider(self) -> None:
        """Advisor without explicit provider flag reports 'custom' metadata (no class name inference)."""
        class OllamaLikeCustomAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                return AdvisorRecommendation(
                    recommended_capability=AgentCapability.GENERAL,
                    reason="Custom advisor response",
                )

        case = EvaluationCase(
            id="test-custom-prov",
            prompt="Research distributed databases",
            expected_capability="research",
            clarity="ambiguous",
            rationale="r",
            category="ambiguous_borderline",
        )
        result = await run_benchmark_async(
            dataset=[case],
            advisor=OllamaLikeCustomAdvisor(),
            warmup=0,
            latency_iterations=1,
        )

        assert result.advisor_metadata is not None
        assert result.advisor_metadata.provider == "custom"

    @pytest.mark.anyio
    async def test_27_quality_outcome_partition_invariant(self) -> None:
        """Successful recommendations partition mutually exclusively and exhaustively."""
        cases = [
            EvaluationCase(id="c1", prompt="Prompt 1", expected_capability="general", clarity="ambiguous", rationale="r", category="ambiguous_borderline"),
            EvaluationCase(id="c2", prompt="Prompt 2", expected_capability="general", clarity="ambiguous", rationale="r", category="ambiguous_borderline"),
            EvaluationCase(id="c3", prompt="Prompt 3", expected_capability="research", clarity="ambiguous", rationale="r", category="ambiguous_borderline"),
            EvaluationCase(id="c4", prompt="Prompt 4", expected_capability="research", clarity="ambiguous", rationale="r", category="ambiguous_borderline"),
        ]

        class MultiOutcomeAdvisor:
            def __init__(self) -> None:
                self.calls = 0

            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                self.calls += 1
                if self.calls == 1:
                    # c1: expected general, provisional research -> changed to general (correct change)
                    return AdvisorRecommendation(recommended_capability=AgentCapability.GENERAL, reason="Correct change")
                elif self.calls == 2:
                    # c2: expected general, provisional general -> changed to research (regression)
                    return AdvisorRecommendation(recommended_capability=AgentCapability.RESEARCH, reason="Regression")
                elif self.calls == 3:
                    # c3: expected research, provisional research -> agreed research (preserved correct)
                    return AdvisorRecommendation(recommended_capability=AgentCapability.RESEARCH, reason="Preserved correct")
                else:
                    # c4: expected research, provisional general -> agreed general (preserved incorrect)
                    return AdvisorRecommendation(recommended_capability=AgentCapability.GENERAL, reason="Preserved incorrect")

        mock_router = MagicMock()
        # Mock assessments for c1..c4
        assessments = [
            RoutingAssessment(provisional_capability=AgentCapability.RESEARCH, clarity=RoutingClarity.AMBIGUOUS, signals=(), reason="r1"),
            RoutingAssessment(provisional_capability=AgentCapability.GENERAL, clarity=RoutingClarity.AMBIGUOUS, signals=(), reason="r2"),
            RoutingAssessment(provisional_capability=AgentCapability.RESEARCH, clarity=RoutingClarity.AMBIGUOUS, signals=(), reason="r3"),
            RoutingAssessment(provisional_capability=AgentCapability.GENERAL, clarity=RoutingClarity.AMBIGUOUS, signals=(), reason="r4"),
        ]
        mock_router.assess.side_effect = assessments
        mock_router.route.side_effect = [a.provisional_capability for a in assessments]

        result = await run_benchmark_async(
            router=mock_router,
            dataset=cases,
            advisor=MultiOutcomeAdvisor(),
            warmup=0,
            latency_iterations=1,
        )

        assert result.advisor_successful_count == 4
        assert result.advisor_correct_change_count == 1
        assert result.advisor_regression_count == 1
        assert result.advisor_preserved_correct_count == 1
        assert result.advisor_preserved_incorrect_count == 1
        # Sum equals successful_count
        assert result.advisor_successful_count == (
            result.advisor_correct_change_count
            + result.advisor_regression_count
            + result.advisor_preserved_correct_count
            + result.advisor_preserved_incorrect_count
        )

    def test_28_separate_latency_populations_for_successful_vs_failed_attempts(self) -> None:
        """compute_advisor_latency_stats cleanly separates successful vs failed populations."""
        case = EvaluationCase(
            id="c",
            prompt="p",
            expected_capability="general",
            clarity="ambiguous",
            rationale="r",
            category="cat",
        )
        assessment = RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.AMBIGUOUS,
            signals=(),
            reason="reason",
        )
        successful_result = CaseResult(
            case=case,
            actual_capability="general",
            matched_rules=(),
            assessment=assessment,
            advisor_recommendation=AdvisorRecommendation(
                recommended_capability=AgentCapability.GENERAL,
                reason="success",
            ),
            fallback_used=False,
            provisional_capability="general",
            advisor_wall_clock_ns=40_000_000,
        )
        failed_result = CaseResult(
            case=case,
            actual_capability="general",
            matched_rules=(),
            assessment=assessment,
            advisor_recommendation=None,
            fallback_used=True,
            provisional_capability="general",
            advisor_wall_clock_ns=100_000_000,
            advisor_error_category="timeout",
        )

        stats = compute_advisor_latency_stats([successful_result, failed_result])
        assert stats is not None
        assert stats.warm_successful is not None
        assert stats.warm_successful.sample_count == 1
        assert stats.warm_successful.mean_ns == 40_000_000.0

        assert stats.failed_attempts is not None
        assert stats.failed_attempts.sample_count == 1
        assert stats.failed_attempts.mean_ns == 100_000_000.0

        # Backward compatible properties reflect warm_successful only
        assert stats.sample_count == 1
        assert stats.client_mean_ns == 40_000_000.0


class TestIncrement3FinalReviewBlockers:
    """Targeted regression tests for Increment 3 final review items (1, 2, 4, 5, 6)."""

    def test_29_cold_probe_outcome_valid_states(self) -> None:
        """Valid ColdProbeOutcome configurations construct cleanly."""
        # 1. not_requested
        outcome_nr = ColdProbeOutcome(status="not_requested")
        assert outcome_nr.status == "not_requested"
        assert outcome_nr.latency_ns is None
        assert outcome_nr.failure_category is None

        # 2. succeeded
        outcome_succ = ColdProbeOutcome(
            status="succeeded",
            latency_ns=25_000_000,
            provider_total_duration_ns=20_000_000,
            provider_load_duration_ns=5_000_000,
            unload_acknowledged=True,
            label="Unload Requested and Acknowledged; Residency Not Independently Verified",
        )
        assert outcome_succ.status == "succeeded"
        assert outcome_succ.latency_ns == 25_000_000
        assert outcome_succ.failure_category is None
        assert outcome_succ.provider_total_duration_ns == 20_000_000

        # 3. failed
        outcome_fail = ColdProbeOutcome(
            status="failed",
            failure_category="unavailable",
            unload_acknowledged=False,
            label="Initial Probe (Residency Not Independently Verified)",
        )
        assert outcome_fail.status == "failed"
        assert outcome_fail.latency_ns is None
        assert outcome_fail.failure_category == "unavailable"

    def test_30_cold_probe_outcome_rejects_contradictory_states(self) -> None:
        """ColdProbeOutcome validation rejects contradictory or invalid field combinations."""
        # not_requested with latency
        with pytest.raises(ValueError, match="latency_ns must be None when status is 'not_requested'"):
            ColdProbeOutcome(status="not_requested", latency_ns=100)

        # not_requested with failure_category
        with pytest.raises(ValueError, match="failure_category must be None when status is 'not_requested'"):
            ColdProbeOutcome(status="not_requested", failure_category="unavailable")

        # not_requested with provider timing
        with pytest.raises(ValueError, match="provider timing fields must be None when status is 'not_requested'"):
            ColdProbeOutcome(status="not_requested", provider_total_duration_ns=50)

        # succeeded with None latency
        with pytest.raises(ValueError, match="latency_ns is required when status is 'succeeded'"):
            ColdProbeOutcome(status="succeeded", latency_ns=None)

        # succeeded with negative latency
        with pytest.raises(ValueError, match="latency_ns must be a non-negative integer"):
            ColdProbeOutcome(status="succeeded", latency_ns=-1)

        # succeeded with boolean latency
        with pytest.raises(ValueError, match="latency_ns must be a non-negative integer"):
            ColdProbeOutcome(status="succeeded", latency_ns=True)  # type: ignore[arg-type]

        # succeeded with failure_category
        with pytest.raises(ValueError, match="failure_category must be None when status is 'succeeded'"):
            ColdProbeOutcome(status="succeeded", latency_ns=1000, failure_category="timeout")

        # succeeded with negative provider duration
        with pytest.raises(ValueError, match="provider_total_duration_ns must be a non-negative integer"):
            ColdProbeOutcome(status="succeeded", latency_ns=1000, provider_total_duration_ns=-10)

        # succeeded with boolean provider duration
        with pytest.raises(ValueError, match="provider_total_duration_ns must be a non-negative integer"):
            ColdProbeOutcome(status="succeeded", latency_ns=1000, provider_total_duration_ns=False)  # type: ignore[arg-type]

        # failed with latency
        with pytest.raises(ValueError, match="latency_ns must be None when status is 'failed'"):
            ColdProbeOutcome(status="failed", latency_ns=5000, failure_category="execution_error")

        # failed with missing failure_category
        with pytest.raises(ValueError, match="failure_category is required when status is 'failed'"):
            ColdProbeOutcome(status="failed", failure_category=None)

        # failed with provider timing
        with pytest.raises(ValueError, match="provider timing fields must be None when status is 'failed'"):
            ColdProbeOutcome(status="failed", failure_category="timeout", provider_total_duration_ns=500)

    @pytest.mark.anyio
    async def test_31_cold_probe_requires_callable_advise_and_valid_recommendation(self) -> None:
        """Cold probe requires callable advise and a valid AdvisorRecommendation instance."""
        # 1. Missing advise method
        class MissingAdvise:
            pass

        res_missing = await execute_cold_probe(MissingAdvise())
        assert res_missing.status == "failed"
        assert res_missing.failure_category == "execution_error"
        assert res_missing.latency_ns is None

        # 2. Non-callable advise attribute
        class NonCallableAdvise:
            advise = "invalid_string"

        res_non_callable = await execute_cold_probe(NonCallableAdvise())
        assert res_non_callable.status == "failed"
        assert res_non_callable.failure_category == "execution_error"

        # 3. Sync advisor returning non-AdvisorRecommendation object (dict)
        class SyncDictReturnAdvisor:
            def advise(self, prompt: str, assessment: RoutingAssessment) -> object:
                return {"capability": "general", "reason": "dict return"}

        res_sync_dict = await execute_cold_probe(SyncDictReturnAdvisor())
        assert res_sync_dict.status == "failed"
        assert res_sync_dict.failure_category == "execution_error"

        # 4. Async advisor returning None
        class AsyncNoneReturnAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> object:
                return None

        res_async_none = await execute_cold_probe(AsyncNoneReturnAdvisor())
        assert res_async_none.status == "failed"
        assert res_async_none.failure_category == "execution_error"

        # 5. Sync advisor returning valid AdvisorRecommendation
        class SyncValidAdvisor:
            def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                return AdvisorRecommendation(AgentCapability.GENERAL, reason="Valid sync rec")

        res_sync_valid = await execute_cold_probe(SyncValidAdvisor())
        assert res_sync_valid.status == "succeeded"
        assert res_sync_valid.latency_ns is not None and res_sync_valid.latency_ns >= 0
        assert res_sync_valid.failure_category is None

        # 6. Async advisor returning valid AdvisorRecommendation
        class AsyncValidAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                return AdvisorRecommendation(AgentCapability.GENERAL, reason="Valid async rec")

        res_async_valid = await execute_cold_probe(AsyncValidAdvisor())
        assert res_async_valid.status == "succeeded"
        assert res_async_valid.latency_ns is not None and res_async_valid.latency_ns >= 0
        assert res_async_valid.failure_category is None

    def test_32_p99_explicit_availability_threshold(self) -> None:
        """P99 latency is explicit float when N >= 100 and None when N < 100."""
        # Population with N = 50 (< 100 threshold)
        dist_small = _compute_distribution([10_000] * 50)
        assert dist_small is not None
        assert dist_small.sample_count == 50
        assert dist_small.p99_ns is None

        # Population with N = 100 (meets threshold)
        dist_100 = _compute_distribution(list(range(100)))
        assert dist_100 is not None
        assert dist_100.sample_count == 100
        assert dist_100.p99_ns is not None
        assert isinstance(dist_100.p99_ns, float)

    @pytest.mark.anyio
    async def test_33_post_run_unload_acknowledgement_metadata(self) -> None:
        """Unload-after-run metadata records not_requested, acknowledged, or not_acknowledged."""
        case = EvaluationCase(
            id="test-unload-meta",
            prompt="Research distributed consensus",
            expected_capability="research",
            clarity="ambiguous",
            rationale="r",
            category="ambiguous_borderline",
        )

        class AdvisorWithUnload:
            def __init__(self, ack: bool) -> None:
                self.ack = ack

            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                return AdvisorRecommendation(AgentCapability.RESEARCH, reason="ok")

            async def unload(self) -> bool:
                return self.ack

        # 1. Unload not requested
        res_not_requested = await run_benchmark_async(
            dataset=[case],
            advisor=AdvisorWithUnload(ack=True),
            unload_after_run=False,
            warmup=0,
            latency_iterations=1,
        )
        assert res_not_requested.advisor_metadata is not None
        assert res_not_requested.advisor_metadata.unload_status == "not_requested"
        rep_nr = format_benchmark_report(res_not_requested)
        assert "Unload After Run:            False (Not Requested)" in rep_nr

        # 2. Unload requested and acknowledged
        res_ack = await run_benchmark_async(
            dataset=[case],
            advisor=AdvisorWithUnload(ack=True),
            unload_after_run=True,
            warmup=0,
            latency_iterations=1,
        )
        assert res_ack.advisor_metadata is not None
        assert res_ack.advisor_metadata.unload_status == "acknowledged"
        rep_ack = format_benchmark_report(res_ack)
        assert "Unload After Run:            True (Unload Requested and Acknowledged; Residency Not Independently Verified)" in rep_ack

        # 3. Unload requested but not acknowledged
        res_not_ack = await run_benchmark_async(
            dataset=[case],
            advisor=AdvisorWithUnload(ack=False),
            unload_after_run=True,
            warmup=0,
            latency_iterations=1,
        )
        assert res_not_ack.advisor_metadata is not None
        assert res_not_ack.advisor_metadata.unload_status == "not_acknowledged"
        rep_not_ack = format_benchmark_report(res_not_ack)
        assert "Unload After Run:            Requested (Not Acknowledged; Residency Not Independently Verified)" in rep_not_ack

    @pytest.mark.anyio
    async def test_34_measured_clear_invocation_reported_in_benchmarks(self) -> None:
        """Report formatting prints the measured clear invocation count in simulated advisor runs."""
        result = await run_benchmark_async(
            simulate_advisor=True,
            warmup=0,
            latency_iterations=1,
        )
        assert result.advisor_clear_invocation_count == 0
        report = format_benchmark_report(result)
        assert "Clear Assessment Invocations:         0  (strictly enforced invariant)" in report or "Clear Assessment Invocations:      0  (strictly enforced invariant)" in report

    @pytest.mark.anyio
    async def test_35_validate_advisor_return_values_async(self) -> None:
        """Invalid return values from async advisor consultations trigger execution_error fallback."""
        assessment = RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.AMBIGUOUS,
            signals=(),
            reason="Ambiguous test prompt",
        )

        invalid_outputs = [
            None,
            "research",
            {"recommended_capability": "research"},
            123,
            object(),
        ]

        class InvalidAsyncAdvisor:
            def __init__(self, output: Any) -> None:
                self.output = output
                self.fallback_recorded: BaseException | None = None

            async def advise(self, prompt: str, assess: RoutingAssessment) -> Any:
                return self.output

            def record_fallback(self, exc: BaseException) -> None:
                self.fallback_recorded = exc

        for invalid_output in invalid_outputs:
            adv = InvalidAsyncAdvisor(invalid_output)
            (
                raw_out,
                rec,
                fallback_used,
                _wall_clock_ns,
                prov_timing,
                error_cat,
            ) = await execute_advisor_consultation_async(adv, "prompt", assessment)

            assert raw_out == AgentCapability.GENERAL
            assert rec is None
            assert fallback_used is True
            assert error_cat == "execution_error"
            assert prov_timing is None
            assert isinstance(adv.fallback_recorded, AdvisorExecutionError)

    def test_36_validate_advisor_return_values_sync(self) -> None:
        """Invalid return values from sync advisor consultations trigger execution_error fallback."""
        assessment = RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.AMBIGUOUS,
            signals=(),
            reason="Ambiguous test prompt",
        )

        invalid_outputs = [
            None,
            "general",
            {"capability": "general"},
            42,
        ]

        class InvalidSyncAdvisor:
            def __init__(self, output: Any) -> None:
                self.output = output
                self.fallback_recorded: BaseException | None = None

            def advise(self, prompt: str, assess: RoutingAssessment) -> Any:
                return self.output

            def record_fallback(self, exc: BaseException) -> None:
                self.fallback_recorded = exc

        for invalid_output in invalid_outputs:
            adv = InvalidSyncAdvisor(invalid_output)
            (
                raw_out,
                rec,
                fallback_used,
                _wall_clock_ns,
                prov_timing,
                error_cat,
            ) = execute_advisor_consultation_sync(adv, "prompt", assessment)

            assert raw_out == AgentCapability.GENERAL
            assert rec is None
            assert fallback_used is True
            assert error_cat == "execution_error"
            assert prov_timing is None
            assert isinstance(adv.fallback_recorded, AdvisorExecutionError)

    def test_37_sync_advisor_coroutine_return_value_is_closed_without_warning(self) -> None:
        """Sync consultation returning a coroutine closes it cleanly and sets execution_error fallback."""
        assessment = RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.AMBIGUOUS,
            signals=(),
            reason="Ambiguous test prompt",
        )

        async def unawaited_target() -> AdvisorRecommendation:
            return AdvisorRecommendation(AgentCapability.RESEARCH, reason="never executed")

        class CoroutineSyncAdvisor:
            def __init__(self) -> None:
                self.fallback_recorded: BaseException | None = None

            def advise(self, prompt: str, assess: RoutingAssessment) -> Any:
                return unawaited_target()

            def record_fallback(self, exc: BaseException) -> None:
                self.fallback_recorded = exc

        adv = CoroutineSyncAdvisor()
        (
            raw_out,
            rec,
            fallback_used,
            _wall_clock_ns,
            prov_timing,
            error_cat,
        ) = execute_advisor_consultation_sync(adv, "prompt", assessment)

        assert raw_out == AgentCapability.GENERAL
        assert rec is None
        assert fallback_used is True
        assert error_cat == "execution_error"
        assert prov_timing is None
        assert isinstance(adv.fallback_recorded, AdvisorExecutionError)

    @pytest.mark.anyio
    async def test_38_failed_attempt_does_not_inherit_stale_provider_timing(self) -> None:
        """Provider timing is attached only on successful recommendations; failed attempts yield None."""
        assessment = RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.AMBIGUOUS,
            signals=(),
            reason="Ambiguous test prompt",
        )

        stale_timing = OllamaProviderTiming(total_duration_ns=1_000_000, load_duration_ns=200_000)

        # 1. Raising advisor retaining stale last_provider_timing
        class StaleTimingRaisingAdvisor:
            def __init__(self) -> None:
                self.last_provider_timing = stale_timing

            async def advise(self, prompt: str, assess: RoutingAssessment) -> Any:
                raise AdvisorExecutionError("Simulated advisor crash")

        raising_adv = StaleTimingRaisingAdvisor()
        (
            _,
            rec,
            fallback_used,
            _,
            prov_timing,
            error_cat,
        ) = await execute_advisor_consultation_async(raising_adv, "prompt", assessment)
        assert fallback_used is True
        assert rec is None
        assert prov_timing is None
        assert error_cat == "execution_error"

        # 2. Invalid return object advisor retaining stale last_provider_timing
        class StaleTimingInvalidReturnAdvisor:
            def __init__(self) -> None:
                self.last_provider_timing = stale_timing

            async def advise(self, prompt: str, assess: RoutingAssessment) -> Any:
                return None  # invalid object

        inv_adv = StaleTimingInvalidReturnAdvisor()
        (
            _,
            rec,
            fallback_used,
            _,
            prov_timing,
            error_cat,
        ) = await execute_advisor_consultation_async(inv_adv, "prompt", assessment)
        assert fallback_used is True
        assert rec is None
        assert prov_timing is None
        assert error_cat == "execution_error"

        # 3. Synchronous consultation retaining stale last_provider_timing on failure
        class StaleTimingSyncAdvisor:
            def __init__(self) -> None:
                self.last_provider_timing = stale_timing

            def advise(self, prompt: str, assess: RoutingAssessment) -> Any:
                raise AdvisorUnavailableError("Ollama unreachable")

        sync_adv = StaleTimingSyncAdvisor()
        (
            _,
            rec,
            fallback_used,
            _,
            prov_timing,
            error_cat,
        ) = execute_advisor_consultation_sync(sync_adv, "prompt", assessment)
        assert fallback_used is True
        assert rec is None
        assert prov_timing is None
        assert error_cat == "unavailable"

    def test_39_advisor_quality_outcome_partition_invariant_raises_runtime_error(self) -> None:
        """Partition invariant violation in compute_advisor_quality_metrics raises RuntimeError."""
        class InconsistentCaseResult:
            actual_clarity = "ambiguous"
            advisor_recommendation = AdvisorRecommendation(AgentCapability.RESEARCH, reason="test")
            fallback_used = False
            effective_provisional_capability = "research"
            is_advisor_clear_invocation = False
            advisor_wall_clock_ns = 100
            advisor_error_category = None
            assessment = None

            def __init__(self) -> None:
                self._calls = 0

            @property
            def is_provisional_correct(self) -> bool:
                self._calls += 1
                # Alternate values across 4 condition checks so none of the 4 branches match
                return self._calls in (1, 4)

            @property
            def is_final_correct(self) -> bool:
                return False

        inconsistent_case = InconsistentCaseResult()
        with pytest.raises(RuntimeError, match="Advisor quality outcome partition invariant violated"):
            compute_advisor_quality_metrics([inconsistent_case])  # type: ignore[list-item]

    @pytest.mark.anyio
    async def test_40_custom_advisor_metrics_derived_from_case_results_not_class_name(self) -> None:
        """Custom advisors are not checked by class name; metrics derive purely from CaseResult evidence."""
        # Custom advisor that always recommends RESEARCH
        class CustomNonSimulatedAdvisor:
            async def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
                return AdvisorRecommendation(AgentCapability.RESEARCH, reason="custom recommendation")

        result = await run_benchmark_async(
            advisor=CustomNonSimulatedAdvisor(),
            warmup=0,
            latency_iterations=1,
        )
        # Default benchmark dataset has 5 ambiguous cases
        assert result.advisor_consultation_count == 5
        assert result.advisor_successful_count == 5
        assert result.advisor_fallback_count == 0
        assert result.advisor_unavailable_count == 0
        assert result.advisor_timeout_count == 0
        assert result.advisor_execution_error_count == 0
        # In the default dataset, 2 ambiguous cases had provisional RESEARCH and 3 had GENERAL
        assert result.advisor_agreement_count == 2
        assert result.advisor_override_count == 3
        assert result.advisor_clear_invocation_count == 0

        # Contrast with standard simulated advisor run:
        sim_result = await run_benchmark_async(
            simulate_advisor=True,
            warmup=0,
            latency_iterations=1,
        )
        assert sim_result.advisor_consultation_count == 5
        assert sim_result.advisor_successful_count == 2
        assert sim_result.advisor_agreement_count == 1
        assert sim_result.advisor_override_count == 1
        assert sim_result.advisor_fallback_count == 3
        assert sim_result.advisor_unavailable_count == 1
        assert sim_result.advisor_timeout_count == 1
        assert sim_result.advisor_execution_error_count == 1
        assert sim_result.advisor_clear_invocation_count == 0
