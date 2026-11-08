"""Deterministic capability router benchmark runner.

Evaluates capability routing performance against the curated routing evaluation dataset.
Separates correctness evaluation from isolated latency profiling.
Supports Phase B RoutingAssessment evaluation and legacy Phase A baseline reproduction.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from orchestrator.domain.advisor import (
    AdvisorExecutionError,
    AdvisorRecommendation,
    AdvisorTimeoutError,
    AdvisorUnavailableError,
)
from orchestrator.domain.assessment import RoutingAssessment, RoutingClarity
from orchestrator.domain.capability import AgentCapability

from evaluation.routing.advisor_evaluation import (
    COLD_PROBE_PROMPT,
    categorize_advisor_error,
    compute_advisor_latency_stats,
    compute_advisor_quality_metrics,
    execute_advisor_consultation_async,
    execute_advisor_consultation_sync,
    execute_cold_probe,
    execute_unload_after_run,
)
from evaluation.routing.models import (
    AdvisorFailureCategory,
    AdvisorLatencyStats,
    AdvisorRunMetadata,
    BenchmarkResult,
    CapabilityType,
    CaseResult,
    ClarityConfusionMatrix,
    ColdProbeOutcome,
    ConfusionMatrix,
    EvaluationCase,
    LatencyStats,
    UnloadStatus,
)

__all__ = [
    "COLD_PROBE_PROMPT",
    "build_parser",
    "categorize_advisor_error",
    "compute_advisor_latency_stats",
    "execute_cold_probe",
    "format_benchmark_report",
    "load_dataset",
    "main",
    "measure_latency",
    "run_benchmark",
    "run_benchmark_async",
    "validate_cli_args",
]

DEFAULT_DATASET_PATH = Path(__file__).parent / "dataset.json"
VALIDATION_DATASET_PATH = Path(__file__).parent / "validation_dataset.json"


@runtime_checkable
class EvaluationCapabilityRouter(Protocol):
    """Protocol for capability routers evaluated by the benchmark."""

    def route(self, prompt: str) -> object:
        """Route a prompt to an agent capability."""
        ...


class SimulatedPlumbingAdvisor:
    """Deterministic simulated routing advisor used strictly for plumbing validation in benchmarks.

    Exercises all hybrid routing plumbing branches (agreement, change/override,
    unavailable fallback, timeout fallback, execution-error fallback) in a deterministic cycle.
    """

    def __init__(self, scenario: str = "cycle") -> None:
        self.consultation_count: int = 0
        self.agreement_count: int = 0
        self.override_count: int = 0
        self.fallback_count: int = 0
        self.unavailable_count: int = 0
        self.timeout_count: int = 0
        self.execution_error_count: int = 0
        self.scenario = scenario

    def advise(
        self, prompt: str, assessment: RoutingAssessment
    ) -> AdvisorRecommendation:
        if assessment.clarity == RoutingClarity.CLEAR:
            raise RuntimeError(
                f"Protocol violation: Advisor was invoked on a CLEAR assessment for prompt: {prompt!r}"
            )

        self.consultation_count += 1
        step = (self.consultation_count - 1) % 5 if self.scenario == "cycle" else {
            "agree": 0,
            "override": 1,
            "unavailable": 2,
            "timeout": 3,
            "error": 4,
        }.get(self.scenario, 0)

        if step == 0:
            self.agreement_count += 1
            recommended = assessment.provisional_capability
            reason = f"Simulated advisor confirms provisional {recommended.value} capability"
            return AdvisorRecommendation(recommended_capability=recommended, reason=reason)
        elif step == 1:
            self.override_count += 1
            recommended = (
                AgentCapability.GENERAL
                if assessment.provisional_capability == AgentCapability.RESEARCH
                else AgentCapability.RESEARCH
            )
            reason = f"Simulated advisor changes capability to {recommended.value}"
            return AdvisorRecommendation(recommended_capability=recommended, reason=reason)
        elif step == 2:
            raise AdvisorUnavailableError("Simulated advisor service unavailable (connection refused)")
        elif step == 3:
            raise AdvisorTimeoutError("Simulated advisor request deadline exceeded (timeout)")
        else:
            raise AdvisorExecutionError("Simulated advisor execution failure (internal error)")

    def record_fallback(self, exc: Exception) -> None:
        """Record fallback metric when an expected error is caught by caller."""
        self.fallback_count += 1
        if isinstance(exc, AdvisorUnavailableError):
            self.unavailable_count += 1
        elif isinstance(exc, (AdvisorTimeoutError, TimeoutError)):
            self.timeout_count += 1
        else:
            self.execution_error_count += 1


def get_default_router(legacy_v2: bool = False) -> EvaluationCapabilityRouter:
    """Instantiate the default deterministic router or the legacy v2 baseline router."""
    if legacy_v2:
        from evaluation.routing.legacy_v2 import LegacyRoutingV2Router

        return LegacyRoutingV2Router()

    from orchestrator.domain.router import DeterministicCapabilityRouter

    return DeterministicCapabilityRouter()


def normalize_capability(output: object) -> CapabilityType:
    """Normalize and validate router capability output to 'general' or 'research'.

    Args:
        output: Raw output from a router's route() method.

    Returns:
        CapabilityType: Either 'general' or 'research'.

    Raises:
        ValueError: If output does not map to a supported capability.
    """
    if hasattr(output, "value") and isinstance(output.value, str):
        candidate = output.value.strip().lower()
    elif isinstance(output, str):
        candidate = output.strip().lower()
    else:
        candidate = str(output).strip().lower()

    if candidate == "general":
        return "general"
    if candidate == "research":
        return "research"

    raise ValueError(
        f"Unsupported router capability output: {output!r}. "
        "Expected 'general' or 'research'."
    )


def load_dataset(path: Path | str | None = None) -> list[EvaluationCase]:
    """Load evaluation cases from JSON dataset."""
    target_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {target_path}")

    with target_path.open("r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    cases: list[EvaluationCase] = []
    for item in raw_cases:
        expected = normalize_capability(item["expected_capability"])
        cases.append(
            EvaluationCase(
                id=item["id"],
                prompt=item["prompt"],
                expected_capability=expected,
                clarity=item["clarity"],
                rationale=item["rationale"],
                category=item["category"],
                tags=item.get("tags", []),
            )
        )
    return cases


def _compute_percentile(sorted_values: list[float], percentile: float) -> float:
    """Compute the percentile value from a sorted list of floats."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = int(k)
    c = f + 1
    if c < len(sorted_values):
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])
    return sorted_values[f]


def _compute_metrics(
    tp: int, fp: int, tn: int, fn: int
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 given true/false positives and negatives."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _prepare_case_provisional(
    router: EvaluationCapabilityRouter, case: EvaluationCase
) -> tuple[RoutingAssessment | None, tuple[str, ...], CapabilityType]:
    """Execute router assessment or route() and derive provisional capability and signals."""
    assessment: RoutingAssessment | None = None
    matched_rules: tuple[str, ...] = ()
    raw_provisional: object = None

    if hasattr(router, "assess") and callable(router.assess):
        res = router.assess(case.prompt)
        if isinstance(res, RoutingAssessment):
            assessment = res
            matched_rules = tuple(s.signal_type.value for s in assessment.signals)
            raw_provisional = assessment.provisional_capability
        else:
            raw_provisional = router.route(case.prompt)
    else:
        raw_provisional = router.route(case.prompt)

    if assessment is None and hasattr(router, "detect_intent"):
        intent = router.detect_intent(case.prompt)
        if intent is not None:
            matched_rules = (intent.value if hasattr(intent, "value") else str(intent),)

    if raw_provisional is None:
        raise ValueError(
            f"Router {type(router).__name__} failed to produce a provisional capability for case '{case.id}'."
        )

    provisional_cap = normalize_capability(raw_provisional)
    return assessment, matched_rules, provisional_cap


def compute_evaluation_matrices(
    case_results: list[CaseResult],
) -> tuple[ConfusionMatrix, dict[str, float]]:
    """Compute final capability ConfusionMatrix and category accuracy breakdown."""
    true_research = sum(
        1
        for r in case_results
        if r.case.expected_capability == "research" and r.final_capability == "research"
    )
    false_research = sum(
        1
        for r in case_results
        if r.case.expected_capability == "general" and r.final_capability == "research"
    )
    true_general = sum(
        1
        for r in case_results
        if r.case.expected_capability == "general" and r.final_capability == "general"
    )
    false_general = sum(
        1
        for r in case_results
        if r.case.expected_capability == "research" and r.final_capability == "general"
    )

    matrix = ConfusionMatrix(
        true_research=true_research,
        false_research=false_research,
        true_general=true_general,
        false_general=false_general,
    )

    categories = sorted({r.case.category for r in case_results})
    category_accuracies: dict[str, float] = {}
    for cat in categories:
        cat_results = [r for r in case_results if r.case.category == cat]
        cat_correct = sum(1 for r in cat_results if r.is_final_correct)
        category_accuracies[cat] = (
            cat_correct / len(cat_results) if cat_results else 0.0
        )

    return matrix, category_accuracies


async def evaluate_correctness_async(
    router: EvaluationCapabilityRouter,
    cases: list[EvaluationCase],
    advisor: object = None,
) -> tuple[
    list[CaseResult],
    ConfusionMatrix,
    dict[str, float],
]:
    """Execute the correctness pass asynchronously evaluating each case once.

    Args:
        router: CapabilityRouter under evaluation.
        cases: Labeled test cases.
        advisor: Optional sync or async advisor invoked only on ambiguous assessments.

    Returns:
        Tuple of (case_results, confusion_matrix, category_accuracies).
    """
    case_results: list[CaseResult] = []

    for case in cases:
        assessment, matched_rules, provisional_cap = _prepare_case_provisional(
            router, case
        )
        advisor_rec = None
        fallback_used = False
        advisor_wall_clock_ns = None
        provider_timing = None
        advisor_error_category: AdvisorFailureCategory | None = None
        is_clear_invocation = False

        if advisor is not None and assessment is not None:
            if assessment.clarity == RoutingClarity.AMBIGUOUS:
                (
                    raw_output,
                    advisor_rec,
                    fallback_used,
                    advisor_wall_clock_ns,
                    provider_timing,
                    advisor_error_category,
                ) = await execute_advisor_consultation_async(
                    advisor, case.prompt, assessment
                )
            else:
                # CLEAR assessment: advisor must NOT be invoked
                raw_output = assessment.provisional_capability
        else:
            raw_output = provisional_cap

        final_cap = normalize_capability(raw_output)
        case_results.append(
            CaseResult(
                case=case,
                actual_capability=final_cap,
                provisional_capability=provisional_cap,
                matched_rules=matched_rules,
                assessment=assessment,
                advisor_recommendation=advisor_rec,
                fallback_used=fallback_used,
                advisor_wall_clock_ns=advisor_wall_clock_ns,
                provider_timing=provider_timing,
                advisor_error_category=advisor_error_category,
                is_advisor_clear_invocation=is_clear_invocation,
            )
        )

    matrix, category_accuracies = compute_evaluation_matrices(case_results)
    return case_results, matrix, category_accuracies


def evaluate_correctness(
    router: EvaluationCapabilityRouter,
    cases: list[EvaluationCase],
    advisor: object = None,
) -> tuple[
    list[CaseResult],
    ConfusionMatrix,
    dict[str, float],
]:
    """Execute the correctness pass evaluating each case once.

    Args:
        router: CapabilityRouter under evaluation.
        cases: Labeled test cases.
        advisor: Optional SimulatedPlumbingAdvisor or async advisor invoked on ambiguous assessments.

    Returns:
        Tuple of (case_results, confusion_matrix, category_accuracies).
    """
    is_async_advisor = advisor is not None and (
        inspect.iscoroutinefunction(getattr(advisor, "advise", None))
    )
    if is_async_advisor:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            raise RuntimeError(
                "evaluate_correctness was invoked with an async advisor inside a running event loop. "
                "Use await evaluate_correctness_async(...) instead."
            )
        return asyncio.run(evaluate_correctness_async(router, cases, advisor=advisor))

    case_results: list[CaseResult] = []

    for case in cases:
        assessment, matched_rules, provisional_cap = _prepare_case_provisional(
            router, case
        )
        advisor_rec = None
        fallback_used = False
        advisor_wall_clock_ns = None
        provider_timing = None
        advisor_error_category: AdvisorFailureCategory | None = None
        is_clear_invocation = False

        if advisor is not None and assessment is not None:
            if assessment.clarity == RoutingClarity.AMBIGUOUS:
                (
                    raw_output,
                    advisor_rec,
                    fallback_used,
                    advisor_wall_clock_ns,
                    provider_timing,
                    advisor_error_category,
                ) = execute_advisor_consultation_sync(advisor, case.prompt, assessment)
            else:
                raw_output = assessment.provisional_capability
        else:
            raw_output = provisional_cap

        final_cap = normalize_capability(raw_output)
        case_results.append(
            CaseResult(
                case=case,
                actual_capability=final_cap,
                provisional_capability=provisional_cap,
                matched_rules=matched_rules,
                assessment=assessment,
                advisor_recommendation=advisor_rec,
                fallback_used=fallback_used,
                advisor_wall_clock_ns=advisor_wall_clock_ns,
                provider_timing=provider_timing,
                advisor_error_category=advisor_error_category,
                is_advisor_clear_invocation=is_clear_invocation,
            )
        )

    matrix, category_accuracies = compute_evaluation_matrices(case_results)
    return case_results, matrix, category_accuracies


def measure_latency(
    router: EvaluationCapabilityRouter,
    prompts: list[str],
    warmup: int,
    latency_iterations: int,
) -> LatencyStats:
    """Execute the latency profiling pass repeatedly measuring router.route(prompt).

    Excludes dataset loading, report formatting, detect_intent/assessment analysis, and terminal output.

    Args:
        router: CapabilityRouter under test.
        prompts: Sequence of prompts to evaluate.
        warmup: Number of unmeasured warmup passes over all prompts.
        latency_iterations: Number of measured repetitions over all prompts.

    Returns:
        LatencyStats with timing distribution and measurement metadata.
    """
    for _ in range(max(0, warmup)):
        for prompt in prompts:
            _ = router.route(prompt)

    latencies_ns: list[int] = []

    for _ in range(max(1, latency_iterations)):
        for prompt in prompts:
            t0 = time.perf_counter_ns()
            _ = router.route(prompt)
            t1 = time.perf_counter_ns()
            latencies_ns.append(max(0, t1 - t0))

    latencies_us = sorted(ns / 1_000.0 for ns in latencies_ns)
    total_samples = len(latencies_us)

    return LatencyStats(
        warmup_iterations=warmup,
        measured_iterations=latency_iterations,
        sample_count=total_samples,
        mean_us=sum(latencies_us) / total_samples if total_samples > 0 else 0.0,
        p50_us=_compute_percentile(latencies_us, 50.0),
        p90_us=_compute_percentile(latencies_us, 90.0),
        p95_us=_compute_percentile(latencies_us, 95.0),
        p99_us=_compute_percentile(latencies_us, 99.0),
        min_us=latencies_us[0] if latencies_us else 0.0,
        max_us=latencies_us[-1] if latencies_us else 0.0,
        total_ns=sum(latencies_ns),
    )


def _build_benchmark_result(
    active_router: EvaluationCapabilityRouter,
    cases: list[EvaluationCase],
    case_results: list[CaseResult],
    matrix: ConfusionMatrix,
    category_accuracies: dict[str, float],
    latency_stats: LatencyStats,
    dataset_name: str,
    simulate_advisor: bool,
    advisor: Any,
    advisor_provider: str | None,
    advisor_metadata: AdvisorRunMetadata | None,
    advisor_latency: AdvisorLatencyStats | None,
) -> BenchmarkResult:
    """Assemble aggregate BenchmarkResult metrics from evaluated case results."""
    total_cases = len(cases)
    clear_cases = [r for r in case_results if r.case.clarity == "clear"]
    ambiguous_cases = [r for r in case_results if r.case.clarity == "ambiguous"]

    # Final post-advisor metrics
    overall_correct = sum(1 for r in case_results if r.is_final_correct)
    clear_correct = sum(1 for r in clear_cases if r.is_final_correct)
    ambiguous_correct = sum(1 for r in ambiguous_cases if r.is_final_correct)

    overall_acc = overall_correct / total_cases if total_cases > 0 else 0.0
    clear_acc = clear_correct / len(clear_cases) if clear_cases else 0.0
    ambiguous_acc = ambiguous_correct / len(ambiguous_cases) if ambiguous_cases else 0.0

    # Deterministic provisional metrics (pre-advisor)
    prov_overall_correct = sum(1 for r in case_results if r.is_provisional_correct)
    prov_clear_correct = sum(1 for r in clear_cases if r.is_provisional_correct)
    prov_ambiguous_correct = sum(1 for r in ambiguous_cases if r.is_provisional_correct)

    prov_overall_acc = prov_overall_correct / total_cases if total_cases > 0 else 0.0
    prov_clear_acc = prov_clear_correct / len(clear_cases) if clear_cases else 0.0
    prov_ambiguous_acc = prov_ambiguous_correct / len(ambiguous_cases) if ambiguous_cases else 0.0

    # Provisional category accuracies
    categories = sorted({r.case.category for r in case_results})
    prov_category_accuracies: dict[str, float] = {}
    for cat in categories:
        cat_results = [r for r in case_results if r.case.category == cat]
        cat_correct = sum(1 for r in cat_results if r.is_provisional_correct)
        prov_category_accuracies[cat] = (
            cat_correct / len(cat_results) if cat_results else 0.0
        )

    # Provisional confusion matrix
    prov_true_research = sum(
        1
        for r in case_results
        if r.case.expected_capability == "research"
        and r.effective_provisional_capability == "research"
    )
    prov_false_research = sum(
        1
        for r in case_results
        if r.case.expected_capability == "general"
        and r.effective_provisional_capability == "research"
    )
    prov_true_general = sum(
        1
        for r in case_results
        if r.case.expected_capability == "general"
        and r.effective_provisional_capability == "general"
    )
    prov_false_general = sum(
        1
        for r in case_results
        if r.case.expected_capability == "research"
        and r.effective_provisional_capability == "general"
    )
    prov_matrix = ConfusionMatrix(
        true_research=prov_true_research,
        false_research=prov_false_research,
        true_general=prov_true_general,
        false_general=prov_false_general,
    )

    res_precision, res_recall, res_f1 = _compute_metrics(
        tp=matrix.true_research,
        fp=matrix.false_research,
        tn=matrix.true_general,
        fn=matrix.false_general,
    )

    gen_precision, gen_recall, gen_f1 = _compute_metrics(
        tp=matrix.true_general,
        fp=matrix.false_general,
        tn=matrix.true_research,
        fn=matrix.false_research,
    )

    # Phase B Clarity & Ambiguity Analysis
    has_assessments = any(r.assessment is not None for r in case_results)
    clarity_correct = 0
    clarity_acc = 0.0
    delegation_count = 0
    delegation_rate = 0.0
    clear_assessed_acc = 0.0
    ambiguous_assessed_acc = 0.0
    clarity_matrix: ClarityConfusionMatrix | None = None

    if has_assessments:
        clarity_correct = sum(1 for r in case_results if r.is_clarity_correct)
        clarity_acc = clarity_correct / total_cases if total_cases > 0 else 0.0

        ambiguous_assessed = [r for r in case_results if r.actual_clarity == "ambiguous"]
        clear_assessed = [r for r in case_results if r.actual_clarity == "clear"]

        delegation_count = len(ambiguous_assessed)
        delegation_rate = delegation_count / total_cases if total_cases > 0 else 0.0

        clear_assessed_acc = (
            sum(1 for r in clear_assessed if r.is_final_correct) / len(clear_assessed)
            if clear_assessed
            else 0.0
        )
        ambiguous_assessed_acc = (
            sum(1 for r in ambiguous_assessed if r.is_final_correct) / len(ambiguous_assessed)
            if ambiguous_assessed
            else 0.0
        )

        true_ambiguous = sum(
            1
            for r in case_results
            if r.case.clarity == "ambiguous" and r.actual_clarity == "ambiguous"
        )
        false_ambiguous = sum(
            1
            for r in case_results
            if r.case.clarity == "clear" and r.actual_clarity == "ambiguous"
        )
        true_clear = sum(
            1
            for r in case_results
            if r.case.clarity == "clear" and r.actual_clarity == "clear"
        )
        false_clear = sum(
            1
            for r in case_results
            if r.case.clarity == "ambiguous" and r.actual_clarity == "clear"
        )

        clarity_matrix = ClarityConfusionMatrix(
            true_ambiguous=true_ambiguous,
            false_ambiguous=false_ambiguous,
            true_clear=true_clear,
            false_clear=false_clear,
        )

    # Advisor metrics computation
    quality = compute_advisor_quality_metrics(case_results, advisor)
    router_name = type(active_router).__name__

    return BenchmarkResult(
        router_name=router_name,
        total_cases=total_cases,
        clear_cases=len(clear_cases),
        ambiguous_cases=len(ambiguous_cases),
        overall_correct=overall_correct,
        clear_correct=clear_correct,
        ambiguous_correct=ambiguous_correct,
        overall_accuracy=overall_acc,
        clear_accuracy=clear_acc,
        ambiguous_accuracy=ambiguous_acc,
        research_precision=res_precision,
        research_recall=res_recall,
        research_f1=res_f1,
        general_precision=gen_precision,
        general_recall=gen_recall,
        general_f1=gen_f1,
        confusion_matrix=matrix,
        latency=latency_stats,
        category_accuracies=category_accuracies,
        dataset_name=dataset_name,
        case_results=case_results,
        clarity_correct=clarity_correct,
        clarity_accuracy=clarity_acc,
        advisor_delegation_count=delegation_count,
        advisor_delegation_rate=delegation_rate,
        clear_assessment_accuracy=clear_assessed_acc,
        ambiguous_assessment_accuracy=ambiguous_assessed_acc,
        clarity_confusion_matrix=clarity_matrix,
        simulated_advisor_enabled=simulate_advisor and (advisor is not None),
        advisor_eligible_count=quality.eligible_count,
        advisor_consultation_count=quality.consultation_count,
        advisor_agreement_count=quality.agreement_count,
        advisor_override_count=quality.override_count,
        advisor_fallback_count=quality.fallback_count,
        advisor_unavailable_count=quality.unavailable_count,
        advisor_timeout_count=quality.timeout_count,
        advisor_execution_error_count=quality.execution_error_count,
        provisional_overall_correct=prov_overall_correct,
        provisional_overall_accuracy=prov_overall_acc,
        provisional_clear_correct=prov_clear_correct,
        provisional_clear_accuracy=prov_clear_acc,
        provisional_ambiguous_correct=prov_ambiguous_correct,
        provisional_ambiguous_accuracy=prov_ambiguous_acc,
        provisional_confusion_matrix=prov_matrix,
        provisional_category_accuracies=prov_category_accuracies,
        advisor_provider=advisor_provider,
        advisor_metadata=advisor_metadata,
        advisor_latency=advisor_latency,
        advisor_successful_count=quality.successful_count,
        advisor_correct_change_count=quality.correct_change_count,
        advisor_regression_count=quality.regression_count,
        advisor_preserved_correct_count=quality.preserved_correct_count,
        advisor_preserved_incorrect_count=quality.preserved_incorrect_count,
        advisor_clear_invocation_count=quality.clear_invocation_count,
    )


async def run_benchmark_async(
    router: EvaluationCapabilityRouter | None = None,
    dataset: list[EvaluationCase] | None = None,
    warmup: int = 10,
    latency_iterations: int = 20,
    legacy_v2: bool = False,
    dataset_name: str = "development",
    simulate_advisor: bool = False,
    advisor_provider: str | None = None,
    ollama_model: str | None = None,
    ollama_url: str | None = None,
    ollama_timeout: float | None = None,
    ollama_keep_alive: int | str | None = None,
    cold_probe: bool = False,
    unload_after_run: bool = False,
    advisor: Any = None,
) -> BenchmarkResult:
    """Asynchronous benchmark runner supporting model-backed and simulated advisor evaluation."""
    active_router = (
        router if router is not None else get_default_router(legacy_v2=legacy_v2)
    )
    cases = dataset if dataset is not None else load_dataset()

    if not cases:
        raise ValueError("Cannot run benchmark on an empty dataset.")

    effective_advisor = advisor
    effective_provider = advisor_provider

    if effective_advisor is None:
        if effective_provider == "ollama":
            from evaluation.routing.ollama_advisor import OllamaRoutingAdvisor

            effective_advisor = OllamaRoutingAdvisor(
                model=ollama_model or "",
                base_url=ollama_url or "http://localhost:11434",
                timeout=ollama_timeout if ollama_timeout is not None else 30.0,
                keep_alive=ollama_keep_alive,
            )
        elif simulate_advisor or effective_provider == "simulated":
            effective_advisor = SimulatedPlumbingAdvisor()
            effective_provider = "simulated"

    cold_probe_outcome: ColdProbeOutcome | None = None
    if not cold_probe:
        cold_probe_outcome = ColdProbeOutcome(status="not_requested")
    elif effective_advisor is not None:
        cold_probe_outcome = await execute_cold_probe(effective_advisor)

    # 1. Correctness Pass
    case_results, matrix, category_accuracies = await evaluate_correctness_async(
        active_router, cases, advisor=effective_advisor
    )

    # Post-run cleanup/unload if requested
    unload_status: UnloadStatus = "not_requested"
    if unload_after_run:
        unload_status = "not_acknowledged"
        if effective_advisor is not None:
            unload_status = await execute_unload_after_run(effective_advisor)

    # 2. Latency Pass (deterministic router latency profiling)
    prompts = [c.prompt for c in cases]
    latency_stats = measure_latency(
        router=active_router,
        prompts=prompts,
        warmup=warmup,
        latency_iterations=latency_iterations,
    )

    # Advisor metadata and latency stats
    advisor_metadata = None
    advisor_latency = None
    if effective_advisor is not None:
        advisor_metadata = AdvisorRunMetadata(
            provider=effective_provider if effective_provider is not None else "custom",
            model=getattr(effective_advisor, "model", ollama_model),
            base_url=getattr(effective_advisor, "base_url", ollama_url),
            timeout_seconds=getattr(effective_advisor, "timeout", ollama_timeout),
            keep_alive=getattr(effective_advisor, "keep_alive", ollama_keep_alive),
            prompt_policy="prompt-only C2 (no assessment reason, signals, or ground-truth)",
            schema='strict raw JSON {"capability", "reason"}',
            cold_probe_outcome=cold_probe_outcome,
            unload_after_run=unload_after_run,
            unload_status=unload_status,
        )
        advisor_latency = compute_advisor_latency_stats(
            case_results=case_results,
            cold_probe=cold_probe_outcome,
        )

    return _build_benchmark_result(
        active_router=active_router,
        cases=cases,
        case_results=case_results,
        matrix=matrix,
        category_accuracies=category_accuracies,
        latency_stats=latency_stats,
        dataset_name=dataset_name,
        simulate_advisor=simulate_advisor or effective_provider == "simulated",
        advisor=effective_advisor,
        advisor_provider=effective_provider,
        advisor_metadata=advisor_metadata,
        advisor_latency=advisor_latency,
    )


def run_benchmark(
    router: EvaluationCapabilityRouter | None = None,
    dataset: list[EvaluationCase] | None = None,
    warmup: int = 10,
    latency_iterations: int = 20,
    legacy_v2: bool = False,
    dataset_name: str = "development",
    simulate_advisor: bool = False,
    advisor_provider: str | None = None,
    ollama_model: str | None = None,
    ollama_url: str | None = None,
    ollama_timeout: float | None = None,
    ollama_keep_alive: int | str | None = None,
    cold_probe: bool = False,
    unload_after_run: bool = False,
    advisor: Any = None,
) -> BenchmarkResult:
    """Run capability routing benchmark across the evaluation dataset.

    Separates correctness evaluation from isolated latency profiling.
    """
    is_async = advisor_provider == "ollama" or (
        advisor is not None
        and inspect.iscoroutinefunction(getattr(advisor, "advise", None))
    )
    if is_async:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            raise RuntimeError(
                "run_benchmark was called with an async advisor inside a running event loop. "
                "Use await run_benchmark_async(...) instead."
            )
        return asyncio.run(
            run_benchmark_async(
                router=router,
                dataset=dataset,
                warmup=warmup,
                latency_iterations=latency_iterations,
                legacy_v2=legacy_v2,
                dataset_name=dataset_name,
                simulate_advisor=simulate_advisor,
                advisor_provider=advisor_provider,
                ollama_model=ollama_model,
                ollama_url=ollama_url,
                ollama_timeout=ollama_timeout,
                ollama_keep_alive=ollama_keep_alive,
                cold_probe=cold_probe,
                unload_after_run=unload_after_run,
                advisor=advisor,
            )
        )

    # Pure synchronous execution path (deterministic baseline, --simulate-advisor, or sync custom advisor)
    active_router = (
        router if router is not None else get_default_router(legacy_v2=legacy_v2)
    )
    cases = dataset if dataset is not None else load_dataset()

    if not cases:
        raise ValueError("Cannot run benchmark on an empty dataset.")

    effective_advisor = advisor
    effective_provider = advisor_provider

    if effective_advisor is None and (simulate_advisor or effective_provider == "simulated"):
        effective_advisor = SimulatedPlumbingAdvisor()
        effective_provider = "simulated"

    # 1. Correctness Pass
    case_results, matrix, category_accuracies = evaluate_correctness(
        active_router, cases, advisor=effective_advisor
    )

    # 2. Latency Pass (isolated repeated executions)
    prompts = [c.prompt for c in cases]
    latency_stats = measure_latency(
        router=active_router,
        prompts=prompts,
        warmup=warmup,
        latency_iterations=latency_iterations,
    )

    advisor_metadata = None
    advisor_latency = None
    if effective_advisor is not None:
        advisor_metadata = AdvisorRunMetadata(
            provider=effective_provider if effective_provider is not None else "custom",
            model=getattr(effective_advisor, "model", None),
            base_url=getattr(effective_advisor, "base_url", None),
            timeout_seconds=getattr(effective_advisor, "timeout", None),
            keep_alive=getattr(effective_advisor, "keep_alive", None),
            prompt_policy="simulated plumbing" if effective_provider == "simulated" else "custom",
            schema="N/A",
            cold_probe_outcome=ColdProbeOutcome(status="not_requested"),
            unload_after_run=False,
            unload_status="not_requested",
        )
        advisor_latency = compute_advisor_latency_stats(case_results)

    return _build_benchmark_result(
        active_router=active_router,
        cases=cases,
        case_results=case_results,
        matrix=matrix,
        category_accuracies=category_accuracies,
        latency_stats=latency_stats,
        dataset_name=dataset_name,
        simulate_advisor=simulate_advisor or effective_provider == "simulated",
        advisor=effective_advisor,
        advisor_provider=effective_provider,
        advisor_metadata=advisor_metadata,
        advisor_latency=advisor_latency,
    )


def format_benchmark_report(result: BenchmarkResult) -> str:
    """Format benchmark results into a clear, human-readable terminal report."""
    has_advisor = result.simulated_advisor_enabled or result.advisor_consultation_count > 0

    lines: list[str] = [
        "=" * 78,
        f"ROUTING BENCHMARK REPORT: {result.router_name} [{result.dataset_name.upper()} DATASET]",
        "=" * 78,
        "",
        "SUMMARY METRICS",
        "-" * 78,
        f"Dataset:                     {result.dataset_name.capitalize()} Dataset",
        f"Total Cases:                 {result.total_cases:4d}",
    ]

    if has_advisor:
        lines.extend([
            f"  Clear Cases:               {result.clear_cases:4d}   (Provisional: {result.provisional_clear_accuracy * 100:6.2f}% | Final: {result.clear_accuracy * 100:6.2f}%)",
            f"  Ambiguous Cases:           {result.ambiguous_cases:4d}   (Provisional: {result.provisional_ambiguous_accuracy * 100:6.2f}% | Final: {result.ambiguous_accuracy * 100:6.2f}%)",
            f"Provisional Capability Acc:  {result.provisional_overall_accuracy * 100:6.2f}% ({result.provisional_overall_correct}/{result.total_cases})",
            f"Final Capability Acc:        {result.overall_accuracy * 100:6.2f}% ({result.overall_correct}/{result.total_cases})",
        ])
    else:
        lines.extend([
            f"  Clear Cases:               {result.clear_cases:4d}   (Accuracy: {result.clear_accuracy * 100:6.2f}% | {result.clear_correct}/{result.clear_cases})",
            f"  Ambiguous Cases:           {result.ambiguous_cases:4d}   (Accuracy: {result.ambiguous_accuracy * 100:6.2f}% | {result.ambiguous_correct}/{result.ambiguous_cases})",
            f"Provisional Capability Acc:  {result.provisional_overall_accuracy * 100:6.2f}% ({result.provisional_overall_correct}/{result.total_cases})",
            f"Final Capability Acc:        {result.overall_accuracy * 100:6.2f}% ({result.overall_correct}/{result.total_cases})",
        ])

    if result.clarity_confusion_matrix is not None:
        lines.extend([
            f"Clarity Label Alignment:     {result.clarity_accuracy * 100:6.2f}% ({result.clarity_correct}/{result.total_cases})",
            f"Hypothetical Advisor Rate:   {result.advisor_delegation_rate * 100:6.2f}% ({result.advisor_delegation_count}/{result.total_cases} ambiguous prompts)",
            f"Capability Acc (Assessed CLEAR):     {result.clear_assessment_accuracy * 100:6.2f}%",
        ])
        if has_advisor:
            lines.extend([
                f"Provisional Acc (Assessed AMBIGUOUS): {result.provisional_ambiguous_accuracy * 100:6.2f}%",
                f"Final Acc (Assessed AMBIGUOUS):       {result.ambiguous_assessment_accuracy * 100:6.2f}%",
            ])
        else:
            lines.append(f"Capability Acc (Assessed AMBIGUOUS): {result.ambiguous_assessment_accuracy * 100:6.2f}%")

        lines.extend([
            "",
            "CLARITY CONFUSION MATRIX (Ambiguous = Positive, Clear = Negative)",
            "-" * 78,
            f"True Ambiguous  (TP): {result.clarity_confusion_matrix.true_ambiguous:4d}    False Ambiguous (FP): {result.clarity_confusion_matrix.false_ambiguous:4d}",
            f"False Clear     (FN): {result.clarity_confusion_matrix.false_clear:4d}    True Clear      (TN): {result.clarity_confusion_matrix.true_clear:4d}",
        ])

    if result.simulated_advisor_enabled:
        lines.extend([
            "",
            "SIMULATED ADVISOR PLUMBING VALIDATION (PLUMBING VERIFICATION ONLY - NOT MODEL QUALITY)",
            "-" * 78,
            f"Total Ambiguous Eligible:       {result.advisor_eligible_count:4d}",
            f"Advisor Consultations:          {result.advisor_consultation_count:4d}",
            f"  - Agreements:                 {result.advisor_agreement_count:4d}",
            f"  - Overrides (Changed):        {result.advisor_override_count:4d}",
            f"  - Fallbacks (Preserved):      {result.advisor_fallback_count:4d}",
            f"    * Unavailable:              {result.advisor_unavailable_count:4d}",
            f"    * Timeout:                  {result.advisor_timeout_count:4d}",
            f"    * Execution Error:          {result.advisor_execution_error_count:4d}",
            f"Clear Assessment Invocations:      {result.advisor_clear_invocation_count}  (strictly enforced invariant)",
            "NOTE: Simulation exercises all plumbing paths; outcomes are synthetic.",
        ])
    elif (
        result.advisor_metadata is not None and result.advisor_metadata.provider != "simulated"
    ) or (result.advisor_consultation_count > 0 and not result.simulated_advisor_enabled):
        meta = result.advisor_metadata
        prov_name = meta.provider if meta else (result.advisor_provider or "custom")
        mod_name = meta.model if meta and meta.model else "unknown"
        url_name = meta.base_url if meta and meta.base_url else "unknown"
        timeout_str = (
            f"{meta.timeout_seconds} s"
            if meta and meta.timeout_seconds is not None
            else "unknown"
        )
        keep_alive_str = (
            str(meta.keep_alive)
            if meta and meta.keep_alive is not None
            else "default"
        )
        prompt_pol = (
            meta.prompt_policy
            if meta
            else "prompt-only C2 (no assessment reason, signals, or ground-truth)"
        )
        schema_str = meta.schema if meta else 'strict raw JSON {"capability", "reason"}'
        cold_probe_flag = meta.cold_probe_performed if meta else False
        unload_str = "False (Not Requested)"
        if meta and meta.unload_status == "acknowledged":
            unload_str = "True (Unload Requested and Acknowledged; Residency Not Independently Verified)"
        elif meta and meta.unload_status == "not_acknowledged":
            unload_str = "Requested (Not Acknowledged; Residency Not Independently Verified)"

        lines.extend([
            "",
            "MODEL ADVISOR EXPERIMENTAL CONFIGURATION (C2 PROMPT-ONLY EVALUATION)",
            "-" * 78,
            f"Provider:                    {prov_name}",
            f"Model:                       {mod_name}",
            f"Base URL:                    {url_name}",
            f"Timeout:                     {timeout_str}",
            f"Keep Alive:                  {keep_alive_str}",
            f"Prompt Policy:               {prompt_pol}",
            f"Schema:                      {schema_str}",
            f"Cold Probe Performed:        {cold_probe_flag}",
            f"Unload After Run:            {unload_str}",
            "",
            "ADVISOR QUALITY & DISAMBIGUATION PERFORMANCE (AMBIGUOUS CASES ONLY)",
            "-" * 78,
            f"Total Ambiguous Eligible:       {result.advisor_eligible_count:4d}",
            f"Advisor Consultations:          {result.advisor_consultation_count:4d}",
            f"  - Successful Recommendations: {result.advisor_successful_count:4d}",
            f"    * Agreements:               {result.advisor_agreement_count:4d}",
            f"    * Changes (Overrides):      {result.advisor_override_count:4d}",
            f"      + Correct Changes:        {result.advisor_correct_change_count:4d}  (Provisional wrong, Advisor correct)",
            f"      + Regressions:            {result.advisor_regression_count:4d}  (Provisional correct, Advisor wrong)",
            f"      + Preserved Correct:      {result.advisor_preserved_correct_count:4d}  (Provisional correct, Final correct)",
            f"      + Preserved Incorrect:    {result.advisor_preserved_incorrect_count:4d}  (Provisional wrong, Final wrong)",
            f"  - Fallbacks (Preserved):      {result.advisor_fallback_count:4d}",
            f"    * Unavailable:              {result.advisor_unavailable_count:4d}",
            f"    * Timeout:                  {result.advisor_timeout_count:4d}",
            f"    * Execution Error:          {result.advisor_execution_error_count:4d}",
            f"Clear Assessment Invocations:      {result.advisor_clear_invocation_count}  (strictly enforced invariant)",
        ])

        if result.advisor_successful_count > 0:
            sem_correct = (
                result.advisor_correct_change_count + result.advisor_preserved_correct_count
            )
            sem_acc = (sem_correct / result.advisor_successful_count) * 100.0
            lines.append(
                f"Advisor Semantic Accuracy:      {sem_correct}/{result.advisor_successful_count} ({sem_acc:6.2f}%)"
            )
        else:
            lines.append("Advisor Semantic Accuracy:      N/A (0 successful)")

        lines.extend([
            f"Final Ambiguous Accuracy:       {result.ambiguous_accuracy * 100:6.2f}%",
            f"Deterministic Provisional Acc:  {result.provisional_ambiguous_accuracy * 100:6.2f}%",
        ])

        if result.advisor_latency is not None:
            lat = result.advisor_latency
            lines.extend([
                "",
                "ADVISOR LATENCY PROFILE (DESCRIPTIVE MEASUREMENTS OVER SMALL SAMPLE)",
                "-" * 78,
            ])
            if lat.warm_successful is not None:
                lines.extend([
                    f"  Successful Warm Consultations: {lat.warm_successful.sample_count}",
                    f"  Warm Wall-Clock Mean:          {lat.warm_successful.mean_ns / 1_000_000.0:8.2f} ms",
                    f"  Warm Wall-Clock P50:           {lat.warm_successful.p50_ns / 1_000_000.0:8.2f} ms",
                    f"  Warm Wall-Clock P90:           {lat.warm_successful.p90_ns / 1_000_000.0:8.2f} ms",
                    f"  Warm Wall-Clock P95:           {lat.warm_successful.p95_ns / 1_000_000.0:8.2f} ms",
                ])
                if lat.warm_successful.p99_ns is not None:
                    lines.append(
                        f"  Warm Wall-Clock P99:           {lat.warm_successful.p99_ns / 1_000_000.0:8.2f} ms"
                    )
                else:
                    lines.append(
                        f"  Warm Wall-Clock P99:           N/A (sample count {lat.warm_successful.sample_count} < 100)"
                    )
                lines.extend([
                    f"  Warm Wall-Clock Min:           {lat.warm_successful.min_ns / 1_000_000.0:8.2f} ms",
                    f"  Warm Wall-Clock Max:           {lat.warm_successful.max_ns / 1_000_000.0:8.2f} ms",
                ])
            else:
                lines.append("  Successful Warm Consultations: 0")

            if lat.failed_attempts is not None:
                lines.extend([
                    f"  Failed Attempt Fallbacks:      {lat.failed_attempts.sample_count}",
                    f"  Failed Attempt Mean:           {lat.failed_attempts.mean_ns / 1_000_000.0:8.2f} ms",
                    f"  Failed Attempt Min:            {lat.failed_attempts.min_ns / 1_000_000.0:8.2f} ms",
                    f"  Failed Attempt Max:            {lat.failed_attempts.max_ns / 1_000_000.0:8.2f} ms",
                ])

            if lat.provider_total_duration_mean_ns is not None:
                lines.append(
                    f"  Provider Total Duration Mean:       {lat.provider_total_duration_mean_ns / 1_000_000.0:8.2f} ms"
                )
            if lat.provider_load_duration_mean_ns is not None:
                lines.append(
                    f"  Provider Load Duration Mean:        {lat.provider_load_duration_mean_ns / 1_000_000.0:8.2f} ms"
                )
            if lat.provider_prompt_eval_duration_mean_ns is not None:
                lines.append(
                    f"  Provider Prompt Eval Duration Mean: {lat.provider_prompt_eval_duration_mean_ns / 1_000_000.0:8.2f} ms"
                )
            if lat.provider_eval_duration_mean_ns is not None:
                lines.append(
                    f"  Provider Eval Duration Mean:        {lat.provider_eval_duration_mean_ns / 1_000_000.0:8.2f} ms"
                )

            if lat.cold_probe is not None and lat.cold_probe.status != "not_requested":
                if lat.cold_probe.status == "succeeded" and lat.cold_probe.latency_ns is not None:
                    lines.append(
                        f"  Cold Probe Measurement:             {lat.cold_probe.latency_ns / 1_000_000.0:8.2f} ms ({lat.cold_probe.label})"
                    )
                    if lat.cold_probe.provider_total_duration_ns is not None:
                        lines.append(
                            f"    * Cold Provider Total Duration:   {lat.cold_probe.provider_total_duration_ns / 1_000_000.0:8.2f} ms"
                        )
                    if lat.cold_probe.provider_load_duration_ns is not None:
                        lines.append(
                            f"    * Cold Provider Load Duration:    {lat.cold_probe.provider_load_duration_ns / 1_000_000.0:8.2f} ms"
                        )
                elif lat.cold_probe.status == "failed":
                    fail_cat = lat.cold_probe.failure_category or "unknown"
                    lines.append(
                        f"  Cold Probe Measurement:             FAILED ({fail_cat}) - {lat.cold_probe.label}"
                    )

    lines.extend([
        "",
        "CAPABILITY CONFUSION MATRIX (Research = Positive, General = Negative)",
        "-" * 78,
        f"True Research   (TP): {result.confusion_matrix.true_research:4d}    False Research  (FP): {result.confusion_matrix.false_research:4d}",
        f"False General   (FN): {result.confusion_matrix.false_general:4d}    True General    (TN): {result.confusion_matrix.true_general:4d}",
        "",
        "CLASS-LEVEL PERFORMANCE",
        "-" * 78,
        f"Research Capability -> Precision: {result.research_precision:6.2f} | Recall: {result.research_recall:6.2f} | F1: {result.research_f1:6.2f}",
        f"General Capability  -> Precision: {result.general_precision:6.2f} | Recall: {result.general_recall:6.2f} | F1: {result.general_f1:6.2f}",
        "",
        "ACCURACY BY CATEGORY",
        "-" * 78,
    ])

    for cat, acc in sorted(result.category_accuracies.items()):
        cat_count = sum(1 for r in result.case_results if r.case.category == cat)
        cat_correct = sum(1 for r in result.case_results if r.case.category == cat and r.is_correct)
        lines.append(f"  {cat:<30} {acc * 100:6.2f}% ({cat_correct}/{cat_count})")

    lines.extend([
        "",
        "LATENCY PROFILE (microseconds - monotonic perf_counter_ns)",
        "-" * 78,
        f"  Warmup Passes:         {result.latency.warmup_iterations}",
        f"  Measured Iterations:   {result.latency.measured_iterations} passes over prompt set",
        f"  Total Latency Samples: {result.latency.sample_count}",
        f"  Mean:                  {result.latency.mean_us:8.2f} µs",
        f"  Median (P50):          {result.latency.p50_us:8.2f} µs",
        f"  P90:                   {result.latency.p90_us:8.2f} µs",
        f"  P95:                   {result.latency.p95_us:8.2f} µs",
        f"  P99:                   {result.latency.p99_us:8.2f} µs",
        f"  Min:                   {result.latency.min_us:8.2f} µs",
        f"  Max:                   {result.latency.max_us:8.2f} µs",
        "",
    ])

    # Final Capability Disagreements
    disagreements = [r for r in result.case_results if not r.is_final_correct]
    if disagreements:
        lines.extend([
            f"CAPABILITY DISAGREEMENTS / ERRORS (FINAL ROUTING) ({len(disagreements)} cases)",
            "-" * 78,
        ])
        for r in disagreements:
            rule_str = f" [Signals: {', '.join(r.matched_rules)}]" if r.matched_rules else ""
            reason_str = f"\n  Assessment Reason: {r.assessment.reason}" if r.assessment is not None else ""
            advisor_str = ""
            if r.advisor_recommendation is not None:
                rec_cap = normalize_capability(r.advisor_recommendation.recommended_capability)
                prov_cap = r.effective_provisional_capability
                if rec_cap == prov_cap:
                    action_label = f"Agreed with {rec_cap.upper()}"
                else:
                    action_label = f"Changed to {rec_cap.upper()}"
                advisor_str = f"\n  Advisor Action:    {action_label} ({r.advisor_recommendation.reason})"
            elif r.fallback_used:
                advisor_str = "\n  Advisor Action:    Fallback used (provisional capability preserved)"

            diff_note = ""
            if r.is_provisional_correct and not r.is_final_correct:
                diff_note = "\n  Disagreement Source: Deterministic assessment was CORRECT; advisor recommendation changed final routing."
            elif not r.is_provisional_correct and not r.is_final_correct:
                diff_note = "\n  Disagreement Source: Deterministic provisional assessment also disagreed with expected ground truth."

            lines.append(
                f"[{r.case.id}] ({r.case.category}) clarity={r.case.clarity}\n"
                f"  Prompt:            \"{r.case.prompt[:70]}{'...' if len(r.case.prompt) > 70 else ''}\"\n"
                f"  Expected:          {r.case.expected_capability.upper()}\n"
                f"  Provisional:       {r.effective_provisional_capability.upper()}{rule_str}\n"
                f"  Final:             {r.final_capability.upper()}{advisor_str}{diff_note}{reason_str}\n"
                f"  Rationale:         {r.case.rationale}\n"
            )
    else:
        lines.append("No final capability classification disagreements found.\n")

    # Deterministic Provisional Disagreements (Pre-advisor)
    prov_disagreements = [r for r in result.case_results if not r.is_provisional_correct]
    if prov_disagreements and has_advisor:
        lines.extend([
            f"PROVISIONAL CAPABILITY DISAGREEMENTS (PRE-ADVISOR) ({len(prov_disagreements)} cases)",
            "-" * 78,
        ])
        for r in prov_disagreements:
            rule_str = f" [Signals: {', '.join(r.matched_rules)}]" if r.matched_rules else ""
            reason_str = f"\n  Assessment Reason: {r.assessment.reason}" if r.assessment is not None else ""
            lines.append(
                f"[{r.case.id}] ({r.case.category}) clarity={r.case.clarity}\n"
                f"  Prompt:            \"{r.case.prompt[:70]}{'...' if len(r.case.prompt) > 70 else ''}\"\n"
                f"  Expected:          {r.case.expected_capability.upper()}\n"
                f"  Provisional:       {r.effective_provisional_capability.upper()}{rule_str}{reason_str}\n"
                f"  Rationale:         {r.case.rationale}\n"
            )

    # Clarity Disagreements (where router clarity != human dataset clarity)
    if result.clarity_confusion_matrix is not None:
        clarity_disagreements = [r for r in result.case_results if not r.is_clarity_correct]
        if clarity_disagreements:
            lines.extend([
                f"CLARITY DISAGREEMENTS ({len(clarity_disagreements)} cases)",
                "-" * 78,
            ])
            for r in clarity_disagreements:
                actual_clarity = r.actual_clarity or "unknown"
                reason_str = f"\n  Assessment Reason: {r.assessment.reason}" if r.assessment is not None else ""
                lines.append(
                    f"[{r.case.id}] ({r.case.category})\n"
                    f"  Prompt:   \"{r.case.prompt[:70]}{'...' if len(r.case.prompt) > 70 else ''}\"\n"
                    f"  Dataset Clarity: {r.case.clarity.upper()} | Assessed Clarity: {actual_clarity.upper()}{reason_str}\n"
                )
        else:
            lines.append("No clarity disagreements found.\n")

    lines.append("=" * 78)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run capability routing benchmark and latency evaluation.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to routing evaluation dataset JSON file (defaults to development dataset).",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        default=False,
        help="Evaluate against the validation dataset (evaluation/routing/validation_dataset.json).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Evaluate both development and validation datasets sequentially.",
    )
    parser.add_argument(
        "--latency-iterations",
        type=int,
        default=20,
        help="Number of repeated measurement iterations over all prompts in the latency pass (default: 20).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of unmeasured warmup iterations over all prompts (default: 10).",
    )
    parser.add_argument(
        "--baseline-v2",
        action="store_true",
        default=False,
        help="Execute the legacy Routing v2 baseline router for comparison.",
    )
    parser.add_argument(
        "--simulate-advisor",
        action="store_true",
        default=False,
        help="Simulate deterministic routing advisor plumbing for ambiguous cases only.",
    )
    parser.add_argument(
        "--advisor-provider",
        choices=["simulated", "ollama"],
        default=None,
        help="Advisor provider for ambiguous cases ('simulated' or 'ollama').",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Ollama model name (required when --advisor-provider ollama is selected).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Ollama server base URL (default: http://localhost:11434 when provider is ollama).",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=None,
        help="Ollama request timeout in seconds (default: 30.0 when provider is ollama).",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        type=str,
        default=None,
        help="Ollama keep_alive parameter (e.g. '5m', '0' for immediate unload, or integer seconds).",
    )
    parser.add_argument(
        "--cold-probe",
        action="store_true",
        default=False,
        help="Execute an isolated, non-scored cold probe request prior to the scored evaluation run.",
    )
    parser.add_argument(
        "--unload-after-run",
        action="store_true",
        default=False,
        help="Send keep_alive=0 to unload the model from memory after benchmark completion.",
    )
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    """Validate CLI flag combinations."""
    if args.simulate_advisor and args.advisor_provider is not None:
        raise ValueError("Cannot combine --simulate-advisor and --advisor-provider")
    if args.advisor_provider == "ollama":
        if not args.ollama_model or not args.ollama_model.strip():
            raise ValueError(
                "--ollama-model is required when --advisor-provider ollama is selected"
            )
        if args.ollama_timeout is not None and args.ollama_timeout <= 0:
            raise ValueError("--ollama-timeout must be greater than zero")
    else:
        ollama_opts: list[str] = []
        if args.ollama_model is not None:
            ollama_opts.append("--ollama-model")
        if args.ollama_url is not None:
            ollama_opts.append("--ollama-url")
        if args.ollama_timeout is not None:
            ollama_opts.append("--ollama-timeout")
        if args.ollama_keep_alive is not None:
            ollama_opts.append("--ollama-keep-alive")
        if args.cold_probe:
            ollama_opts.append("--cold-probe")
        if args.unload_after_run:
            ollama_opts.append("--unload-after-run")
        if ollama_opts:
            raise ValueError(
                f"Ollama options ({', '.join(ollama_opts)}) require --advisor-provider ollama"
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for routing benchmark execution."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_cli_args(args)
    except ValueError as err:
        parser.error(str(err))

    run_kwargs: dict[str, Any] = {
        "warmup": args.warmup,
        "latency_iterations": args.latency_iterations,
        "legacy_v2": args.baseline_v2,
        "simulate_advisor": args.simulate_advisor,
        "advisor_provider": args.advisor_provider,
        "ollama_model": args.ollama_model,
        "ollama_url": args.ollama_url,
        "ollama_timeout": args.ollama_timeout,
        "ollama_keep_alive": args.ollama_keep_alive,
        "cold_probe": args.cold_probe,
        "unload_after_run": args.unload_after_run,
    }

    if args.all:
        dev_cases = load_dataset(DEFAULT_DATASET_PATH)
        dev_result = run_benchmark(
            dataset=dev_cases,
            dataset_name="development",
            **run_kwargs,
        )
        print(format_benchmark_report(dev_result))
        print("\n" + "=" * 78 + "\n")
        val_cases = load_dataset(VALIDATION_DATASET_PATH)
        val_result = run_benchmark(
            dataset=val_cases,
            dataset_name="validation",
            **run_kwargs,
        )
        print(format_benchmark_report(val_result))
        return 0

    if args.validation:
        target_path = VALIDATION_DATASET_PATH
        dataset_name = "validation"
    elif args.dataset is not None:
        target_path = args.dataset
        dataset_name = (
            "validation" if "validation" in str(target_path).lower() else "custom"
        )
    else:
        target_path = DEFAULT_DATASET_PATH
        dataset_name = "development"

    cases = load_dataset(target_path)
    result = run_benchmark(
        dataset=cases,
        dataset_name=dataset_name,
        **run_kwargs,
    )
    print(format_benchmark_report(result))
    return 0


if __name__ == "__main__":
    main()
