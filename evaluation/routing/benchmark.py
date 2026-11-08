"""Deterministic capability router benchmark runner.

Evaluates capability routing performance against the curated routing evaluation dataset.
Separates correctness evaluation from isolated latency profiling.
Supports Phase B RoutingAssessment evaluation and legacy Phase A baseline reproduction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from orchestrator.domain.advisor import (
    AdvisorError,
    AdvisorExecutionError,
    AdvisorRecommendation,
    AdvisorTimeoutError,
    AdvisorUnavailableError,
)
from orchestrator.domain.assessment import RoutingAssessment, RoutingClarity
from orchestrator.domain.capability import AgentCapability

from evaluation.routing.models import (
    BenchmarkResult,
    CapabilityType,
    CaseResult,
    ClarityConfusionMatrix,
    ConfusionMatrix,
    EvaluationCase,
    LatencyStats,
)

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


def evaluate_correctness(
    router: EvaluationCapabilityRouter,
    cases: list[EvaluationCase],
    advisor: SimulatedPlumbingAdvisor | None = None,
) -> tuple[
    list[CaseResult],
    ConfusionMatrix,
    dict[str, float],
]:
    """Execute the correctness pass evaluating each case once.

    Args:
        router: CapabilityRouter under evaluation.
        cases: Labeled test cases.
        advisor: Optional SimulatedPlumbingAdvisor invoked on ambiguous assessments.

    Returns:
        Tuple of (case_results, confusion_matrix, category_accuracies).
    """
    case_results: list[CaseResult] = []

    for case in cases:
        assessment = None
        matched_rules: tuple[str, ...] = ()
        advisor_rec = None
        fallback_used = False
        raw_provisional = None

        if hasattr(router, "assess") and callable(router.assess):
            res = router.assess(case.prompt)
            if isinstance(res, RoutingAssessment):
                assessment = res
                matched_rules = tuple(s.signal_type.value for s in assessment.signals)
                raw_provisional = assessment.provisional_capability
                if (
                    advisor is not None
                    and assessment.clarity == RoutingClarity.AMBIGUOUS
                ):
                    try:
                        advisor_rec = advisor.advise(case.prompt, assessment)
                        raw_output = advisor_rec.recommended_capability
                    except (AdvisorError, TimeoutError) as exc:
                        advisor.record_fallback(exc)
                        raw_output = assessment.provisional_capability
                        fallback_used = True
                else:
                    raw_output = assessment.provisional_capability
            else:
                raw_output = router.route(case.prompt)
                raw_provisional = raw_output
        else:
            raw_output = router.route(case.prompt)
            raw_provisional = raw_output

        if assessment is None and hasattr(router, "detect_intent"):
            intent = router.detect_intent(case.prompt)
            if intent is not None:
                matched_rules = (intent.value if hasattr(intent, "value") else str(intent),)

        if raw_provisional is None:
            raise ValueError(
                f"Router {type(router).__name__} failed to produce a provisional capability for case '{case.id}'."
            )

        provisional_cap = normalize_capability(raw_provisional)
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
            )
        )

    # Confusion matrix for final capability (Research = positive, General = negative)
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

    # Category accuracies (final capability)
    categories = sorted({r.case.category for r in case_results})
    category_accuracies: dict[str, float] = {}
    for cat in categories:
        cat_results = [r for r in case_results if r.case.category == cat]
        cat_correct = sum(1 for r in cat_results if r.is_final_correct)
        category_accuracies[cat] = (
            cat_correct / len(cat_results) if cat_results else 0.0
        )

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


def run_benchmark(
    router: EvaluationCapabilityRouter | None = None,
    dataset: list[EvaluationCase] | None = None,
    warmup: int = 10,
    latency_iterations: int = 20,
    legacy_v2: bool = False,
    dataset_name: str = "development",
    simulate_advisor: bool = False,
) -> BenchmarkResult:
    """Run capability routing benchmark across the evaluation dataset.

    Separates correctness evaluation from isolated latency profiling.

    Args:
        router: CapabilityRouter instance (defaults to DeterministicCapabilityRouter or LegacyRoutingV2Router).
        dataset: List of EvaluationCase objects (defaults to loading dataset.json).
        warmup: Number of unmeasured warmup iterations over the prompt set.
        latency_iterations: Number of measured repetitions over the prompt set.
        legacy_v2: If True, uses the legacy Routing v2 baseline router.
        dataset_name: Name of the dataset being evaluated ('development' or 'validation').
        simulate_advisor: If True, wires a deterministic test advisor into ambiguous cases only.

    Returns:
        BenchmarkResult containing detailed accuracy, confusion matrix, clarity analysis, and latency statistics.
    """
    active_router = (
        router if router is not None else get_default_router(legacy_v2=legacy_v2)
    )
    cases = dataset if dataset is not None else load_dataset()

    if not cases:
        raise ValueError("Cannot run benchmark on an empty dataset.")

    advisor = (
        SimulatedPlumbingAdvisor()
        if simulate_advisor
        else None
    )

    # 1. Correctness Pass (evaluates each case once)
    case_results, matrix, category_accuracies = evaluate_correctness(
        active_router, cases, advisor=advisor
    )

    # 2. Latency Pass (isolated repeated executions)
    prompts = [c.prompt for c in cases]
    latency_stats = measure_latency(
        router=active_router,
        prompts=prompts,
        warmup=warmup,
        latency_iterations=latency_iterations,
    )

    # Capability accuracy breakdown
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
        1 for r in case_results if r.case.expected_capability == "research" and r.effective_provisional_capability == "research"
    )
    prov_false_research = sum(
        1 for r in case_results if r.case.expected_capability == "general" and r.effective_provisional_capability == "research"
    )
    prov_true_general = sum(
        1 for r in case_results if r.case.expected_capability == "general" and r.effective_provisional_capability == "general"
    )
    prov_false_general = sum(
        1 for r in case_results if r.case.expected_capability == "research" and r.effective_provisional_capability == "general"
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

    # 3. Phase B Clarity & Ambiguity Analysis
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

        true_ambiguous = sum(1 for r in case_results if r.case.clarity == "ambiguous" and r.actual_clarity == "ambiguous")
        false_ambiguous = sum(1 for r in case_results if r.case.clarity == "clear" and r.actual_clarity == "ambiguous")
        true_clear = sum(1 for r in case_results if r.case.clarity == "clear" and r.actual_clarity == "clear")
        false_clear = sum(1 for r in case_results if r.case.clarity == "ambiguous" and r.actual_clarity == "clear")

        clarity_matrix = ClarityConfusionMatrix(
            true_ambiguous=true_ambiguous,
            false_ambiguous=false_ambiguous,
            true_clear=true_clear,
            false_clear=false_clear,
        )

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
        advisor_eligible_count=len([r for r in case_results if r.actual_clarity == "ambiguous"]),
        advisor_consultation_count=advisor.consultation_count if advisor is not None else 0,
        advisor_agreement_count=advisor.agreement_count if advisor is not None else 0,
        advisor_override_count=advisor.override_count if advisor is not None else 0,
        advisor_fallback_count=advisor.fallback_count if advisor is not None else 0,
        advisor_unavailable_count=advisor.unavailable_count if advisor is not None else 0,
        advisor_timeout_count=advisor.timeout_count if advisor is not None else 0,
        advisor_execution_error_count=advisor.execution_error_count if advisor is not None else 0,
        provisional_overall_correct=prov_overall_correct,
        provisional_overall_accuracy=prov_overall_acc,
        provisional_clear_correct=prov_clear_correct,
        provisional_clear_accuracy=prov_clear_acc,
        provisional_ambiguous_correct=prov_ambiguous_correct,
        provisional_ambiguous_accuracy=prov_ambiguous_acc,
        provisional_confusion_matrix=prov_matrix,
        provisional_category_accuracies=prov_category_accuracies,
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
            "Clear Assessment Invocations:      0  (strictly enforced invariant)",
            "NOTE: Simulation exercises all plumbing paths; outcomes are synthetic.",
        ])

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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for routing benchmark execution."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.all:
        dev_cases = load_dataset(DEFAULT_DATASET_PATH)
        dev_result = run_benchmark(
            dataset=dev_cases,
            warmup=args.warmup,
            latency_iterations=args.latency_iterations,
            legacy_v2=args.baseline_v2,
            dataset_name="development",
            simulate_advisor=args.simulate_advisor,
        )
        print(format_benchmark_report(dev_result))
        print("\n" + "=" * 78 + "\n")
        val_cases = load_dataset(VALIDATION_DATASET_PATH)
        val_result = run_benchmark(
            dataset=val_cases,
            warmup=args.warmup,
            latency_iterations=args.latency_iterations,
            legacy_v2=args.baseline_v2,
            dataset_name="validation",
            simulate_advisor=args.simulate_advisor,
        )
        print(format_benchmark_report(val_result))
        return 0

    if args.validation:
        target_path = VALIDATION_DATASET_PATH
        dataset_name = "validation"
    elif args.dataset is not None:
        target_path = args.dataset
        dataset_name = "validation" if "validation" in str(target_path).lower() else "custom"
    else:
        target_path = DEFAULT_DATASET_PATH
        dataset_name = "development"

    cases = load_dataset(target_path)
    result = run_benchmark(
        dataset=cases,
        warmup=args.warmup,
        latency_iterations=args.latency_iterations,
        legacy_v2=args.baseline_v2,
        dataset_name=dataset_name,
        simulate_advisor=args.simulate_advisor,
    )
    print(format_benchmark_report(result))
    return 0


if __name__ == "__main__":
    main()
