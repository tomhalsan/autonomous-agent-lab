"""C2 Model-backed routing advisor evaluation and telemetry helpers.

Provides cohesive utilities for:
- advisor error categorization
- cold-probe execution with recommendation validation
- advisor latency distribution and percentile calculations
- advisor consultation execution (sync and async)
- advisor quality and disambiguation outcome aggregation
"""

from __future__ import annotations

import contextlib
import inspect
import time
from dataclasses import dataclass
from typing import Any

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
    AdvisorFailureCategory,
    AdvisorLatencyStats,
    CaseResult,
    ColdProbeOutcome,
    LatencyDistribution,
    OllamaProviderTiming,
    UnloadStatus,
)

COLD_PROBE_PROMPT = (
    "Explain the fundamental difference between a process and a thread in modern operating systems."
)
P99_MIN_SAMPLE_COUNT = 100


def categorize_advisor_error(exc: Exception) -> AdvisorFailureCategory:
    """Map caught advisor exception into a typed failure category."""
    if isinstance(exc, AdvisorUnavailableError):
        return "unavailable"
    if isinstance(exc, (AdvisorTimeoutError, TimeoutError)):
        return "timeout"
    return "execution_error"


def _compute_percentile(sorted_samples: list[int | float], percentile: float) -> float:
    """Compute percentile using linear interpolation on 1-based rank."""
    if not sorted_samples:
        return 0.0
    n = len(sorted_samples)
    if n == 1:
        return float(sorted_samples[0])

    rank = (percentile / 100.0) * (n - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, n - 1)
    fraction = rank - lower_idx

    return float(
        sorted_samples[lower_idx]
        + fraction * (sorted_samples[upper_idx] - sorted_samples[lower_idx])
    )


def _compute_distribution(samples: list[int]) -> LatencyDistribution | None:
    """Compute distribution statistics for a sample of latency nanoseconds.

    P99 is computed only when sample count meets the minimum threshold (>= 100).
    """
    if not samples:
        return None
    sorted_samples = sorted(samples)
    sample_count = len(sorted_samples)
    p99_val = (
        _compute_percentile(sorted_samples, 99.0)
        if sample_count >= P99_MIN_SAMPLE_COUNT
        else None
    )
    return LatencyDistribution(
        sample_count=sample_count,
        mean_ns=sum(sorted_samples) / sample_count,
        p50_ns=_compute_percentile(sorted_samples, 50.0),
        p90_ns=_compute_percentile(sorted_samples, 90.0),
        p95_ns=_compute_percentile(sorted_samples, 95.0),
        p99_ns=p99_val,
        min_ns=sorted_samples[0],
        max_ns=sorted_samples[-1],
    )


def compute_advisor_latency_stats(
    case_results: list[CaseResult],
    cold_probe: ColdProbeOutcome | None = None,
) -> AdvisorLatencyStats | None:
    """Compute summary latency statistics for scored advisor consultations."""
    successful_samples = [
        r.advisor_wall_clock_ns
        for r in case_results
        if r.advisor_wall_clock_ns is not None and r.advisor_recommendation is not None
    ]
    failed_samples = [
        r.advisor_wall_clock_ns
        for r in case_results
        if r.advisor_wall_clock_ns is not None
        and r.fallback_used
        and r.advisor_recommendation is None
    ]

    if not successful_samples and not failed_samples and cold_probe is None:
        return None

    warm_dist = _compute_distribution(successful_samples)
    failed_dist = _compute_distribution(failed_samples)

    total_durations = [
        r.provider_timing.total_duration_ns
        for r in case_results
        if r.advisor_recommendation is not None
        and r.provider_timing is not None
        and r.provider_timing.total_duration_ns is not None
    ]
    load_durations = [
        r.provider_timing.load_duration_ns
        for r in case_results
        if r.advisor_recommendation is not None
        and r.provider_timing is not None
        and r.provider_timing.load_duration_ns is not None
    ]
    prompt_eval_durations = [
        r.provider_timing.prompt_eval_duration_ns
        for r in case_results
        if r.advisor_recommendation is not None
        and r.provider_timing is not None
        and r.provider_timing.prompt_eval_duration_ns is not None
    ]
    eval_durations = [
        r.provider_timing.eval_duration_ns
        for r in case_results
        if r.advisor_recommendation is not None
        and r.provider_timing is not None
        and r.provider_timing.eval_duration_ns is not None
    ]

    return AdvisorLatencyStats(
        warm_successful=warm_dist,
        failed_attempts=failed_dist,
        provider_total_duration_mean_ns=(
            sum(total_durations) / len(total_durations) if total_durations else None
        ),
        provider_load_duration_mean_ns=(
            sum(load_durations) / len(load_durations) if load_durations else None
        ),
        provider_prompt_eval_duration_mean_ns=(
            sum(prompt_eval_durations) / len(prompt_eval_durations)
            if prompt_eval_durations
            else None
        ),
        provider_eval_duration_mean_ns=(
            sum(eval_durations) / len(eval_durations) if eval_durations else None
        ),
        cold_probe=cold_probe,
    )


async def execute_cold_probe(
    advisor: object,
    probe_prompt: str = COLD_PROBE_PROMPT,
) -> ColdProbeOutcome:
    """Execute evaluation cold-probe, validating recommendation and capturing telemetry.

    Requires:
    - a callable advise method;
    - successful completion;
    - a returned AdvisorRecommendation instance.

    Missing/non-callable advise or an invalid return object produces a failed outcome
    categorized as execution_error without leaking returned content.
    """
    unload_acknowledged = False
    if hasattr(advisor, "unload") and callable(advisor.unload):
        with contextlib.suppress(AdvisorError, TimeoutError, OSError):
            unload_fn = advisor.unload
            if inspect.iscoroutinefunction(unload_fn):
                ack = await unload_fn()
            elif callable(unload_fn):
                ack = unload_fn()
                if inspect.iscoroutine(ack):
                    ack = await ack
            if ack is True:
                unload_acknowledged = True

    label = (
        "Unload Requested and Acknowledged; Residency Not Independently Verified"
        if unload_acknowledged
        else "Initial Probe (Residency Not Independently Verified)"
    )

    advise_fn = getattr(advisor, "advise", None)
    if not callable(advise_fn):
        return ColdProbeOutcome(
            status="failed",
            latency_ns=None,
            failure_category="execution_error",
            unload_acknowledged=unload_acknowledged,
            label=label,
        )

    probe_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.GENERAL,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Cold probe baseline evaluation inquiry",
    )

    t0 = time.perf_counter_ns()
    rec: object = None
    probe_failure_category: AdvisorFailureCategory | None = None
    probe_pt: OllamaProviderTiming | None = None

    try:
        if inspect.iscoroutinefunction(advise_fn):
            rec = await advise_fn(probe_prompt, probe_assessment)
        else:
            res = advise_fn(probe_prompt, probe_assessment)
            rec = await res if inspect.iscoroutine(res) else res

        if not isinstance(rec, AdvisorRecommendation):
            probe_failure_category = "execution_error"
        else:
            pt_attr = getattr(advisor, "last_provider_timing", None)
            if isinstance(pt_attr, OllamaProviderTiming):
                probe_pt = pt_attr
    except (AdvisorError, TimeoutError, OSError) as exc:
        probe_failure_category = categorize_advisor_error(exc)
    except (RuntimeError, ValueError, TypeError, AttributeError):
        probe_failure_category = "execution_error"
    finally:
        t1 = time.perf_counter_ns()
        elapsed_ns = max(0, t1 - t0)

    if probe_failure_category is None and isinstance(rec, AdvisorRecommendation):
        return ColdProbeOutcome(
            status="succeeded",
            latency_ns=elapsed_ns,
            failure_category=None,
            unload_acknowledged=unload_acknowledged,
            label=label,
            provider_total_duration_ns=probe_pt.total_duration_ns if probe_pt else None,
            provider_load_duration_ns=probe_pt.load_duration_ns if probe_pt else None,
        )
    return ColdProbeOutcome(
        status="failed",
        latency_ns=None,
        failure_category=probe_failure_category or "execution_error",
        unload_acknowledged=unload_acknowledged,
        label=label,
        provider_total_duration_ns=None,
        provider_load_duration_ns=None,
    )


async def execute_unload_after_run(advisor: object) -> UnloadStatus:
    """Execute post-run model unload if supported by the advisor.

    Returns UnloadStatus indicating whether unload was acknowledged.
    """
    if hasattr(advisor, "unload") and callable(advisor.unload):
        with contextlib.suppress(AdvisorError, TimeoutError, OSError):
            unload_fn = advisor.unload
            if inspect.iscoroutinefunction(unload_fn):
                ack = await unload_fn()
            elif callable(unload_fn):
                ack = unload_fn()
                if inspect.iscoroutine(ack):
                    ack = await ack
            if ack is True:
                return "acknowledged"
    return "not_acknowledged"


async def execute_advisor_consultation_async(
    advisor: object,
    prompt: str,
    assessment: RoutingAssessment,
) -> tuple[
    AgentCapability,
    AdvisorRecommendation | None,
    bool,
    int,
    OllamaProviderTiming | None,
    AdvisorFailureCategory | None,
]:
    """Execute asynchronous advisor consultation for an ambiguous assessment."""
    t0 = time.perf_counter_ns()
    advisor_rec: AdvisorRecommendation | None = None
    fallback_used = False
    advisor_error_category: AdvisorFailureCategory | None = None
    provider_timing: OllamaProviderTiming | None = None

    try:
        advise_fn = getattr(advisor, "advise", None)
        if not callable(advise_fn):
            raise TypeError("Advisor does not have a callable advise method")

        if inspect.iscoroutinefunction(advise_fn):
            raw_res = await advise_fn(prompt, assessment)
        else:
            adv_res = advise_fn(prompt, assessment)
            raw_res = await adv_res if inspect.iscoroutine(adv_res) else adv_res

        if not isinstance(raw_res, AdvisorRecommendation):
            record_fb = getattr(advisor, "record_fallback", None)
            if callable(record_fb):
                record_fb(
                    AdvisorExecutionError("Advisor returned invalid recommendation object")
                )
            raw_output = assessment.provisional_capability
            advisor_rec = None
            fallback_used = True
            advisor_error_category = "execution_error"
        else:
            advisor_rec = raw_res
            raw_output = advisor_rec.recommended_capability
    except (AdvisorError, TimeoutError, OSError) as exc:
        record_fb = getattr(advisor, "record_fallback", None)
        if callable(record_fb):
            record_fb(exc)
        raw_output = assessment.provisional_capability
        advisor_rec = None
        fallback_used = True
        advisor_error_category = categorize_advisor_error(exc)
    except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
        record_fb = getattr(advisor, "record_fallback", None)
        if callable(record_fb):
            record_fb(exc)
        raw_output = assessment.provisional_capability
        advisor_rec = None
        fallback_used = True
        advisor_error_category = "execution_error"
    finally:
        t1 = time.perf_counter_ns()
        advisor_wall_clock_ns = max(0, t1 - t0)
        if not fallback_used and isinstance(advisor_rec, AdvisorRecommendation):
            timing_attr = getattr(advisor, "last_provider_timing", None)
            if isinstance(timing_attr, OllamaProviderTiming):
                provider_timing = timing_attr

    return (
        raw_output,
        advisor_rec,
        fallback_used,
        advisor_wall_clock_ns,
        provider_timing,
        advisor_error_category,
    )


def execute_advisor_consultation_sync(
    advisor: object,
    prompt: str,
    assessment: RoutingAssessment,
) -> tuple[
    AgentCapability,
    AdvisorRecommendation | None,
    bool,
    int,
    OllamaProviderTiming | None,
    AdvisorFailureCategory | None,
]:
    """Execute synchronous advisor consultation for an ambiguous assessment."""
    t0 = time.perf_counter_ns()
    advisor_rec: AdvisorRecommendation | None = None
    fallback_used = False
    advisor_error_category: AdvisorFailureCategory | None = None
    provider_timing: OllamaProviderTiming | None = None

    try:
        advise_fn = getattr(advisor, "advise", None)
        if not callable(advise_fn):
            raise TypeError("Advisor does not have a callable advise method")

        raw_res = advise_fn(prompt, assessment)
        if inspect.iscoroutine(raw_res):
            raw_res.close()
            record_fb = getattr(advisor, "record_fallback", None)
            if callable(record_fb):
                record_fb(
                    AdvisorExecutionError(
                        "Advisor returned unawaited coroutine in synchronous execution"
                    )
                )
            raw_output = assessment.provisional_capability
            advisor_rec = None
            fallback_used = True
            advisor_error_category = "execution_error"
        elif not isinstance(raw_res, AdvisorRecommendation):
            record_fb = getattr(advisor, "record_fallback", None)
            if callable(record_fb):
                record_fb(
                    AdvisorExecutionError("Advisor returned invalid recommendation object")
                )
            raw_output = assessment.provisional_capability
            advisor_rec = None
            fallback_used = True
            advisor_error_category = "execution_error"
        else:
            advisor_rec = raw_res
            raw_output = advisor_rec.recommended_capability
    except (AdvisorError, TimeoutError, OSError) as exc:
        record_fb = getattr(advisor, "record_fallback", None)
        if callable(record_fb):
            record_fb(exc)
        raw_output = assessment.provisional_capability
        advisor_rec = None
        fallback_used = True
        advisor_error_category = categorize_advisor_error(exc)
    except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
        record_fb = getattr(advisor, "record_fallback", None)
        if callable(record_fb):
            record_fb(exc)
        raw_output = assessment.provisional_capability
        advisor_rec = None
        fallback_used = True
        advisor_error_category = "execution_error"
    finally:
        t1 = time.perf_counter_ns()
        advisor_wall_clock_ns = max(0, t1 - t0)
        if not fallback_used and isinstance(advisor_rec, AdvisorRecommendation):
            timing_attr = getattr(advisor, "last_provider_timing", None)
            if isinstance(timing_attr, OllamaProviderTiming):
                provider_timing = timing_attr

    return (
        raw_output,
        advisor_rec,
        fallback_used,
        advisor_wall_clock_ns,
        provider_timing,
        advisor_error_category,
    )


@dataclass(frozen=True)
class AdvisorQualityMetrics:
    """Aggregate quality and disambiguation metrics for advisor consultations."""

    eligible_count: int
    consultation_count: int
    successful_count: int
    agreement_count: int
    override_count: int
    correct_change_count: int
    regression_count: int
    preserved_correct_count: int
    preserved_incorrect_count: int
    fallback_count: int
    unavailable_count: int
    timeout_count: int
    execution_error_count: int
    clear_invocation_count: int


def compute_advisor_quality_metrics(
    case_results: list[CaseResult],
    advisor: Any = None,
) -> AdvisorQualityMetrics:
    """Compute advisor quality metrics, verifying partition and CLEAR invariants."""
    ambiguous_assessed_cases = [r for r in case_results if r.actual_clarity == "ambiguous"]
    consulted_cases = [
        r
        for r in ambiguous_assessed_cases
        if r.advisor_recommendation is not None or r.fallback_used
    ]
    successful_recs = [r for r in consulted_cases if r.advisor_recommendation is not None]
    successful_count = len(successful_recs)

    agreement_count = 0
    override_count = 0
    correct_change_count = 0
    regression_count = 0
    preserved_correct_count = 0
    preserved_incorrect_count = 0

    for r in successful_recs:
        rec_cap_raw = r.advisor_recommendation.recommended_capability  # type: ignore[union-attr]
        rec_cap = rec_cap_raw.value if hasattr(rec_cap_raw, "value") else str(rec_cap_raw).lower()
        prov_cap = r.effective_provisional_capability
        if rec_cap == prov_cap:
            agreement_count += 1
        else:
            override_count += 1

        if not r.is_provisional_correct and r.is_final_correct:
            correct_change_count += 1
        elif r.is_provisional_correct and not r.is_final_correct:
            regression_count += 1
        elif r.is_provisional_correct and r.is_final_correct:
            preserved_correct_count += 1
        elif not r.is_provisional_correct and not r.is_final_correct:
            preserved_incorrect_count += 1

    # Invariant verification: successful recommendations must partition exhaustively and mutually exclusively
    # Explicit runtime check active under optimized Python execution (-O)
    partition_sum = (
        correct_change_count
        + regression_count
        + preserved_correct_count
        + preserved_incorrect_count
    )
    if successful_count != partition_sum:
        raise RuntimeError(
            f"Advisor quality outcome partition invariant violated: successful_count={successful_count} "
            f"!= partition_sum={partition_sum} (correct_change={correct_change_count}, "
            f"regression={regression_count}, preserved_correct={preserved_correct_count}, "
            f"preserved_incorrect={preserved_incorrect_count})"
        )

    fallback_count = sum(1 for r in consulted_cases if r.fallback_used)
    unavailable_count = sum(
        1 for r in consulted_cases if r.advisor_error_category == "unavailable"
    )
    timeout_count = sum(
        1 for r in consulted_cases if r.advisor_error_category == "timeout"
    )
    execution_error_count = sum(
        1 for r in consulted_cases if r.advisor_error_category == "execution_error"
    )
    consultation_count = len(consulted_cases)

    # Invariant verification: CLEAR assessments must never invoke the advisor
    clear_advisor_invocations = sum(
        1
        for r in case_results
        if (
            (r.actual_clarity == "clear" or (r.assessment is not None and r.assessment.clarity == RoutingClarity.CLEAR))
            and (
                r.advisor_recommendation is not None
                or r.fallback_used
                or r.advisor_wall_clock_ns is not None
                or r.is_advisor_clear_invocation
            )
        )
    )
    if clear_advisor_invocations > 0:
        raise RuntimeError(
            f"Routing invariant violation: Advisor was invoked on {clear_advisor_invocations} "
            "CLEAR assessment(s). Only AMBIGUOUS assessments are advisor-eligible."
        )

    return AdvisorQualityMetrics(
        eligible_count=len(ambiguous_assessed_cases),
        consultation_count=consultation_count,
        successful_count=successful_count,
        agreement_count=agreement_count,
        override_count=override_count,
        correct_change_count=correct_change_count,
        regression_count=regression_count,
        preserved_correct_count=preserved_correct_count,
        preserved_incorrect_count=preserved_incorrect_count,
        fallback_count=fallback_count,
        unavailable_count=unavailable_count,
        timeout_count=timeout_count,
        execution_error_count=execution_error_count,
        clear_invocation_count=clear_advisor_invocations,
    )
