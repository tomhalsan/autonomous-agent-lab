"""Domain and evaluation models for capability routing benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from orchestrator.domain.advisor import AdvisorRecommendation
    from orchestrator.domain.assessment import RoutingAssessment

CapabilityType = Literal["general", "research"]
ClarityType = Literal["clear", "ambiguous"]
AdvisorFailureCategory = Literal["unavailable", "timeout", "execution_error"]
ColdProbeStatus = Literal["not_requested", "succeeded", "failed"]
UnloadStatus = Literal["not_requested", "acknowledged", "not_acknowledged"]


def parse_provider_duration_ns(val: object) -> int | None:
    """Validate and extract a non-negative integer duration in nanoseconds."""
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        return None
    return val


@dataclass(frozen=True)
class ColdProbeOutcome:
    """Explicit evaluation-side cold-probe telemetry and outcome."""

    status: ColdProbeStatus
    latency_ns: int | None = None
    failure_category: AdvisorFailureCategory | None = None
    unload_acknowledged: bool = False
    label: str | None = None
    provider_total_duration_ns: int | None = None
    provider_load_duration_ns: int | None = None

    def __post_init__(self) -> None:
        def _check_nonnegative_int(val: object, name: str) -> None:
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {val!r}")

        if self.status == "not_requested":
            if self.latency_ns is not None:
                raise ValueError("latency_ns must be None when status is 'not_requested'")
            if self.failure_category is not None:
                raise ValueError("failure_category must be None when status is 'not_requested'")
            if (
                self.provider_total_duration_ns is not None
                or self.provider_load_duration_ns is not None
            ):
                raise ValueError(
                    "provider timing fields must be None when status is 'not_requested'"
                )

        elif self.status == "succeeded":
            if self.latency_ns is None:
                raise ValueError("latency_ns is required when status is 'succeeded'")
            _check_nonnegative_int(self.latency_ns, "latency_ns")
            if self.failure_category is not None:
                raise ValueError("failure_category must be None when status is 'succeeded'")
            if self.provider_total_duration_ns is not None:
                _check_nonnegative_int(
                    self.provider_total_duration_ns, "provider_total_duration_ns"
                )
            if self.provider_load_duration_ns is not None:
                _check_nonnegative_int(
                    self.provider_load_duration_ns, "provider_load_duration_ns"
                )

        elif self.status == "failed":
            if self.latency_ns is not None:
                raise ValueError("latency_ns must be None when status is 'failed'")
            if self.failure_category is None:
                raise ValueError("failure_category is required when status is 'failed'")
            if (
                self.provider_total_duration_ns is not None
                or self.provider_load_duration_ns is not None
            ):
                raise ValueError("provider timing fields must be None when status is 'failed'")

        else:
            raise ValueError(f"Invalid ColdProbeStatus: {self.status!r}")


@dataclass(frozen=True)
class OllamaProviderTiming:
    """Provider-reported timing telemetry from Ollama /api/generate in nanoseconds."""

    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_duration_ns: int | None = None


@dataclass(frozen=True)
class EvaluationCase:
    """A single labeled routing evaluation test case."""

    id: str
    prompt: str
    expected_capability: CapabilityType
    clarity: ClarityType
    rationale: str
    category: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CaseResult:
    """Result of evaluating a single test case for correctness and clarity."""

    case: EvaluationCase
    actual_capability: CapabilityType  # Final capability after routing/advising (retained for backward compat)
    matched_rules: tuple[str, ...] = ()
    assessment: RoutingAssessment | None = None
    advisor_recommendation: AdvisorRecommendation | None = None
    fallback_used: bool = False
    provisional_capability: CapabilityType | None = None
    advisor_wall_clock_ns: int | None = None
    provider_timing: OllamaProviderTiming | None = None
    advisor_error_category: AdvisorFailureCategory | None = None
    is_advisor_clear_invocation: bool = False

    @property
    def final_capability(self) -> CapabilityType:
        """Final post-advisor capability (or provisional if fallback/no advisor)."""
        return self.actual_capability

    @property
    def effective_provisional_capability(self) -> CapabilityType:
        """Deterministic provisional capability before advisor consultation."""
        if self.provisional_capability is not None:
            return self.provisional_capability
        if self.assessment is not None:
            cap = self.assessment.provisional_capability
            raw_val = cap.value if hasattr(cap, "value") else cap
            if isinstance(raw_val, str):
                normalized = raw_val.strip().lower()
                if normalized == "general":
                    return "general"
                if normalized == "research":
                    return "research"
            raise ValueError(
                f"Unsupported assessment provisional capability: {cap!r}. "
                "Expected AgentCapability.GENERAL or AgentCapability.RESEARCH."
            )
        return self.actual_capability

    @property
    def is_correct(self) -> bool:
        """Final capability matches expected ground truth."""
        return self.actual_capability == self.case.expected_capability

    @property
    def is_final_correct(self) -> bool:
        """Final capability matches expected ground truth."""
        return self.is_correct

    @property
    def is_provisional_correct(self) -> bool:
        """Deterministic provisional capability matches expected ground truth."""
        return self.effective_provisional_capability == self.case.expected_capability

    @property
    def actual_clarity(self) -> ClarityType | None:
        if self.assessment is not None:
            return "clear" if str(self.assessment.clarity).lower() == "clear" else "ambiguous"
        return None

    @property
    def is_clarity_correct(self) -> bool:
        if self.actual_clarity is not None:
            return self.actual_clarity == self.case.clarity
        return False


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary confusion matrix for capability determination (Research as Positive class)."""

    true_research: int
    false_research: int  # False positive: predicted research, expected general
    true_general: int
    false_general: int  # False negative: predicted general, expected research

    @property
    def total(self) -> int:
        return (
            self.true_research
            + self.false_research
            + self.true_general
            + self.false_general
        )


@dataclass(frozen=True)
class ClarityConfusionMatrix:
    """Confusion matrix for clarity determination (Ambiguous as Positive class)."""

    true_ambiguous: int
    false_ambiguous: int  # Assessed ambiguous, expected clear
    true_clear: int
    false_clear: int  # Assessed clear, expected ambiguous

    @property
    def total(self) -> int:
        return (
            self.true_ambiguous
            + self.false_ambiguous
            + self.true_clear
            + self.false_clear
        )


@dataclass(frozen=True)
class LatencyStats:
    """Statistical summary of routing execution latency in microseconds."""

    warmup_iterations: int
    measured_iterations: int
    sample_count: int
    mean_us: float
    p50_us: float
    p90_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float
    total_ns: int


@dataclass(frozen=True)
class LatencyDistribution:
    """Statistical distribution of latency measurements in nanoseconds."""

    sample_count: int
    mean_ns: float
    p50_ns: float
    p90_ns: float
    p95_ns: float
    p99_ns: float | None
    min_ns: int
    max_ns: int


@dataclass(frozen=True)
class AdvisorLatencyStats:
    """Statistical summary of advisor latency populations in nanoseconds."""

    # Warm successful consultations wall-clock distribution
    warm_successful: LatencyDistribution | None = None

    # Failed / fallback attempts wall-clock distribution (if any exist)
    failed_attempts: LatencyDistribution | None = None

    # Provider duration averages across successful consultations
    provider_total_duration_mean_ns: float | None = None
    provider_load_duration_mean_ns: float | None = None
    provider_prompt_eval_duration_mean_ns: float | None = None
    provider_eval_duration_mean_ns: float | None = None

    # Explicit cold probe outcome
    cold_probe: ColdProbeOutcome | None = None

    @property
    def sample_count(self) -> int:
        return self.warm_successful.sample_count if self.warm_successful else 0

    @property
    def client_mean_ns(self) -> float:
        return self.warm_successful.mean_ns if self.warm_successful else 0.0

    @property
    def client_p50_ns(self) -> float:
        return self.warm_successful.p50_ns if self.warm_successful else 0.0

    @property
    def client_p90_ns(self) -> float:
        return self.warm_successful.p90_ns if self.warm_successful else 0.0

    @property
    def client_p95_ns(self) -> float:
        return self.warm_successful.p95_ns if self.warm_successful else 0.0

    @property
    def client_p99_ns(self) -> float | None:
        return self.warm_successful.p99_ns if self.warm_successful else None

    @property
    def client_min_ns(self) -> int:
        return self.warm_successful.min_ns if self.warm_successful else 0

    @property
    def client_max_ns(self) -> int:
        return self.warm_successful.max_ns if self.warm_successful else 0

    @property
    def cold_probe_client_ns(self) -> int | None:
        return (
            self.cold_probe.latency_ns
            if self.cold_probe and self.cold_probe.status == "succeeded"
            else None
        )

    @property
    def cold_probe_label(self) -> str | None:
        return self.cold_probe.label if self.cold_probe else None


@dataclass(frozen=True)
class AdvisorRunMetadata:
    """Experimental metadata documenting advisor configuration and methodology."""

    provider: str
    model: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = None
    keep_alive: int | str | None = None
    prompt_policy: str = "prompt-only C2"
    schema: str = '{"capability": "general"|"research", "reason": str}'
    cold_probe_outcome: ColdProbeOutcome | None = None
    unload_after_run: bool = False
    unload_status: UnloadStatus = "not_requested"

    @property
    def cold_probe_performed(self) -> bool:
        return (
            self.cold_probe_outcome is not None
            and self.cold_probe_outcome.status != "not_requested"
        )

    @property
    def cold_probe_label(self) -> str | None:
        return self.cold_probe_outcome.label if self.cold_probe_outcome else None


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate result of running a capability router across an evaluation dataset."""

    router_name: str
    total_cases: int
    clear_cases: int
    ambiguous_cases: int
    overall_correct: int
    clear_correct: int
    ambiguous_correct: int
    overall_accuracy: float
    clear_accuracy: float
    ambiguous_accuracy: float
    research_precision: float
    research_recall: float
    research_f1: float
    general_precision: float
    general_recall: float
    general_f1: float
    confusion_matrix: ConfusionMatrix
    latency: LatencyStats
    category_accuracies: dict[str, float]
    dataset_name: str = "development"
    case_results: list[CaseResult] = field(default_factory=list)
    # Phase B assessment and ambiguity metrics
    clarity_correct: int = 0
    clarity_accuracy: float = 0.0
    advisor_delegation_count: int = 0
    advisor_delegation_rate: float = 0.0
    clear_assessment_accuracy: float = 0.0
    ambiguous_assessment_accuracy: float = 0.0
    clarity_confusion_matrix: ClarityConfusionMatrix | None = None
    # Phase C1 simulated advisor metrics
    simulated_advisor_enabled: bool = False
    advisor_eligible_count: int = 0
    advisor_consultation_count: int = 0
    advisor_agreement_count: int = 0
    advisor_override_count: int = 0
    advisor_fallback_count: int = 0
    advisor_unavailable_count: int = 0
    advisor_timeout_count: int = 0
    advisor_execution_error_count: int = 0
    # Phase C1 / C2 explicit provisional vs final metrics
    provisional_overall_correct: int = 0
    provisional_overall_accuracy: float = 0.0
    provisional_clear_correct: int = 0
    provisional_clear_accuracy: float = 0.0
    provisional_ambiguous_correct: int = 0
    provisional_ambiguous_accuracy: float = 0.0
    provisional_confusion_matrix: ConfusionMatrix | None = None
    provisional_category_accuracies: dict[str, float] = field(default_factory=dict)
    # Phase C2 model advisor evaluation metrics
    advisor_provider: str | None = None
    advisor_metadata: AdvisorRunMetadata | None = None
    advisor_latency: AdvisorLatencyStats | None = None
    advisor_successful_count: int = 0
    advisor_correct_change_count: int = 0
    advisor_regression_count: int = 0
    advisor_preserved_correct_count: int = 0
    advisor_preserved_incorrect_count: int = 0
    advisor_clear_invocation_count: int = 0

    @property
    def final_overall_correct(self) -> int:
        return self.overall_correct

    @property
    def final_overall_accuracy(self) -> float:
        return self.overall_accuracy

    @property
    def final_clear_correct(self) -> int:
        return self.clear_correct

    @property
    def final_clear_accuracy(self) -> float:
        return self.clear_accuracy

    @property
    def final_ambiguous_correct(self) -> int:
        return self.ambiguous_correct

    @property
    def final_ambiguous_accuracy(self) -> float:
        return self.ambiguous_accuracy

    @property
    def final_confusion_matrix(self) -> ConfusionMatrix:
        return self.confusion_matrix
