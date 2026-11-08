"""Domain and evaluation models for capability routing benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from orchestrator.domain.advisor import AdvisorRecommendation
    from orchestrator.domain.assessment import RoutingAssessment

CapabilityType = Literal["general", "research"]
ClarityType = Literal["clear", "ambiguous"]


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
