"""Domain and evaluation models for capability routing benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
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
    actual_capability: CapabilityType
    matched_rules: tuple[str, ...] = ()
    assessment: RoutingAssessment | None = None

    @property
    def is_correct(self) -> bool:
        return self.actual_capability == self.case.expected_capability

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
