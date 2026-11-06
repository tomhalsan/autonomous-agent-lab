from dataclasses import dataclass

from research_agent.domain.evidence import EvaluatedEvidence


@dataclass(frozen=True)
class ResearchResult:
    task_id: str
    output: str
    acknowledged: bool = True
    truncated: bool = False
    sources: tuple[EvaluatedEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if isinstance(self.sources, (list, tuple)):
            object.__setattr__(self, "sources", tuple(self.sources))

    @property
    def retained_sources(self) -> tuple[EvaluatedEvidence, ...]:
        return tuple(s for s in self.sources if s.retained)


@dataclass(frozen=True)
class SynthesisResult:
    question: str
    content: str
    evidence: str
    provider: str = "stub"
    model: str = "stub"
    truncated: bool = False
    finish_reason: str | None = None
    usage: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
