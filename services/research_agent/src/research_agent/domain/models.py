from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchResult:
    task_id: str
    output: str
    acknowledged: bool = True

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")


@dataclass(frozen=True)
class SynthesisResult:
    question: str
    content: str
    evidence: str
    provider: str = "stub"
    model: str = "stub"

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")

