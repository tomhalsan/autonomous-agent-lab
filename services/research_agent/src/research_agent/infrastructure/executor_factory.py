import os

from research_agent.application.ports import ResearchExecutor
from research_agent.infrastructure.stub_executor import StubResearchExecutor


def create_research_executor(name: str | None = None) -> ResearchExecutor:
    """Instantiate a ResearchExecutor based on the provided name or RESEARCH_EXECUTOR env var.

    Defaults to 'stub' if no name is provided and RESEARCH_EXECUTOR is unset.
    Raises ValueError on unsupported executor names.
    """
    raw_name = name if name is not None else os.getenv("RESEARCH_EXECUTOR", "stub")
    normalized = raw_name.strip().lower()

    if normalized == "stub":
        return StubResearchExecutor()

    raise ValueError(
        f"Unsupported research executor: {raw_name!r}. Supported executors: ['stub']"
    )

