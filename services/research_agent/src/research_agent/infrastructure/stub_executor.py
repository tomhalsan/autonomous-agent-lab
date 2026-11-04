from agent_contracts import TaskRequested

from research_agent.domain.models import ResearchResult


class StubResearchExecutor:
    """Deterministic stub executor for the initial research agent increment.

    Does not introduce an LLM, web search, MCP, vector database, external provider,
    or fake research findings.
    """

    def __init__(self, failure_trigger: str | None = None) -> None:
        self.failure_trigger = failure_trigger

    async def execute(self, task: TaskRequested) -> ResearchResult:
        if self.failure_trigger and self.failure_trigger in task.prompt:
            raise RuntimeError(f"Simulated execution failure for prompt: {task.prompt}")
        return ResearchResult(
            task_id=task.task_id,
            output=f"Deterministic execution completed for task: {task.task_id}",
        )

