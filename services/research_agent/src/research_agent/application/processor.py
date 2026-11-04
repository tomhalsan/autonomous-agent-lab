import logging

from agent_contracts import AgentEvent, TaskRequested

from research_agent.application.ports import (
    InferenceProvider,
    MessageBus,
    ResearchExecutor,
    ResultStore,
)
from research_agent.domain.models import ResearchResult

logger = logging.getLogger(__name__)


class ResearchExecutionError(Exception):
    """Raised when an expected task failure occurs during research execution."""

    def __init__(self, task_id: str, original_error: Exception) -> None:
        self.task_id = task_id
        self.original_error = original_error
        super().__init__(f"Research task {task_id} failed: {original_error}")


class ResearchProcessor:
    def __init__(
        self,
        agent_id: str,
        bus: MessageBus,
        executor: ResearchExecutor,
        result_store: ResultStore | None = None,
        inference_provider: InferenceProvider | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.bus = bus
        self.executor = executor
        self.result_store = result_store
        self.inference_provider = inference_provider

    async def run_once(self, task: TaskRequested) -> None:
        logger.info("Research agent %s processing task %s", self.agent_id, task.task_id)
        await self.bus.publish(
            AgentEvent.create(
                self.agent_id, "working", f"Processing: {task.prompt}", task.task_id
            )
        )
        try:
            result = await self.executor.execute(task)

            if self.inference_provider is not None:
                synthesis = await self.inference_provider.synthesize(
                    question=task.prompt, evidence=result.output
                )
                result = ResearchResult(
                    task_id=task.task_id,
                    output=synthesis.content,
                    acknowledged=result.acknowledged,
                )

            if self.result_store is not None:
                await self.result_store.save(result)
            elif hasattr(self.bus, "save"):
                await self.bus.save(result)

            await self.bus.publish(
                AgentEvent.create(
                    self.agent_id, "idle", f"Completed: {result.output}", task.task_id
                )
            )
            logger.info(
                "Research agent %s completed task %s successfully",
                self.agent_id,
                task.task_id,
            )
        except Exception as exc:
            logger.exception("Research task %s failed during execution", task.task_id)
            # Record failure in event stream and set current state to error.
            # Do not immediately overwrite error with idle.
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "error", str(exc), task.task_id)
            )
            raise ResearchExecutionError(task.task_id, exc) from exc
