import logging
from typing import Any

from agent_contracts import AgentEvent, TaskRequested, TaskResult

from .ports import InferenceProvider, MessageBus, ResultStore

logger = logging.getLogger(__name__)


class TaskExecutionError(Exception):
    """Raised when an expected task failure occurs during execution."""

    def __init__(self, task_id: str, original_error: Exception) -> None:
        self.task_id = task_id
        self.original_error = original_error
        super().__init__(f"Task {task_id} failed: {original_error}")


class TaskProcessor:
    def __init__(
        self,
        agent_id: str,
        bus: MessageBus,
        result_store: ResultStore | None = None,
        inference_provider: InferenceProvider | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.bus = bus
        self.result_store = result_store
        self.inference_provider = inference_provider

    async def process(self, task: TaskRequested) -> Any:
        """Execute the task prompt using the configured inference provider."""
        if self.inference_provider is not None:
            return await self.inference_provider.generate(task.prompt)
        return f"Completed: {task.prompt}"

    async def run_once(self, task: TaskRequested) -> None:
        logger.info("Worker %s processing task %s", self.agent_id, task.task_id)
        await self.bus.publish(
            AgentEvent.create(
                self.agent_id, "working", f"Processing: {task.prompt}", task.task_id
            )
        )
        try:
            res = await self.process(task)
            if isinstance(res, TaskResult):
                result = res
            else:
                result_output = (
                    res
                    if isinstance(res, str) and res.strip()
                    else f"Completed: {task.prompt}"
                )
                result = TaskResult(task_id=task.task_id, output=result_output)

            if self.result_store is not None:
                await self.result_store.save(result)
            elif hasattr(self.bus, "save"):
                await self.bus.save(result)

            await self.bus.publish(
                AgentEvent.create(
                    self.agent_id, "idle", f"Completed: {task.prompt}", task.task_id
                )
            )
            logger.info("Worker %s completed task %s successfully", self.agent_id, task.task_id)
        except Exception as exc:
            logger.exception("Task %s failed during execution", task.task_id)
            # 1. Record failure in event stream and set current state to error
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "error", str(exc), task.task_id)
            )
            # 2. Transition current state back to idle so the worker is available for next task
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "idle", "Waiting for work")
            )
            raise TaskExecutionError(task.task_id, exc) from exc
