import asyncio
import logging

from agent_contracts import AgentEvent, TaskRequested

from .ports import MessageBus

logger = logging.getLogger(__name__)


class TaskExecutionError(Exception):
    """Raised when an expected task failure occurs during execution."""

    def __init__(self, task_id: str, original_error: Exception) -> None:
        self.task_id = task_id
        self.original_error = original_error
        super().__init__(f"Task {task_id} failed: {original_error}")


class TaskProcessor:
    def __init__(self, agent_id: str, bus: MessageBus) -> None:
        self.agent_id = agent_id
        self.bus = bus

    async def process(self, task: TaskRequested) -> None:
        """Execute the task payload. Subclasses/extensions can override."""
        await asyncio.sleep(1)

    async def run_once(self, task: TaskRequested) -> None:
        logger.info("Worker %s processing task %s", self.agent_id, task.task_id)
        await self.bus.publish(
            AgentEvent.create(
                self.agent_id, "working", f"Processing: {task.prompt}", task.task_id
            )
        )
        try:
            await self.process(task)
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
