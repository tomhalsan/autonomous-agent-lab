from agent_contracts import TaskRequested

from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.task import Task

from .ports import TaskQueue


class SubmitTask:
    def __init__(self, queue: TaskQueue) -> None:
        self._queue = queue

    async def execute(
        self,
        prompt: str,
        capability: AgentCapability = AgentCapability.GENERAL,
    ) -> TaskRequested:
        task = Task(prompt=prompt)
        message = TaskRequested.create(task.prompt)
        await self._queue.enqueue(message, capability)
        return message
