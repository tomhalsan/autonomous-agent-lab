from agent_contracts import TaskRequested

from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.router import DeterministicCapabilityRouter
from orchestrator.domain.task import Task

from .ports import CapabilityRouter, TaskQueue


class SubmitTask:
    def __init__(
        self,
        queue: TaskQueue,
        router: CapabilityRouter | None = None,
    ) -> None:
        self._queue = queue
        self._router = router or DeterministicCapabilityRouter()

    async def execute(
        self,
        prompt: str,
        capability: AgentCapability | None = None,
    ) -> TaskRequested:
        task = Task(prompt=prompt)
        resolved_capability = (
            capability if capability is not None else self._router.route(task.prompt)
        )
        message = TaskRequested.create(task.prompt)
        await self._queue.enqueue(message, resolved_capability)
        return message
