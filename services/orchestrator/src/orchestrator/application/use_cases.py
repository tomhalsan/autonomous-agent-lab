from uuid import uuid4

from agent_contracts import AgentEvent, TaskRequested

from orchestrator.domain.capability import AgentCapability, CapabilityUnavailableError
from orchestrator.domain.router import DeterministicCapabilityRouter
from orchestrator.domain.task import Task

from .ports import CandidateFinder, CapabilityRouter, EventPublisher, TaskQueue


class SubmitTask:
    def __init__(
        self,
        queue: TaskQueue,
        router: CapabilityRouter | None = None,
        candidate_finder: CandidateFinder | None = None,
        events: EventPublisher | None = None,
    ) -> None:
        self._queue = queue
        self._router = router or DeterministicCapabilityRouter()
        self._candidate_finder = candidate_finder
        self._events = events

    async def execute(
        self,
        prompt: str,
        capability: AgentCapability | None = None,
    ) -> TaskRequested:
        task = Task(prompt=prompt)
        resolved_capability = (
            capability if capability is not None else self._router.route(task.prompt)
        )

        if self._candidate_finder is not None:
            candidates = await self._candidate_finder.execute(resolved_capability)
            if not candidates:
                rejection_task_id = str(uuid4())
                rejection_message = (
                    f"No eligible agent is currently available for capability '{resolved_capability}'"
                )

                if self._events is not None:
                    event = AgentEvent.create(
                        agent_id="orchestrator",
                        state="error",
                        detail=rejection_message,
                        task_id=rejection_task_id,
                    )
                    await self._events.publish(event)

                raise CapabilityUnavailableError(
                    capability=resolved_capability,
                    message=rejection_message,
                    task_id=rejection_task_id,
                )

        message = TaskRequested.create(task.prompt)
        await self._queue.enqueue(message, resolved_capability)
        return message
