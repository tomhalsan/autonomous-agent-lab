from uuid import uuid4

from agent_contracts import AgentEvent, TaskRequested

from orchestrator.domain.capability import AgentCapability, CapabilityUnavailableError
from orchestrator.domain.router import DeterministicCapabilityRouter
from orchestrator.domain.selection import (
    CandidateSelectionPolicy,
    DeterministicSelectionPolicy,
)
from orchestrator.domain.task import Task

from .ports import (
    CandidateFinder,
    CapabilityRouter,
    EventPublisher,
    TaskDispatcher,
    TaskQueue,
)


class SubmitTask:
    def __init__(
        self,
        queue: TaskQueue | TaskDispatcher | None = None,
        router: CapabilityRouter | None = None,
        candidate_finder: CandidateFinder | None = None,
        events: EventPublisher | None = None,
        selection_policy: CandidateSelectionPolicy | None = None,
        dispatcher: TaskDispatcher | None = None,
    ) -> None:
        self._dispatcher: TaskDispatcher | None = dispatcher or (
            queue if isinstance(queue, TaskDispatcher) else None
        )
        self._queue: TaskQueue | None = queue if isinstance(queue, TaskQueue) else None
        self._router = router or DeterministicCapabilityRouter()
        self._candidate_finder = candidate_finder
        self._events = events
        self._selection_policy = selection_policy or DeterministicSelectionPolicy()

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

            selected_candidate = self._selection_policy.select(candidates)
            target = selected_candidate.dispatch_target

            if not target or not isinstance(target, str) or not target.strip():
                raise ValueError(
                    f"Selected candidate '{selected_candidate.agent_id}' does not have a valid dispatch target advertised"
                )

            clean_target = target.strip()
            message = TaskRequested.create(task.prompt)

            if self._dispatcher is not None:
                await self._dispatcher.dispatch(message, clean_target)
                return message

            # Preserve backward compatibility only where required for older isolated test doubles
            # that implement TaskQueue but not TaskDispatcher
            if self._queue is not None:
                await self._queue.enqueue(message, resolved_capability, target=clean_target)
                return message

            raise RuntimeError("No task dispatcher or queue configured for dispatch")

        # Non-dynamic fallback path (candidate_finder is None) - for backward compatibility with older isolated tests
        if self._queue is not None:
            message = TaskRequested.create(task.prompt)
            await self._queue.enqueue(message, resolved_capability)
            return message

        raise RuntimeError("No task queue configured for legacy enqueue")
