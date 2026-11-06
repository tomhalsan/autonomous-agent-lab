from uuid import uuid4

from agent_contracts import AgentEvent, RoutingDecision, TaskRequested

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
    RoutingRecorder,
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
        routing_recorder: RoutingRecorder | None = None,
    ) -> None:
        self._dispatcher: TaskDispatcher | None = dispatcher or (
            queue if isinstance(queue, TaskDispatcher) else None
        )
        self._queue: TaskQueue | None = queue if isinstance(queue, TaskQueue) else None
        self._router = router or DeterministicCapabilityRouter()
        self._candidate_finder = candidate_finder
        self._events = events
        self._selection_policy = selection_policy or DeterministicSelectionPolicy()
        self._routing_recorder: RoutingRecorder | None = routing_recorder

    async def execute(
        self,
        prompt: str,
        capability: AgentCapability | None = None,
    ) -> TaskRequested:
        task = Task(prompt=prompt)
        resolved_capability = (
            capability if capability is not None else self._router.route(task.prompt)
        )

        prompt_classification = (
            resolved_capability.value
            if not hasattr(self._router, "detect_intent")
            else (
                intent.value
                if (intent := self._router.detect_intent(task.prompt)) is not None
                else resolved_capability.value
            )
        )

        if self._candidate_finder is not None:
            candidates = await self._candidate_finder.execute(resolved_capability)
            if not candidates:
                rejection_task_id = str(uuid4())
                rejection_message = (
                    f"No eligible agent is currently available for capability '{resolved_capability}'"
                )

                decision = RoutingDecision(
                    task_id=rejection_task_id,
                    prompt_classification=prompt_classification,
                    required_capabilities=[resolved_capability.value],
                    candidates_evaluated=[],
                    selected_agent_id=None,
                    dispatch_target=None,
                    rationale=rejection_message,
                )

                if self._routing_recorder is not None:
                    await self._routing_recorder.record_routing_decision(decision)

                if self._events is not None:
                    event = AgentEvent.create(
                        agent_id="orchestrator",
                        state="error",
                        detail=rejection_message,
                        task_id=rejection_task_id,
                        routing_decision=decision.to_dict(),
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

            evaluated_candidates = [
                {
                    "agent_id": c.agent_id,
                    "agent_type": c.agent_type,
                    "capabilities": list(c.capabilities),
                    "execution_state": (
                        c.execution_state.value
                        if hasattr(c.execution_state, "value")
                        else str(c.execution_state)
                    ),
                    "availability_state": (
                        c.availability.state.value
                        if hasattr(c.availability.state, "value")
                        else str(c.availability.state)
                    ),
                    "dispatch_target": c.dispatch_target,
                    "is_eligible": c.availability.is_eligible,
                }
                for c in candidates
            ]

            policy_name = getattr(
                self._selection_policy, "__class__", type(self._selection_policy)
            ).__name__
            rationale = (
                f"Selected agent '{selected_candidate.agent_id}' targeting '{clean_target}' "
                f"from {len(candidates)} candidate(s) using {policy_name} for capability '{resolved_capability.value}'"
            )

            decision = RoutingDecision(
                task_id=message.task_id,
                prompt_classification=prompt_classification,
                required_capabilities=[resolved_capability.value],
                candidates_evaluated=evaluated_candidates,
                selected_agent_id=selected_candidate.agent_id,
                dispatch_target=clean_target,
                rationale=rationale,
            )

            if self._routing_recorder is not None:
                await self._routing_recorder.record_routing_decision(decision)

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
            rationale = (
                f"Static configuration routed to capability '{resolved_capability.value}'"
            )
            decision = RoutingDecision(
                task_id=message.task_id,
                prompt_classification=prompt_classification,
                required_capabilities=[resolved_capability.value],
                candidates_evaluated=[],
                selected_agent_id=None,
                dispatch_target=None,
                rationale=rationale,
            )

            if self._routing_recorder is not None:
                await self._routing_recorder.record_routing_decision(decision)

            await self._queue.enqueue(message, resolved_capability)
            return message

        raise RuntimeError("No task queue configured for legacy enqueue")
