from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agent_contracts import (
    AgentAvailability,
    AgentAvailabilityState,
    AgentCandidate,
    AgentEvent,
    AgentExecutionState,
    TaskRequested,
)
from fastapi import HTTPException
from orchestrator.application import (
    FindCandidatesByCapability,
    SubmitTask,
)
from orchestrator.domain import (
    AgentCapability,
    CapabilityUnavailableError,
)
from orchestrator.infrastructure.redis_adapter import RedisGateway
from orchestrator.presentation.api import TaskInput, create_app, create_task
from starlette.requests import Request

from tests.test_application import FakeStateReader, FakeTaskQueue


class FakeCandidateFinder:
    def __init__(
        self, candidates_by_capability: dict[str, list[AgentCandidate]] | None = None
    ) -> None:
        self.candidates_by_capability = candidates_by_capability or {}
        self.calls: list[str] = []

    async def execute(
        self, capability: AgentCapability | str, as_of: Any = None
    ) -> list[AgentCandidate]:
        cap_str = capability.value if isinstance(capability, AgentCapability) else str(capability)
        self.calls.append(cap_str)
        return list(self.candidates_by_capability.get(cap_str, []))


class FakeEventPublisher:
    def __init__(self) -> None:
        self.published: list[AgentEvent] = []

    async def publish(self, event: AgentEvent) -> None:
        self.published.append(event)


def make_candidate(
    agent_id: str,
    capability: str,
    state: AgentAvailabilityState = AgentAvailabilityState.READY,
    execution_state: AgentExecutionState = AgentExecutionState.IDLE,
    dispatch_target: str | None = None,
) -> AgentCandidate:
    now_str = datetime.now(UTC).isoformat()
    return AgentCandidate(
        agent_id=agent_id,
        capabilities=[capability],
        availability=AgentAvailability(
            state=state,
            execution_state=execution_state,
            heartbeat_age_seconds=1.0,
            last_seen_at=now_str,
            as_of=now_str,
            reason="test",
        ),
        agent_type="worker",
        execution_state=execution_state,
        dispatch_target=dispatch_target or f"agent.tasks.{capability}",
    )


@pytest.mark.anyio
async def test_general_submission_succeeds_when_eligible_general_candidate_exists() -> None:
    queue = FakeTaskQueue()
    finder = FakeCandidateFinder(
        {"general": [make_candidate("worker-1", "general")]}
    )
    events = FakeEventPublisher()
    submit_task = SubmitTask(queue, candidate_finder=finder, events=events)

    task = await submit_task.execute("Explain the difference between containers and VMs")

    assert isinstance(task, TaskRequested)
    assert task.prompt == "Explain the difference between containers and VMs"
    assert len(queue.enqueued) == 1
    enqueued_task, capability = queue.enqueued[0]
    assert enqueued_task.task_id == task.task_id
    assert capability == AgentCapability.GENERAL
    assert len(events.published) == 0


@pytest.mark.anyio
async def test_research_submission_succeeds_when_eligible_research_candidate_exists() -> None:
    queue = FakeTaskQueue()
    finder = FakeCandidateFinder(
        {"research": [make_candidate("research-1", "research")]}
    )
    events = FakeEventPublisher()
    submit_task = SubmitTask(queue, candidate_finder=finder, events=events)

    task = await submit_task.execute(
        "Research quantum key distribution breakthroughs",
        capability=AgentCapability.RESEARCH,
    )

    assert isinstance(task, TaskRequested)
    assert len(queue.enqueued) == 1
    enqueued_task, capability = queue.enqueued[0]
    assert enqueued_task.task_id == task.task_id
    assert capability == AgentCapability.RESEARCH
    assert len(events.published) == 0


@pytest.mark.anyio
async def test_automatically_resolved_capability_is_checked_for_availability() -> None:
    queue = FakeTaskQueue()
    # Only GENERAL candidate is online, but prompt requires RESEARCH
    finder = FakeCandidateFinder(
        {"general": [make_candidate("worker-1", "general")]}
    )
    events = FakeEventPublisher()
    submit_task = SubmitTask(queue, candidate_finder=finder, events=events)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        # Phase B requires external retrieval/freshness/source evidence rather than weak verbs alone
        await submit_task.execute("Research the latest quantum key distribution breakthroughs")

    assert exc_info.value.capability == AgentCapability.RESEARCH
    assert "research" in str(exc_info.value.message).lower()
    # Critical: task must NOT be enqueued
    assert len(queue.enqueued) == 0
    # Operational event emitted
    assert len(events.published) == 1
    assert events.published[0].state == "error"
    assert events.published[0].agent_id == "orchestrator"
    assert "research" in events.published[0].detail


@pytest.mark.anyio
async def test_explicit_capability_override_is_checked_for_availability() -> None:
    queue = FakeTaskQueue()
    # Prompt is general knowledge, but explicit override requests RESEARCH which has no candidates
    finder = FakeCandidateFinder(
        {"general": [make_candidate("worker-1", "general")]}
    )
    events = FakeEventPublisher()
    submit_task = SubmitTask(queue, candidate_finder=finder, events=events)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await submit_task.execute(
            "What is a hash table?",
            capability=AgentCapability.RESEARCH,
        )

    assert exc_info.value.capability == AgentCapability.RESEARCH
    assert len(queue.enqueued) == 0
    assert len(events.published) == 1


@pytest.mark.anyio
async def test_missing_capability_candidate_produces_structured_rejection() -> None:
    queue = FakeTaskQueue()
    finder = FakeCandidateFinder({})  # Zero candidates for any capability
    events = FakeEventPublisher()
    submit_task = SubmitTask(queue, candidate_finder=finder, events=events)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await submit_task.execute("Simple prompt")

    err = exc_info.value
    assert err.capability == AgentCapability.GENERAL
    assert err.task_id is not None
    assert len(err.task_id) > 0
    assert "No eligible agent" in err.message
    assert "general" in err.message
    assert len(queue.enqueued) == 0


@pytest.mark.anyio
async def test_only_busy_matching_agents_produces_rejection() -> None:
    now = datetime.now(UTC)
    queue = FakeTaskQueue()
    events = FakeEventPublisher()
    # Matching agent exists but is actively WORKING (availability BUSY, is_eligible == False)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-busy",
                "capabilities": ["general"],
                "state": "working",
                "updated_at": now.isoformat(),
            }
        ]
    )
    finder = FindCandidatesByCapability(reader)
    submit_task = SubmitTask(queue, candidate_finder=finder, events=events)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await submit_task.execute("Simple general task")

    assert exc_info.value.capability == AgentCapability.GENERAL
    assert len(queue.enqueued) == 0
    assert len(events.published) == 1


@pytest.mark.anyio
async def test_only_stale_matching_agents_produces_rejection() -> None:
    now = datetime.now(UTC)
    queue = FakeTaskQueue()
    events = FakeEventPublisher()
    # Matching agent exists but heartbeat is 45s old (exceeds 30s stale threshold)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-stale",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": (now - timedelta(seconds=45)).isoformat(),
            }
        ]
    )
    finder = FindCandidatesByCapability(reader)
    submit_task = SubmitTask(queue, candidate_finder=finder, events=events)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await submit_task.execute("Simple general task")

    assert exc_info.value.capability == AgentCapability.GENERAL
    assert len(queue.enqueued) == 0
    assert len(events.published) == 1


@pytest.mark.anyio
async def test_only_offline_matching_agents_produces_rejection() -> None:
    now = datetime.now(UTC)
    queue = FakeTaskQueue()
    events = FakeEventPublisher()
    # Matching agent exists but heartbeat is 120s old (exceeds 90s offline threshold)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-offline",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": (now - timedelta(seconds=120)).isoformat(),
            }
        ]
    )
    finder = FindCandidatesByCapability(reader)
    submit_task = SubmitTask(queue, candidate_finder=finder, events=events)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await submit_task.execute("Simple general task")

    assert exc_info.value.capability == AgentCapability.GENERAL
    assert len(queue.enqueued) == 0
    assert len(events.published) == 1


@pytest.mark.anyio
async def test_rejected_tasks_are_not_enqueued() -> None:
    queue = FakeTaskQueue()
    finder = FakeCandidateFinder({})
    submit_task = SubmitTask(queue, candidate_finder=finder)

    for cap in (AgentCapability.GENERAL, AgentCapability.RESEARCH):
        with pytest.raises(CapabilityUnavailableError):
            await submit_task.execute("Test task prompt", capability=cap)

    # Completely clean: 0 tasks in queue
    assert queue.enqueued == []


@pytest.mark.anyio
async def test_operational_unavailable_capability_event_is_emitted() -> None:
    queue = FakeTaskQueue()
    events = FakeEventPublisher()
    finder = FakeCandidateFinder({})
    submit_task = SubmitTask(queue, candidate_finder=finder, events=events)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await submit_task.execute(
            "Research test prompt",
            capability=AgentCapability.RESEARCH,
        )

    err = exc_info.value
    assert len(events.published) == 1
    event = events.published[0]
    assert event.agent_id == "orchestrator"
    assert event.state == "error"
    assert event.task_id == err.task_id
    assert "research" in event.detail
    assert "No eligible agent" in event.detail


@pytest.mark.anyio
async def test_successful_submissions_retain_existing_enqueue_behavior() -> None:
    queue = FakeTaskQueue()
    events = FakeEventPublisher()
    finder = FakeCandidateFinder(
        {"general": [make_candidate("worker-1", "general")]}
    )
    submit_task = SubmitTask(queue, candidate_finder=finder, events=events)

    msg = await submit_task.execute("Hello World")

    assert msg.prompt == "Hello World"
    assert len(queue.enqueued) == 1
    enqueued_task, cap = queue.enqueued[0]
    assert enqueued_task.task_id == msg.task_id
    assert enqueued_task.prompt == "Hello World"
    assert cap == AgentCapability.GENERAL
    assert events.published == []


@pytest.mark.anyio
async def test_presentation_layer_returns_http_503_unavailable_response() -> None:
    fake_queue = FakeTaskQueue()
    # Empty state reader -> zero eligible candidates
    app = create_app(
        task_queue=fake_queue,
        state_reader=FakeStateReader(agents_data=[]),
        enforce_availability=True,
    )
    req = Request({"type": "http", "app": app})

    with pytest.raises(HTTPException) as exc_info:
        await create_task(TaskInput(prompt="build agent"), request=req)

    assert exc_info.value.status_code == 503
    assert isinstance(exc_info.value.detail, dict)
    assert exc_info.value.detail["error"] == "capability_unavailable"
    assert exc_info.value.detail["capability"] == "general"
    assert "No eligible agent" in exc_info.value.detail["message"]
    # Ensure task was not enqueued
    assert len(fake_queue.enqueued) == 0


@pytest.mark.anyio
async def test_presentation_layer_successful_submission_returns_202_accepted() -> None:
    fake_queue = FakeTaskQueue()
    now_str = datetime.now(UTC).isoformat()
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-1",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": now_str,
                "dispatch_target": "agent.tasks.general",
            }
        ]
    )
    app = create_app(
        task_queue=fake_queue,
        state_reader=reader,
        enforce_availability=True,
    )
    req = Request({"type": "http", "app": app})

    res = await create_task(TaskInput(prompt="build agent"), request=req)
    assert res["status"] == "accepted"
    assert "task_id" in res
    assert len(fake_queue.enqueued) == 1


@pytest.mark.anyio
async def test_malformed_requests_retain_existing_validation_behavior() -> None:
    from pydantic import ValidationError

    # 1. Pydantic level validation
    with pytest.raises(ValidationError):
        TaskInput(prompt="")

    with pytest.raises(ValidationError):
        TaskInput(prompt="a" * 2001)

    # 2. Domain Task level validation in create_task
    fake_queue = FakeTaskQueue()
    app = create_app(
        task_queue=fake_queue,
        state_reader=FakeStateReader(agents_data=[]),
        enforce_availability=True,
    )
    req = Request({"type": "http", "app": app})

    # Submitting an object that bypasses Pydantic but fails Task validation
    class MalformedTaskInput:
        prompt = "   "
        capability = None

    with pytest.raises(HTTPException) as exc_info:
        await create_task(MalformedTaskInput(), request=req)  # type: ignore[arg-type]

    # Malformed prompt should return 400 Bad Request, NOT 503
    assert exc_info.value.status_code == 400


@pytest.mark.anyio
async def test_redis_gateway_publish_operational_event() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.xadd = AsyncMock()

    event = AgentEvent.create(
        agent_id="orchestrator",
        state="error",
        detail="Capability 'research' unavailable",
        task_id="test-task-1",
    )

    await gateway.publish(event)

    gateway._client.xadd.assert_awaited_once()
    args, kwargs = gateway._client.xadd.call_args
    assert args[0] == "agent.events"
    mapping = args[1]
    assert mapping["agent_id"] == "orchestrator"
    assert mapping["state"] == "error"
    assert mapping["detail"] == "Capability 'research' unavailable"
    assert mapping["task_id"] == "test-task-1"
    assert kwargs.get("maxlen") == 500
