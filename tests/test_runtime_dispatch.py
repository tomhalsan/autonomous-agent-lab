import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from agent_contracts import (
    AgentAvailability,
    AgentAvailabilityState,
    AgentCandidate,
    AgentCapabilityMetadata,
    AgentEvent,
    AgentExecutionState,
    TaskRequested,
)
from orchestrator.application import (
    FindCandidatesByCapability,
    SubmitTask,
    TaskDispatcher,
)
from orchestrator.domain import (
    AgentCapability,
    CandidateSelectionPolicy,
    CapabilityUnavailableError,
    DeterministicSelectionPolicy,
)
from orchestrator.infrastructure.redis_adapter import (
    CAPABILITY_QUEUE_MAP,
    RedisGateway,
)
from orchestrator.presentation.api import TaskInput, create_app, create_task
from research_agent.infrastructure.redis_bus import RedisBus as ResearchRedisBus
from starlette.requests import Request
from worker.infrastructure.redis_bus import RedisBus as WorkerRedisBus

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
    capability: str = "general",
    dispatch_target: str | None = None,
    state: AgentAvailabilityState = AgentAvailabilityState.READY,
    execution_state: AgentExecutionState = AgentExecutionState.IDLE,
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
            reason="healthy",
        ),
        agent_type="worker",
        execution_state=execution_state,
        dispatch_target=dispatch_target,
    )


# ==============================================================================
# 1. Metadata Contracts & Validation Tests
# ==============================================================================


def test_agent_capability_metadata_dispatch_target_defaults_to_none() -> None:
    meta = AgentCapabilityMetadata(
        agent_id="agent-1",
        capabilities=["general"],
    )
    assert meta.dispatch_target is None
    data = meta.to_dict()
    assert "dispatch_target" not in data or data["dispatch_target"] is None


def test_agent_capability_metadata_stores_and_serializes_dispatch_target() -> None:
    meta = AgentCapabilityMetadata(
        agent_id="worker-1",
        capabilities=["general"],
        dispatch_target="agent.tasks.general",
    )
    assert meta.dispatch_target == "agent.tasks.general"

    # Serialization round-trip
    data = meta.to_dict()
    assert data["dispatch_target"] == "agent.tasks.general"

    restored = AgentCapabilityMetadata.from_dict(data)
    assert restored.dispatch_target == "agent.tasks.general"
    assert restored == meta

    json_str = meta.to_json()
    restored_json = AgentCapabilityMetadata.from_json(json_str)
    assert restored_json.dispatch_target == "agent.tasks.general"
    assert restored_json == meta


def test_agent_capability_metadata_rejects_empty_dispatch_target() -> None:
    with pytest.raises(ValueError, match="dispatch_target must be a non-empty string"):
        AgentCapabilityMetadata(
            agent_id="worker-1",
            capabilities=["general"],
            dispatch_target="   ",
        )


def test_agent_candidate_stores_and_serializes_dispatch_target() -> None:
    candidate = make_candidate("worker-1", dispatch_target="agent.tasks.worker-1")
    assert candidate.dispatch_target == "agent.tasks.worker-1"

    data = candidate.to_dict()
    assert data["dispatch_target"] == "agent.tasks.worker-1"

    restored = AgentCandidate.from_dict(data)
    assert restored.dispatch_target == "agent.tasks.worker-1"
    assert restored.agent_id == "worker-1"


def test_agent_candidate_rejects_empty_dispatch_target() -> None:
    with pytest.raises(ValueError, match="dispatch_target must be a non-empty string"):
        make_candidate("worker-1", dispatch_target=" \t ")


# ==============================================================================
# 2. Agent Registration & Advertisement Tests
# ==============================================================================


@pytest.mark.anyio
async def test_worker_registers_configured_dispatch_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_ID", "worker-prod-1")
    monkeypatch.setenv("TASK_QUEUE", "agent.tasks.custom-worker-queue")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_bus.register.assert_awaited_once()
        metadata: AgentCapabilityMetadata = mock_bus.register.call_args[0][0]
        assert metadata.agent_id == "worker-prod-1"
        assert metadata.dispatch_target == "agent.tasks.custom-worker-queue"
        assert "general" in metadata.capabilities


@pytest.mark.anyio
async def test_worker_default_dispatch_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ID", "worker-default")
    monkeypatch.delenv("TASK_QUEUE", raising=False)

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_bus.register.assert_awaited_once()
        metadata: AgentCapabilityMetadata = mock_bus.register.call_args[0][0]
        assert metadata.dispatch_target == "agent.tasks.general"


@pytest.mark.anyio
async def test_research_agent_registers_configured_dispatch_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_ID", "research-prod-1")
    monkeypatch.setenv("TASK_QUEUE", "agent.tasks.custom-research-queue")

    with patch("research_agent.main.RedisBus") as mock_bus_cls, patch(
        "research_agent.main.ResearchProcessor"
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_bus.register.assert_awaited_once()
        metadata: AgentCapabilityMetadata = mock_bus.register.call_args[0][0]
        assert metadata.agent_id == "research-prod-1"
        assert metadata.dispatch_target == "agent.tasks.custom-research-queue"
        assert "research" in metadata.capabilities


@pytest.mark.anyio
async def test_research_agent_default_dispatch_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ID", "research-default")
    monkeypatch.delenv("TASK_QUEUE", raising=False)

    with patch("research_agent.main.RedisBus") as mock_bus_cls, patch(
        "research_agent.main.ResearchProcessor"
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_bus.register.assert_awaited_once()
        metadata: AgentCapabilityMetadata = mock_bus.register.call_args[0][0]
        assert metadata.dispatch_target == "agent.tasks.research"


@pytest.mark.anyio
async def test_worker_redis_bus_persists_dispatch_target_in_hash() -> None:
    bus = WorkerRedisBus("redis://localhost:6379/0", queue_name="agent.tasks.worker-pool-1")
    bus.client.sadd = AsyncMock()  # type: ignore[method-assign]
    bus.client.hset = AsyncMock()  # type: ignore[method-assign]

    meta = AgentCapabilityMetadata(
        agent_id="worker-pool-1",
        capabilities=["general"],
        agent_type="worker",
        dispatch_target="agent.tasks.worker-pool-1",
    )
    await bus.register(meta)

    bus.client.hset.assert_awaited_once()
    call_kwargs = bus.client.hset.call_args[1]
    mapping = call_kwargs["mapping"]
    assert mapping["dispatch_target"] == "agent.tasks.worker-pool-1"


@pytest.mark.anyio
async def test_research_redis_bus_persists_dispatch_target_in_hash() -> None:
    bus = ResearchRedisBus("redis://localhost:6379/0", queue_name="agent.tasks.research-alpha")
    bus.client.sadd = AsyncMock()  # type: ignore[method-assign]
    bus.client.hset = AsyncMock()  # type: ignore[method-assign]

    meta = AgentCapabilityMetadata(
        agent_id="research-alpha",
        capabilities=["research"],
        agent_type="research-agent",
        dispatch_target="agent.tasks.research-alpha",
    )
    await bus.register(meta)

    bus.client.hset.assert_awaited_once()
    call_kwargs = bus.client.hset.call_args[1]
    mapping = call_kwargs["mapping"]
    assert mapping["dispatch_target"] == "agent.tasks.research-alpha"


# ==============================================================================
# 3. Candidate Discovery & Metadata Propagation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_find_candidates_propagates_advertised_dispatch_target() -> None:
    now = datetime.now(UTC)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-42",
                "agent_type": "worker",
                "capabilities": ["general"],
                "dispatch_target": "agent.tasks.worker-42-custom",
                "state": "idle",
                "updated_at": now.isoformat(),
            }
        ]
    )
    query = FindCandidatesByCapability(reader)
    candidates = await query.execute(AgentCapability.GENERAL, as_of=now + timedelta(seconds=1))

    assert len(candidates) == 1
    assert candidates[0].agent_id == "worker-42"
    assert candidates[0].dispatch_target == "agent.tasks.worker-42-custom"


@pytest.mark.anyio
async def test_find_candidates_handles_missing_dispatch_target_gracefully() -> None:
    now = datetime.now(UTC)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "legacy-worker",
                "agent_type": "worker",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": now.isoformat(),
            }
        ]
    )
    query = FindCandidatesByCapability(reader)
    candidates = await query.execute(AgentCapability.GENERAL, as_of=now + timedelta(seconds=1))

    assert len(candidates) == 1
    assert candidates[0].agent_id == "legacy-worker"
    assert candidates[0].dispatch_target is None


# ==============================================================================
# 4. Selection Policy Tests
# ==============================================================================


def test_deterministic_selection_policy_empty_candidates_raises() -> None:
    policy = DeterministicSelectionPolicy()
    with pytest.raises(ValueError, match="Cannot select candidate from an empty candidate list"):
        policy.select([])


def test_deterministic_selection_policy_single_candidate() -> None:
    policy = DeterministicSelectionPolicy()
    c = make_candidate("worker-1")
    assert policy.select([c]) == c


def test_deterministic_selection_policy_multiple_candidates_is_order_independent() -> None:
    policy = DeterministicSelectionPolicy()
    c1 = make_candidate("agent-bravo", dispatch_target="queue-bravo")
    c2 = make_candidate("agent-alpha", dispatch_target="queue-alpha")
    c3 = make_candidate("agent-charlie", dispatch_target="queue-charlie")

    # Regardless of input permutation, "agent-alpha" is lexicographically first
    assert policy.select([c1, c2, c3]).agent_id == "agent-alpha"
    assert policy.select([c3, c1, c2]).agent_id == "agent-alpha"
    assert policy.select([c2, c3, c1]).agent_id == "agent-alpha"


def test_deterministic_selection_policy_arbitrary_ids() -> None:
    policy = DeterministicSelectionPolicy()
    c_uuid1 = make_candidate("00000000-0000-0000-0000-000000000002")
    c_uuid2 = make_candidate("00000000-0000-0000-0000-000000000001")

    selected = policy.select([c_uuid1, c_uuid2])
    assert selected.agent_id == "00000000-0000-0000-0000-000000000001"


def test_custom_selection_policy_extension() -> None:
    class ReverseAlphabeticalPolicy:
        def select(self, candidates: list[AgentCandidate]) -> AgentCandidate:
            return max(candidates, key=lambda c: c.agent_id)

    policy: CandidateSelectionPolicy = ReverseAlphabeticalPolicy()
    c1 = make_candidate("agent-a")
    c2 = make_candidate("agent-z")
    assert policy.select([c1, c2]).agent_id == "agent-z"


# ==============================================================================
# 5. Orchestrator Dynamic Dispatch Tests
# ==============================================================================


@pytest.mark.anyio
async def test_orchestrator_dispatches_to_selected_candidate_advertised_target() -> None:
    queue = FakeTaskQueue()
    target_queue = "agent.tasks.custom-worker-cluster"
    candidate = make_candidate("worker-1", capability="general", dispatch_target=target_queue)
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(queue, candidate_finder=finder)
    task = await submit_task.execute("Process background task")

    # Verify task was dispatched to the advertised target queue
    assert len(queue.dispatches) == 1
    dispatched_task, destination = queue.dispatches[0]
    assert dispatched_task.task_id == task.task_id
    assert destination == target_queue


@pytest.mark.anyio
async def test_dynamic_queue_change_requires_no_orchestrator_topology_changes() -> None:
    queue = FakeTaskQueue()
    # A completely unexpected, dynamically registered queue name
    dynamic_queue = "agent.tasks.ephemeral-worker-node-883"
    candidate = make_candidate(
        "worker-ephemeral-883", capability="general", dispatch_target=dynamic_queue
    )
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(queue, candidate_finder=finder)
    task = await submit_task.execute("Process background batch job")

    assert len(queue.dispatches) == 1
    assert queue.dispatches[0][1] == dynamic_queue
    assert queue.dispatches[0][0].task_id == task.task_id


@pytest.mark.anyio
async def test_dynamic_dispatch_does_not_depend_on_capability_queue_map() -> None:
    queue = FakeTaskQueue()
    target = "agent.tasks.dynamic-research-instance"
    candidate = make_candidate("research-agent-dynamic", capability="research", dispatch_target=target)
    finder = FakeCandidateFinder({"research": [candidate]})

    submit_task = SubmitTask(queue, candidate_finder=finder)

    # Temporarily clear the static queue map to prove zero runtime dependency during dynamic dispatch
    with patch.dict(CAPABILITY_QUEUE_MAP, {}, clear=True):
        assert len(CAPABILITY_QUEUE_MAP) == 0
        task = await submit_task.execute(
            "Survey distributed tracing protocols",
            capability=AgentCapability.RESEARCH,
        )

    assert len(queue.dispatches) == 1
    assert queue.dispatches[0][1] == target
    assert queue.dispatches[0][0].task_id == task.task_id


@pytest.mark.anyio
async def test_dispatch_with_multiple_candidates_selects_deterministically() -> None:
    queue = FakeTaskQueue()
    c_beta = make_candidate("worker-beta", capability="general", dispatch_target="queue.worker.beta")
    c_alpha = make_candidate(
        "worker-alpha", capability="general", dispatch_target="queue.worker.alpha"
    )
    c_gamma = make_candidate(
        "worker-gamma", capability="general", dispatch_target="queue.worker.gamma"
    )

    # Provide in non-alphabetical order
    finder = FakeCandidateFinder({"general": [c_beta, c_alpha, c_gamma]})
    submit_task = SubmitTask(queue, candidate_finder=finder)

    task = await submit_task.execute("Compile documentation")

    # worker-alpha is selected deterministically
    assert len(queue.dispatches) == 1
    dispatched_task, destination = queue.dispatches[0]
    assert dispatched_task.task_id == task.task_id
    assert destination == "queue.worker.alpha"


@pytest.mark.anyio
async def test_rejection_preserved_when_zero_candidates() -> None:
    queue = FakeTaskQueue()
    finder = FakeCandidateFinder({"general": []})
    events = FakeEventPublisher()

    submit_task = SubmitTask(queue, candidate_finder=finder, events=events)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await submit_task.execute("Execute standard batch task")

    assert exc_info.value.capability == AgentCapability.GENERAL
    assert "No eligible agent is currently available" in str(exc_info.value.message)

    # Rejection event published
    assert len(events.published) == 1
    assert events.published[0].state == "error"
    assert events.published[0].agent_id == "orchestrator"

    # Nothing dispatched or enqueued
    assert len(queue.dispatches) == 0
    assert len(queue.enqueued) == 0


@pytest.mark.anyio
async def test_ineligible_candidates_are_excluded_by_candidate_finder() -> None:
    now = datetime.now(UTC)
    # Register agents that are BUSY, STALE, and OFFLINE
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-busy",
                "capabilities": ["general"],
                "state": "working",
                "updated_at": now.isoformat(),
                "dispatch_target": "queue.busy",
            },
            {
                "agent_id": "worker-stale",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": (now - timedelta(seconds=120)).isoformat(),
                "dispatch_target": "queue.stale",
            },
            {
                "agent_id": "worker-offline",
                "capabilities": ["general"],
                "state": "offline",
                "updated_at": now.isoformat(),
                "dispatch_target": "queue.offline",
            },
        ]
    )
    finder = FindCandidatesByCapability(reader)
    queue = FakeTaskQueue()
    submit_task = SubmitTask(queue, candidate_finder=finder)

    # Zero eligible candidates should be found, resulting in rejection
    with pytest.raises(CapabilityUnavailableError):
        await submit_task.execute("Analyze telemetry", capability=AgentCapability.GENERAL)

    assert len(queue.dispatches) == 0


@pytest.mark.anyio
async def test_candidate_without_dispatch_target_fails_explicitly_without_queue_map_fallback() -> None:
    queue = FakeTaskQueue()
    # Candidate advertised no dispatch target
    candidate = make_candidate("worker-legacy", capability="general", dispatch_target=None)
    # Ensure dispatch_target is explicitly None
    object.__setattr__(candidate, "dispatch_target", None)
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(queue, candidate_finder=finder)

    with (
        patch.dict(CAPABILITY_QUEUE_MAP, {AgentCapability.GENERAL: "agent.tasks.general"}),
        pytest.raises(
            ValueError,
            match="Selected candidate 'worker-legacy' does not have a valid dispatch target advertised",
        ),
    ):
        await submit_task.execute("Process data pipeline")

    # Proves no silent fallback occurred:
    assert len(queue.dispatches) == 0
    assert len(queue.enqueued) == 0


@pytest.mark.anyio
async def test_candidate_with_blank_dispatch_target_fails_explicitly() -> None:
    queue = FakeTaskQueue()
    # Candidate with blank/empty string dispatch target
    candidate = make_candidate("worker-blank", capability="general")
    object.__setattr__(candidate, "dispatch_target", "   ")
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(queue, candidate_finder=finder)

    with pytest.raises(
        ValueError,
        match="Selected candidate 'worker-blank' does not have a valid dispatch target advertised",
    ):
        await submit_task.execute("Process data pipeline")

    assert len(queue.dispatches) == 0
    assert len(queue.enqueued) == 0


@pytest.mark.anyio
async def test_production_dynamic_dispatch_via_redis_gateway_never_depends_on_capability_queue_map() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.set = AsyncMock()
    gateway._client.rpush = AsyncMock()

    custom_target = "agent.tasks.production-custom-node"
    candidate = make_candidate("worker-prod", capability="general", dispatch_target=custom_target)
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(gateway, candidate_finder=finder)

    # Completely clear the capability queue map
    with patch.dict(CAPABILITY_QUEUE_MAP, {}, clear=True):
        assert len(CAPABILITY_QUEUE_MAP) == 0
        task = await submit_task.execute("Process production event")

    # Dynamic dispatch succeeded strictly using the candidate's advertised target
    gateway._client.rpush.assert_awaited_once_with(custom_target, task.to_json())
    gateway._client.set.assert_awaited_once_with(
        f"task.status:{task.task_id}",
        json.dumps({"task_id": task.task_id, "status": "pending"}),
    )


@pytest.mark.anyio
async def test_runtime_dispatch_prefers_explicit_task_dispatcher() -> None:
    class PureDispatcher:
        def __init__(self) -> None:
            self.dispatches: list[tuple[TaskRequested, str]] = []

        async def dispatch(self, task: TaskRequested, target: str) -> None:
            self.dispatches.append((task, target))

    dispatcher = PureDispatcher()
    assert isinstance(dispatcher, TaskDispatcher)
    candidate = make_candidate("worker-dispatch", capability="general", dispatch_target="queue.target")
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(dispatcher=dispatcher, candidate_finder=finder)
    task = await submit_task.execute("Process batch job")

    assert len(dispatcher.dispatches) == 1
    assert dispatcher.dispatches[0][0].task_id == task.task_id
    assert dispatcher.dispatches[0][1] == "queue.target"


@pytest.mark.anyio
async def test_legacy_non_dynamic_path_preserves_enqueue_for_isolated_callers() -> None:
    queue = FakeTaskQueue()
    # Non-dynamic mode: candidate_finder is None
    submit_task = SubmitTask(queue, candidate_finder=None)
    task = await submit_task.execute("Process legacy task")

    assert len(queue.enqueued) == 1
    assert queue.enqueued[0][0].task_id == task.task_id
    assert queue.enqueued[0][1] == AgentCapability.GENERAL


# ==============================================================================
# 6. RedisGateway & Task Status / Result Lifecycle Tests
# ==============================================================================


@pytest.mark.anyio
async def test_redis_gateway_dispatch_sets_pending_status_and_pushes_to_target() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.set = AsyncMock()
    gateway._client.rpush = AsyncMock()

    task = TaskRequested.create("Execute runtime task")
    target = "agent.tasks.custom-executor-1"

    await gateway.dispatch(task, target=target)

    # Sets status: pending
    gateway._client.set.assert_awaited_once_with(
        f"task.status:{task.task_id}",
        json.dumps({"task_id": task.task_id, "status": "pending"}),
    )
    # Pushes to advertised target
    gateway._client.rpush.assert_awaited_once_with(target, task.to_json())


@pytest.mark.anyio
async def test_redis_gateway_dispatch_rejects_empty_target() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    task = TaskRequested.create("Some prompt")

    with pytest.raises(ValueError, match="Dispatch target must be a non-empty string"):
        await gateway.dispatch(task, target="   ")


@pytest.mark.anyio
async def test_redis_gateway_result_lifecycle_after_dispatch() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    task_id = "test-task-lifecycle-123"

    # 1. While task is pending (written by dispatch)
    gateway._client.get = AsyncMock(
        side_effect=lambda key: (
            json.dumps({"task_id": task_id, "status": "pending"})
            if key == f"task.status:{task_id}"
            else None
        )
    )
    res_pending = await gateway.get_result(task_id)
    assert res_pending is not None
    assert res_pending["status"] == "pending"
    assert res_pending["output"] is None

    # 2. When task has completed result
    completed_data = {
        "task_id": task_id,
        "status": "completed",
        "output": "Execution successful",
        "acknowledged": True,
    }
    gateway._client.get = AsyncMock(
        side_effect=lambda key: (
            json.dumps(completed_data) if key == f"task.result:{task_id}" else None
        )
    )
    res_completed = await gateway.get_result(task_id)
    assert res_completed is not None
    assert res_completed["status"] == "completed"
    assert res_completed["output"] == "Execution successful"


# ==============================================================================
# 7. Presentation & API Route Tests
# ==============================================================================


@pytest.mark.anyio
async def test_api_create_task_dispatches_to_advertised_target() -> None:
    queue = FakeTaskQueue()
    target_queue = "agent.tasks.api-worker"
    candidate = make_candidate("api-worker", capability="general", dispatch_target=target_queue)
    finder = FakeCandidateFinder({"general": [candidate]})

    app = create_app(
        task_queue=queue,
        candidate_finder=finder,
        enforce_availability=True,
    )

    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "app": app,
            "path": "/api/tasks",
            "headers": [],
        }
    )

    response = await create_task(TaskInput(prompt="Check system status"), request)
    assert response["status"] == "accepted"
    task_id = response["task_id"]

    assert len(queue.dispatches) == 1
    dispatched_task, destination = queue.dispatches[0]
    assert dispatched_task.task_id == task_id
    assert destination == target_queue


@pytest.mark.anyio
async def test_api_create_task_returns_503_when_no_candidates() -> None:
    queue = FakeTaskQueue()
    finder = FakeCandidateFinder({"general": []})

    app = create_app(
        task_queue=queue,
        candidate_finder=finder,
        enforce_availability=True,
    )

    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "app": app,
            "path": "/api/tasks",
            "headers": [],
        }
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await create_task(TaskInput(prompt="Check system status"), request)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "capability_unavailable"
    assert exc_info.value.detail["capability"] == "general"
