import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from agent_contracts import (
    AgentAvailabilityState,
    AgentCapabilityMetadata,
    AgentEvent,
    AgentExecutionState,
    AgentHeartbeat,
    HeartbeatRunner,
    evaluate_agent_record,
    parse_heartbeat_interval,
)
from orchestrator.presentation.api import dashboard
from research_agent.infrastructure.redis_bus import RedisBus as ResearchRedisBus
from worker.infrastructure.redis_bus import RedisBus as WorkerRedisBus

# ==============================================================================
# 1. Configuration & Validation Tests
# ==============================================================================


def test_parse_heartbeat_interval_defaults_and_valid_values() -> None:
    assert parse_heartbeat_interval(None) == 10.0
    assert parse_heartbeat_interval(None, default=15.0) == 15.0
    assert parse_heartbeat_interval("") == 10.0
    assert parse_heartbeat_interval("   ") == 10.0
    assert parse_heartbeat_interval(5) == 5.0
    assert parse_heartbeat_interval(12.5) == 12.5
    assert parse_heartbeat_interval("15.2") == 15.2


@pytest.mark.parametrize(
    "invalid_val",
    [
        0,
        "0",
        -1,
        "-5.5",
        "invalid",
        "abc",
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_parse_heartbeat_interval_rejects_non_positive_and_invalid(invalid_val: object) -> None:
    with pytest.raises(ValueError, match="Invalid heartbeat interval"):
        parse_heartbeat_interval(invalid_val)  # type: ignore[arg-type]


def test_heartbeat_runner_validates_inputs() -> None:
    async def dummy(hb: AgentHeartbeat) -> None:
        pass

    with pytest.raises(ValueError, match="agent_id must not be empty"):
        HeartbeatRunner("", dummy)

    with pytest.raises(ValueError, match="agent_id must not be empty"):
        HeartbeatRunner("   ", dummy)

    with pytest.raises(ValueError, match="Invalid heartbeat interval"):
        HeartbeatRunner("worker-1", dummy, interval_seconds=-1.0)


# ==============================================================================
# 2. HeartbeatRunner Lifecycle & Cancellation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_heartbeat_runner_start_stop_lifecycle() -> None:
    published: list[AgentHeartbeat] = []

    async def mock_publish(hb: AgentHeartbeat) -> None:
        published.append(hb)

    runner = HeartbeatRunner("worker-1", mock_publish, interval_seconds=0.05)
    assert not runner.is_running

    runner.start()
    assert runner.is_running

    # Multiple start calls are idempotent
    runner.start()
    assert runner.is_running

    await asyncio.sleep(0.12)
    await runner.stop()
    assert not runner.is_running

    # Verify at least one heartbeat was published and loop stopped
    count_after_stop = len(published)
    assert count_after_stop >= 1

    await asyncio.sleep(0.08)
    assert len(published) == count_after_stop


@pytest.mark.anyio
async def test_heartbeat_runner_async_context_manager() -> None:
    published: list[AgentHeartbeat] = []

    async def mock_publish(hb: AgentHeartbeat) -> None:
        published.append(hb)

    async with HeartbeatRunner("worker-1", mock_publish, interval_seconds=0.05) as runner:
        assert runner.is_running
        await asyncio.sleep(0.08)

    assert not runner.is_running
    assert len(published) >= 1


@pytest.mark.anyio
async def test_heartbeat_runner_publish_once_captures_state() -> None:
    state = "idle"
    published: list[AgentHeartbeat] = []

    async def mock_publish(hb: AgentHeartbeat) -> None:
        published.append(hb)

    runner = HeartbeatRunner(
        "worker-1",
        mock_publish,
        get_state=lambda: state,
        interval_seconds=10.0,
    )

    hb1 = await runner.publish_once()
    assert hb1.agent_id == "worker-1"
    assert hb1.execution_state == "idle"

    state = "working"
    hb2 = await runner.publish_once()
    assert hb2.agent_id == "worker-1"
    assert hb2.execution_state == "working"


# ==============================================================================
# 3. General Worker & Research Agent Periodic Refresh Tests
# ==============================================================================


@pytest.mark.anyio
async def test_general_worker_periodically_refreshes_liveness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_ID", "worker-live-1")
    monkeypatch.setenv("HEARTBEAT_INTERVAL_SECONDS", "0.04")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus.current_state = "idle"
        mock_bus_cls.return_value = mock_bus

        heartbeats: list[AgentHeartbeat] = []

        async def capture_heartbeat(hb: AgentHeartbeat) -> None:
            heartbeats.append(hb)

        mock_bus.heartbeat.side_effect = capture_heartbeat

        # Wait long enough for ~3 heartbeats before raising CancelledError
        async def wait_and_cancel():
            await asyncio.sleep(0.13)
            raise asyncio.CancelledError()

        mock_bus.next_task.side_effect = wait_and_cancel

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        # Registration + initial idle event occurred
        mock_bus.register.assert_awaited_once()
        mock_bus.publish.assert_awaited_once()

        # Heartbeat loop published periodically
        assert len(heartbeats) >= 2
        for hb in heartbeats:
            assert hb.agent_id == "worker-live-1"
            assert hb.execution_state == "idle"
            assert hb.occurred_at != ""


@pytest.mark.anyio
async def test_research_agent_periodically_refreshes_liveness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_ID", "research-live-1")
    monkeypatch.setenv("HEARTBEAT_INTERVAL_SECONDS", "0.04")

    with patch("research_agent.main.RedisBus") as mock_bus_cls, patch(
        "research_agent.main.ResearchProcessor"
    ):
        mock_bus = AsyncMock()
        mock_bus.current_state = "idle"
        mock_bus_cls.return_value = mock_bus

        heartbeats: list[AgentHeartbeat] = []

        async def capture_heartbeat(hb: AgentHeartbeat) -> None:
            heartbeats.append(hb)

        mock_bus.heartbeat.side_effect = capture_heartbeat

        async def wait_and_cancel():
            await asyncio.sleep(0.13)
            raise asyncio.CancelledError()

        mock_bus.next_task.side_effect = wait_and_cancel

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_bus.register.assert_awaited_once()
        mock_bus.publish.assert_awaited_once()

        assert len(heartbeats) >= 2
        for hb in heartbeats:
            assert hb.agent_id == "research-live-1"
            assert hb.execution_state == "idle"
            assert hb.occurred_at != ""


# ==============================================================================
# 4. Availability Domain Eligibility & Preservation Tests
# ==============================================================================


def test_repeated_heartbeat_updates_keep_idle_agent_eligible_beyond_90s() -> None:
    t0 = datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)

    # Agent without periodic heartbeats (timestamp remains at t0)
    stale_record = {
        "agent_id": "worker-1",
        "state": "idle",
        "occurred_at": t0.isoformat(),
    }
    # Evaluated at t0 + 95 seconds: exceeds 90s offline threshold
    offline_eval = evaluate_agent_record(stale_record, as_of=t0 + timedelta(seconds=95))
    assert offline_eval.state == AgentAvailabilityState.OFFLINE
    assert offline_eval.is_eligible is False

    # Agent with periodic heartbeats: updated at t0 + 90s
    refreshed_record = {
        "agent_id": "worker-1",
        "state": "idle",
        "occurred_at": t0.isoformat(),
        "updated_at": (t0 + timedelta(seconds=90)).isoformat(),
    }
    # Evaluated at t0 + 95s: only 5 seconds since last heartbeat!
    ready_eval = evaluate_agent_record(refreshed_record, as_of=t0 + timedelta(seconds=95))
    assert ready_eval.state == AgentAvailabilityState.READY
    assert ready_eval.is_eligible is True
    assert ready_eval.heartbeat_age_seconds == 5.0


def test_heartbeat_while_working_preserves_working_and_busy() -> None:
    t0 = datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)

    # While working, heartbeat sends execution_state="working" and updates timestamp
    working_record = {
        "agent_id": "worker-1",
        "state": "working",
        "updated_at": (t0 + timedelta(seconds=45)).isoformat(),
    }
    eval_working = evaluate_agent_record(working_record, as_of=t0 + timedelta(seconds=48))
    # Heartbeat is fresh (3s old), so not stale/offline, but correctly BUSY
    assert eval_working.state == AgentAvailabilityState.BUSY
    assert eval_working.execution_state == AgentExecutionState.WORKING
    assert eval_working.is_eligible is False


def test_heartbeat_does_not_overwrite_error_state() -> None:
    t0 = datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)

    # An agent that encountered a failure has state="error"
    error_record = {
        "agent_id": "worker-error",
        "state": "error",
        "updated_at": (t0 + timedelta(seconds=10)).isoformat(),
    }
    eval_error = evaluate_agent_record(error_record, as_of=t0 + timedelta(seconds=12))
    assert eval_error.execution_state == AgentExecutionState.ERROR
    assert eval_error.is_eligible is False


# ==============================================================================
# 5. RedisBus State Updates & Metadata Preservation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_worker_redis_bus_heartbeat_updates_state_and_preserves_metadata() -> None:
    bus = WorkerRedisBus("redis://localhost:6379/0", queue_name="agent.tasks.general")
    bus.client.sadd = AsyncMock()  # type: ignore[method-assign]
    bus.client.hset = AsyncMock()  # type: ignore[method-assign]
    bus.client.xadd = AsyncMock()  # type: ignore[method-assign]

    # 1. Registration
    meta = AgentCapabilityMetadata(
        agent_id="worker-meta-test",
        agent_type="worker",
        capabilities=["general"],
        providers={"inference_provider": "stub"},
        dispatch_target="agent.tasks.general",
    )
    await bus.register(meta)

    # 2. Publish initial event
    await bus.publish(AgentEvent.create("worker-meta-test", "idle", "Waiting for work"))
    assert bus.current_state == "idle"

    # 3. Publish heartbeat
    hb = AgentHeartbeat.create("worker-meta-test", execution_state="idle")
    await bus.heartbeat(hb)

    # Verify hset was called with updated_at and does NOT overwrite state
    last_call = bus.client.hset.call_args_list[-1]
    key, kwargs = last_call[0][0], last_call[1]
    assert key == "agent.state:worker-meta-test"
    mapping = kwargs["mapping"]
    assert mapping["updated_at"] == hb.occurred_at
    assert "state" not in mapping


@pytest.mark.anyio
async def test_research_redis_bus_heartbeat_refreshes_timestamp_without_overwriting_state() -> None:
    bus = ResearchRedisBus("redis://localhost:6379/0", queue_name="agent.tasks.research")
    bus.client.sadd = AsyncMock()  # type: ignore[method-assign]
    bus.client.hset = AsyncMock()  # type: ignore[method-assign]

    hb = AgentHeartbeat.create("research-1", execution_state="working")
    await bus.heartbeat(hb)

    last_call = bus.client.hset.call_args_list[-1]
    mapping = last_call[1]["mapping"]
    assert "state" not in mapping
    assert mapping["updated_at"] == hb.occurred_at


@pytest.mark.anyio
async def test_older_heartbeat_cannot_overwrite_newer_execution_state_event() -> None:
    """Deterministic concurrency test proving an older delayed heartbeat cannot overwrite

    a newer execution-state transition (both idle->working and working->idle transitions).
    """
    bus = WorkerRedisBus("redis://localhost:6379/0")

    # In-memory dictionary simulating the Redis hash agent.state:worker-1
    stored_redis_hash: dict[str, str] = {}

    async def mock_hset(name: str, mapping: dict[str, str]) -> int:
        stored_redis_hash.update(mapping)
        return len(mapping)

    bus.client.hset = AsyncMock(side_effect=mock_hset)
    bus.client.sadd = AsyncMock()
    bus.client.xadd = AsyncMock()

    # 1. Agent starts up in idle state at t0
    t0 = "2026-11-05T12:00:00+00:00"
    init_event = AgentEvent.create("worker-1", "idle", "Waiting for work")
    object.__setattr__(init_event, "occurred_at", t0)
    await bus.publish(init_event)

    assert stored_redis_hash["state"] == "idle"
    assert stored_redis_hash["updated_at"] == t0

    # 2. Heartbeat loop captures state at t1 (while worker is idle)
    t1 = "2026-11-05T12:00:10+00:00"
    delayed_idle_heartbeat = AgentHeartbeat.create(
        "worker-1", execution_state="idle", occurred_at=t1
    )

    # 3. Before the delayed heartbeat write executes, a task arrives at t2!
    # TaskProcessor authoritatively transitions worker state to "working"
    t2 = "2026-11-05T12:00:12+00:00"
    working_event = AgentEvent.create(
        "worker-1", "working", "Processing task", task_id="task-123"
    )
    object.__setattr__(working_event, "occurred_at", t2)
    await bus.publish(working_event)

    assert stored_redis_hash["state"] == "working"
    assert stored_redis_hash["task_id"] == "task-123"

    # 4. Now the delayed heartbeat (captured at t1 when idle) writes to Redis
    await bus.heartbeat(delayed_idle_heartbeat)

    # CRUCIAL ASSERTION: Stored state in Redis MUST remain "working"!
    assert stored_redis_hash["state"] == "working"
    assert stored_redis_hash["task_id"] == "task-123"

    # Availability must evaluate as BUSY (working + fresh heartbeat timestamp)
    eval_busy = evaluate_agent_record(
        stored_redis_hash, as_of="2026-11-05T12:00:13+00:00"
    )
    assert eval_busy.state == AgentAvailabilityState.BUSY
    assert eval_busy.execution_state == AgentExecutionState.WORKING
    assert eval_busy.is_eligible is False

    # 5. Reverse scenario:
    # Heartbeat captures state at t3 while worker is working
    t3 = "2026-11-05T12:00:20+00:00"
    delayed_working_heartbeat = AgentHeartbeat.create(
        "worker-1", execution_state="working", occurred_at=t3
    )

    # Task finishes: TaskProcessor authoritatively transitions worker to "idle" at t4
    t4 = "2026-11-05T12:00:22+00:00"
    idle_event = AgentEvent.create(
        "worker-1", "idle", "Completed task", task_id="task-123"
    )
    object.__setattr__(idle_event, "occurred_at", t4)
    await bus.publish(idle_event)

    assert stored_redis_hash["state"] == "idle"

    # Now the delayed working heartbeat (captured at t3) writes to Redis
    await bus.heartbeat(delayed_working_heartbeat)

    # CRUCIAL ASSERTION: Stored state MUST remain "idle"! It must not be reverted to "working"!
    assert stored_redis_hash["state"] == "idle"

    # Availability must evaluate as READY (idle + fresh timestamp)
    eval_ready = evaluate_agent_record(
        stored_redis_hash, as_of="2026-11-05T12:00:23+00:00"
    )
    assert eval_ready.state == AgentAvailabilityState.READY
    assert eval_ready.execution_state == AgentExecutionState.IDLE
    assert eval_ready.is_eligible is True


@pytest.mark.anyio
async def test_heartbeat_preserves_registration_metadata_in_candidate_query() -> None:
    from orchestrator.application import FindCandidatesByCapability
    from orchestrator.domain import AgentCapability

    from tests.test_application import FakeStateReader

    now = datetime.now(UTC)
    # Simulate Redis hash state where an initial registration occurred and heartbeats update updated_at
    agent_hash = {
        "agent_id": "worker-hb-query",
        "agent_type": "worker",
        "capabilities": ["general"],
        "providers": {"inference_provider": "stub"},
        "dispatch_target": "agent.tasks.worker-hb-query",
        "state": "idle",
        "occurred_at": (now - timedelta(seconds=120)).isoformat(),  # initial registration 120s ago
        "updated_at": (now - timedelta(seconds=5)).isoformat(),  # heartbeat refreshed 5s ago
    }
    reader = FakeStateReader(agents_data=[agent_hash])
    finder = FindCandidatesByCapability(reader)

    candidates = await finder.execute(AgentCapability.GENERAL, as_of=now)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.agent_id == "worker-hb-query"
    assert candidate.dispatch_target == "agent.tasks.worker-hb-query"
    assert candidate.availability.state == AgentAvailabilityState.READY
    assert candidate.availability.is_eligible is True
    assert candidate.availability.heartbeat_age_seconds == 5.0


# ==============================================================================
# 6. UI Error Fix Tests
# ==============================================================================


@pytest.mark.anyio
async def test_dashboard_renders_structured_error_message_instead_of_object_object() -> None:
    html = await dashboard()
    # Check that index.html explicitly checks for errJson.detail.message or typeof object
    assert "errJson.detail.message" in html
    assert "typeof errJson.detail === 'object'" in html
    # Ensure raw object is not passed to Error constructor
    assert "throw new Error(errJson.detail)" not in html
