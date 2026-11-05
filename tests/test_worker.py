import asyncio
import json
from typing import NoReturn
from unittest.mock import AsyncMock, patch

import pytest
import redis.asyncio as redis
from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    TaskRequested,
    TaskResult,
)
from orchestrator.infrastructure.redis_adapter import RedisGateway
from worker.application.processor import TaskExecutionError, TaskProcessor
from worker.infrastructure.redis_bus import RedisBus, retry_with_backoff


class FakeResultStore:
    def __init__(self) -> None:
        self.saved: dict[str, TaskResult] = {}

    async def save(self, result: TaskResult) -> None:
        self.saved[result.task_id] = result


class FakeMessageBus:
    def __init__(self) -> None:
        self.published: list[AgentEvent] = []
        self.saved: dict[str, TaskResult] = {}

    async def publish(self, event: AgentEvent) -> None:
        self.published.append(event)

    async def save(self, result: TaskResult) -> None:
        self.saved[result.task_id] = result

    async def next_task(self) -> TaskRequested:
        raise NotImplementedError


class FailingTaskProcessor(TaskProcessor):
    def __init__(self, agent_id: str, bus: FakeMessageBus, error_to_raise: Exception) -> None:
        super().__init__(agent_id, bus)
        self.error_to_raise = error_to_raise

    async def process(self, task: TaskRequested) -> NoReturn:
        raise self.error_to_raise


class FastTaskProcessor(TaskProcessor):
    async def process(self, task: TaskRequested) -> None:
        # Fast no-op for tests
        return None


@pytest.mark.anyio
async def test_successful_task_processing_transitions() -> None:
    bus = FakeMessageBus()
    processor = FastTaskProcessor("worker-1", bus)
    task = TaskRequested.create("test prompt")

    await processor.run_once(task)

    assert len(bus.published) == 2
    working_event, idle_event = bus.published

    assert working_event.state == "working"
    assert working_event.task_id == task.task_id

    assert idle_event.state == "idle"
    assert idle_event.task_id == task.task_id
    assert "Completed" in idle_event.detail


@pytest.mark.anyio
async def test_failed_task_processing_transitions_and_restores_idle() -> None:
    bus = FakeMessageBus()
    error = ValueError("Invalid prompt parameters")
    processor = FailingTaskProcessor("worker-1", bus, error)
    task = TaskRequested.create("broken prompt")

    with pytest.raises(TaskExecutionError) as exc_info:
        await processor.run_once(task)

    assert exc_info.value.original_error == error
    assert exc_info.value.task_id == task.task_id

    # State transitions must be: working -> error -> idle
    assert len(bus.published) == 3
    working_event, error_event, restored_idle_event = bus.published

    # 1. Working
    assert working_event.state == "working"
    assert working_event.task_id == task.task_id

    # 2. Error recorded in history with task id
    assert error_event.state == "error"
    assert error_event.task_id == task.task_id
    assert "Invalid prompt parameters" in error_event.detail

    # 3. Restored to idle so worker is available for next task
    assert restored_idle_event.state == "idle"
    assert restored_idle_event.detail == "Waiting for work"


@pytest.mark.anyio
async def test_retry_with_backoff_transient_failure() -> None:
    call_count = 0

    async def flaky_operation() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise redis.ConnectionError("Connection dropped")
        return "success"

    result = await retry_with_backoff(
        flaky_operation, max_retries=3, initial_delay=0.01, backoff_factor=1.5
    )
    assert result == "success"
    assert call_count == 3


@pytest.mark.anyio
async def test_retry_with_backoff_exhausts_and_raises() -> None:
    call_count = 0

    async def always_failing() -> None:
        nonlocal call_count
        call_count += 1
        raise redis.ConnectionError("Persistent connection failure")

    with pytest.raises(redis.ConnectionError):
        await retry_with_backoff(
            always_failing, max_retries=3, initial_delay=0.01, backoff_factor=1.5
        )

    assert call_count == 3


@pytest.mark.anyio
async def test_retry_with_backoff_does_not_retry_unexpected_defects() -> None:
    call_count = 0

    async def code_bug() -> None:
        nonlocal call_count
        call_count += 1
        raise TypeError("Unexpected code type error")

    with pytest.raises(TypeError, match="Unexpected code type error"):
        await retry_with_backoff(code_bug, max_retries=3, initial_delay=0.01)

    # Must fail immediately on attempt 1 without retrying programming defects
    assert call_count == 1


@pytest.mark.anyio
async def test_worker_redis_bus_register() -> None:
    bus = RedisBus("redis://localhost:6379/0", queue_name="agent.tasks.general")
    bus.client.sadd = AsyncMock()  # type: ignore[method-assign]
    bus.client.hset = AsyncMock()  # type: ignore[method-assign]

    metadata = AgentCapabilityMetadata(
        agent_id="worker-test",
        capabilities=["general"],
        agent_type="worker",
    )

    await bus.register(metadata)

    bus.client.sadd.assert_awaited_once_with("agent.ids", "worker-test")
    bus.client.hset.assert_awaited_once()
    key, kwargs = bus.client.hset.call_args[0][0], bus.client.hset.call_args[1]
    assert key == "agent.state:worker-test"
    mapping = kwargs["mapping"]
    assert mapping["agent_id"] == "worker-test"
    assert mapping["agent_type"] == "worker"
    assert mapping["capabilities"] == '["general"]'
    assert mapping["providers"] == "{}"


@pytest.mark.anyio
async def test_run_worker_registers_capability_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ID", "worker-1")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_bus.register.assert_awaited_once()
        metadata = mock_bus.register.call_args[0][0]
        assert isinstance(metadata, AgentCapabilityMetadata)
        assert metadata.agent_id == "worker-1"
        assert metadata.capabilities == ["general"]
        assert metadata.agent_type == "worker"

        mock_bus.publish.assert_awaited_once()
        event = mock_bus.publish.call_args[0][0]
        assert event.agent_id == "worker-1"
        assert event.state == "idle"


# --- Processor Result Persistence Tests ---


@pytest.mark.anyio
async def test_task_processor_stores_result_on_success() -> None:
    bus = FakeMessageBus()
    store = FakeResultStore()
    processor = FastTaskProcessor("worker-1", bus, result_store=store)
    task = TaskRequested.create("Build docker image")

    await processor.run_once(task)

    assert task.task_id in store.saved
    saved = store.saved[task.task_id]
    assert saved.task_id == task.task_id
    assert saved.status == "completed"
    assert saved.output == f"Completed: {task.prompt}"
    assert saved.acknowledged is True

    # Working and idle events preserved
    assert len(bus.published) == 2
    assert bus.published[0].state == "working"
    assert bus.published[0].task_id == task.task_id
    assert bus.published[1].state == "idle"
    assert bus.published[1].task_id == task.task_id
    assert f"Completed: {task.prompt}" in bus.published[1].detail


@pytest.mark.anyio
async def test_task_processor_error_does_not_store_result() -> None:
    bus = FakeMessageBus()
    store = FakeResultStore()
    error = RuntimeError("Execution crashed")
    processor = FailingTaskProcessor("worker-1", bus, error)
    task = TaskRequested.create("Failing general task")

    processor.result_store = store

    with pytest.raises(TaskExecutionError):
        await processor.run_once(task)

    assert len(store.saved) == 0

    # Lifecycle events: working -> error -> idle
    assert len(bus.published) == 3
    assert bus.published[0].state == "working"
    assert bus.published[1].state == "error"
    assert bus.published[2].state == "idle"


@pytest.mark.anyio
async def test_task_processor_with_custom_output() -> None:
    class CustomOutputProcessor(TaskProcessor):
        async def process(self, task: TaskRequested) -> str:
            return f"Calculated metrics for: {task.prompt}"

    bus = FakeMessageBus()
    store = FakeResultStore()
    processor = CustomOutputProcessor("worker-1", bus, result_store=store)
    task = TaskRequested.create("Aggregate stats")

    await processor.run_once(task)

    assert task.task_id in store.saved
    assert store.saved[task.task_id].output == f"Calculated metrics for: {task.prompt}"


@pytest.mark.anyio
async def test_task_processor_saves_via_bus_if_result_store_omitted() -> None:
    bus = FakeMessageBus()
    processor = FastTaskProcessor("worker-1", bus)
    task = TaskRequested.create("Auto save task")

    await processor.run_once(task)

    assert task.task_id in bus.saved
    assert bus.saved[task.task_id].status == "completed"


# --- Infrastructure RedisBus Save Tests ---


@pytest.mark.anyio
async def test_worker_redis_bus_save() -> None:
    bus = RedisBus("redis://localhost:6379/0", queue_name="agent.tasks.general")
    bus.client.set = AsyncMock()  # type: ignore[method-assign]

    result = TaskResult(task_id="task-work-99", output="Rendered artifact")
    await bus.save(result)

    bus.client.set.assert_awaited_once()
    key, val = bus.client.set.await_args[0]
    assert key == "task.result:task-work-99"
    data = json.loads(val)
    assert data == {
        "task_id": "task-work-99",
        "status": "completed",
        "output": "Rendered artifact",
        "acknowledged": True,
    }


@pytest.mark.anyio
async def test_run_worker_wires_result_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ID", "worker-1")

    with (
        patch("worker.main.RedisBus") as mock_bus_cls,
        patch("worker.main.TaskProcessor") as mock_processor_cls,
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_processor_cls.assert_called_once()
        _, kwargs = mock_processor_cls.call_args
        assert kwargs.get("result_store") == mock_bus


# --- Orchestrator Retrieval Integration Test ---


@pytest.mark.anyio
async def test_general_worker_task_produces_retrievable_result_in_orchestrator() -> None:
    redis_storage: dict[str, str] = {}

    worker_bus = RedisBus("redis://localhost:6379/0", queue_name="agent.tasks.general")
    worker_bus.client.sadd = AsyncMock()  # type: ignore[method-assign]
    worker_bus.client.hset = AsyncMock()  # type: ignore[method-assign]
    worker_bus.client.xadd = AsyncMock()  # type: ignore[method-assign]

    async def mock_set(key: str, value: str) -> None:
        redis_storage[key] = value

    worker_bus.client.set = AsyncMock(side_effect=mock_set)  # type: ignore[method-assign]

    processor = FastTaskProcessor("worker-1", worker_bus, result_store=worker_bus)
    task = TaskRequested.create("Verify compile output")

    await processor.run_once(task)

    # Now verify that Orchestrator's RedisGateway retrieves the exact completed result
    gateway = RedisGateway("redis://localhost:6379/0")

    async def mock_get(key: str) -> str | None:
        return redis_storage.get(key)

    gateway._client.get = AsyncMock(side_effect=mock_get)  # type: ignore[method-assign]

    retrieved = await gateway.get_result(task.task_id)
    assert retrieved is not None
    assert retrieved["task_id"] == task.task_id
    assert retrieved["status"] == "completed"
    assert retrieved["output"] == f"Completed: {task.prompt}"
    assert retrieved["acknowledged"] is True


