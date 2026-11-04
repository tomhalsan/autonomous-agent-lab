import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from agent_contracts import TaskRequested
from orchestrator.domain.capability import AgentCapability
from orchestrator.infrastructure.redis_adapter import RedisGateway
from worker.infrastructure.redis_bus import RedisBus


@pytest.mark.anyio
async def test_redis_gateway_routes_general_capability() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.rpush = AsyncMock()
    task = TaskRequested.create("general task prompt")

    await gateway.enqueue(task, capability=AgentCapability.GENERAL)

    gateway._client.rpush.assert_awaited_once_with("agent.tasks.general", task.to_json())


@pytest.mark.anyio
async def test_redis_gateway_routes_research_capability() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.rpush = AsyncMock()
    task = TaskRequested.create("research task prompt")

    await gateway.enqueue(task, capability=AgentCapability.RESEARCH)

    gateway._client.rpush.assert_awaited_once_with("agent.tasks.research", task.to_json())


@pytest.mark.anyio
async def test_redis_gateway_defaults_to_general() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.rpush = AsyncMock()
    task = TaskRequested.create("default task prompt")

    await gateway.enqueue(task)

    gateway._client.rpush.assert_awaited_once_with("agent.tasks.general", task.to_json())


@pytest.mark.anyio
async def test_redis_gateway_invalid_capability_raises_value_error() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.rpush = AsyncMock()
    task = TaskRequested.create("bad capability prompt")

    with pytest.raises(ValueError, match="Unknown capability"):
        await gateway.enqueue(task, capability="unsupported_capability")  # type: ignore[arg-type]

    gateway._client.rpush.assert_not_awaited()


def test_redis_bus_defaults_to_general_queue() -> None:
    bus = RedisBus("redis://localhost:6379/0")
    assert bus.queue_name == "agent.tasks.general"


def test_redis_bus_custom_queue_configuration() -> None:
    bus = RedisBus("redis://localhost:6379/0", queue_name="agent.tasks.research")
    assert bus.queue_name == "agent.tasks.research"


@pytest.mark.anyio
async def test_redis_bus_consumes_from_default_queue() -> None:
    bus = RedisBus("redis://localhost:6379/0")
    task = TaskRequested.create("general worker prompt")
    bus.client.blpop = AsyncMock(return_value=("agent.tasks.general", task.to_json()))

    result = await bus.next_task()

    assert result.task_id == task.task_id
    assert result.prompt == "general worker prompt"
    bus.client.blpop.assert_awaited_once_with("agent.tasks.general", timeout=2)


@pytest.mark.anyio
async def test_redis_bus_consumes_from_configured_queue() -> None:
    bus = RedisBus("redis://localhost:6379/0", queue_name="agent.tasks.research")
    task = TaskRequested.create("research worker prompt")
    bus.client.blpop = AsyncMock(return_value=("agent.tasks.research", task.to_json()))

    result = await bus.next_task()

    assert result.task_id == task.task_id
    assert result.prompt == "research worker prompt"
    bus.client.blpop.assert_awaited_once_with("agent.tasks.research", timeout=2)


@pytest.mark.anyio
async def test_run_worker_uses_task_queue_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASK_QUEUE", "agent.tasks.research")
    monkeypatch.setenv("AGENT_ID", "research-worker-1")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_bus_cls.assert_called_once_with(
            "redis://localhost:6379/0",
            queue_name="agent.tasks.research",
        )


@pytest.mark.anyio
async def test_run_worker_defaults_to_general_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TASK_QUEUE", raising=False)
    monkeypatch.setenv("AGENT_ID", "worker-1")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_bus_cls.assert_called_once_with(
            "redis://localhost:6379/0",
            queue_name="agent.tasks.general",
        )

