import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from agent_contracts import InferenceExecutionTelemetry, TaskRequested
from orchestrator.domain.capability import AgentCapability
from orchestrator.infrastructure.redis_adapter import RedisGateway
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.redis_bus import RedisBus as ResearchRedisBus
from worker.infrastructure.redis_bus import RedisBus


@pytest.mark.anyio
async def test_redis_gateway_routes_general_capability() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.rpush = AsyncMock()
    gateway._client.set = AsyncMock()
    task = TaskRequested.create("general task prompt")

    await gateway.enqueue(task, capability=AgentCapability.GENERAL)

    gateway._client.rpush.assert_awaited_once_with("agent.tasks.general", task.to_json())
    gateway._client.set.assert_awaited_once()


@pytest.mark.anyio
async def test_redis_gateway_routes_research_capability() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.rpush = AsyncMock()
    gateway._client.set = AsyncMock()
    task = TaskRequested.create("research task prompt")

    await gateway.enqueue(task, capability=AgentCapability.RESEARCH)

    gateway._client.rpush.assert_awaited_once_with("agent.tasks.research", task.to_json())
    gateway._client.set.assert_awaited_once()


@pytest.mark.anyio
async def test_redis_gateway_defaults_to_general() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.rpush = AsyncMock()
    gateway._client.set = AsyncMock()
    task = TaskRequested.create("default task prompt")

    await gateway.enqueue(task)

    gateway._client.rpush.assert_awaited_once_with("agent.tasks.general", task.to_json())
    gateway._client.set.assert_awaited_once()


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


@pytest.mark.anyio
async def test_redis_gateway_get_result_completed() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    result_data = {
        "task_id": "task-res-1",
        "status": "completed",
        "output": "Found 3 articles",
        "acknowledged": True,
    }
    gateway._client.get = AsyncMock(return_value=json.dumps(result_data))

    res = await gateway.get_result("task-res-1")

    assert res == result_data
    gateway._client.get.assert_awaited_once_with("task.result:task-res-1")


@pytest.mark.anyio
async def test_redis_gateway_get_result_pending() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    # First get() for task.result returns None, second for task.status returns status json
    gateway._client.get = AsyncMock(
        side_effect=[None, json.dumps({"task_id": "task-res-2", "status": "pending"})]
    )

    res = await gateway.get_result("task-res-2")

    assert res == {
        "task_id": "task-res-2",
        "status": "pending",
        "output": None,
    }


@pytest.mark.anyio
async def test_redis_gateway_get_result_unknown() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.get = AsyncMock(return_value=None)
    gateway.events = AsyncMock(return_value=[])

    res = await gateway.get_result("unknown-task-id")

    assert res is None


@pytest.mark.anyio
async def test_research_redis_bus_save_result() -> None:
    bus = ResearchRedisBus("redis://localhost:6379/0")
    bus.client.set = AsyncMock()

    result = ResearchResult(task_id="res-abc-123", output="Full research text")
    await bus.save(result)

    bus.client.set.assert_awaited_once()
    key, val = bus.client.set.await_args[0]
    assert key == "task.result:res-abc-123"
    loaded = json.loads(val)
    assert loaded["task_id"] == "res-abc-123"
    assert loaded["status"] == "completed"
    assert loaded["output"] == "Full research text"
    assert loaded["acknowledged"] is True


@pytest.mark.anyio
async def test_redis_gateway_agents_parses_capability_metadata() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.smembers = AsyncMock(return_value={"worker-1", "research-agent-1"})  # type: ignore[method-assign]

    async def mock_hgetall(key: str) -> dict[str, str]:
        if "research" in key:
            return {
                "agent_id": "research-agent-1",
                "agent_type": "research-agent",
                "capabilities": '["research"]',
                "providers": '{"research_executor": "tavily", "inference_provider": "cloudflare"}',
                "research_executor": "tavily",
                "inference_provider": "cloudflare",
                "state": "idle",
                "detail": "Waiting for work",
            }
        return {
            "agent_id": "worker-1",
            "agent_type": "worker",
            "capabilities": '["general"]',
            "providers": "{}",
            "state": "working",
            "detail": "Processing task",
        }

    gateway._client.hgetall = AsyncMock(side_effect=mock_hgetall)  # type: ignore[method-assign]

    agents = await gateway.agents()
    assert len(agents) == 2

    # Sorted by agent_id: research-agent-1, worker-1
    research_agent = agents[0]
    worker = agents[1]

    assert research_agent["agent_id"] == "research-agent-1"
    assert research_agent["agent_type"] == "research-agent"
    assert research_agent["capabilities"] == ["research"]
    assert research_agent["research_executor"] == "tavily"
    assert research_agent["inference_provider"] == "cloudflare"
    assert research_agent["providers"] == {
        "research_executor": "tavily",
        "inference_provider": "cloudflare",
    }
    assert research_agent["state"] == "idle"

    assert worker["agent_id"] == "worker-1"
    assert worker["agent_type"] == "worker"
    assert worker["capabilities"] == ["general"]
    assert worker["providers"] == {}
    assert worker["state"] == "working"


@pytest.mark.anyio
async def test_redis_gateway_agents_legacy_fallback() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.smembers = AsyncMock(return_value={"legacy-agent"})  # type: ignore[method-assign]
    gateway._client.hgetall = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "agent_id": "legacy-agent",
            "state": "idle",
            "detail": "Legacy worker",
        }
    )

    agents = await gateway.agents()
    assert len(agents) == 1
    assert agents[0]["agent_id"] == "legacy-agent"
    assert agents[0]["capabilities"] == []
    assert agents[0]["providers"] == {}
    assert agents[0]["state"] == "idle"


# ==============================================================================
# RedisGateway Task Inference Telemetry Query Tests (Issue #18 Increment 1)
# ==============================================================================


@pytest.mark.anyio
async def test_redis_gateway_task_inference_telemetry_newest_successful() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    telem_cf_failed = InferenceExecutionTelemetry(
        provider_name="cloudflare",
        attempt_index=0,
        latency_ms=120.0,
        status="fallback_eligible_failure",
        failure_reason="provider_unavailable_error",
    )
    telem_gemini_success = InferenceExecutionTelemetry(
        provider_name="gemini",
        attempt_index=1,
        latency_ms=85.5,
        status="succeeded",
        model_name="gemini-2.5-flash",
        finish_reason="stop",
    )

    # xrevrange returns newest entries first
    gateway._client.xrevrange = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            (
                "1000-2",
                {
                    "task_id": "task-test-1",
                    "inference_telemetry": json.dumps(telem_gemini_success.to_dict()),
                },
            ),
            (
                "1000-1",
                {
                    "task_id": "task-test-1",
                    "inference_telemetry": json.dumps(telem_cf_failed.to_dict()),
                },
            ),
        ]
    )

    result = await gateway.task_inference_telemetry("task-test-1")
    assert result is not None
    assert result["task_id"] == "task-test-1"
    assert result["inference_telemetry"]["provider_name"] == "gemini"
    assert result["inference_telemetry"]["status"] == "succeeded"
    assert result["inference_telemetry"]["attempt_index"] == 1
    assert result["inference_telemetry"]["latency_ms"] == 85.5
    gateway._client.xrevrange.assert_awaited_once_with("agent.events")


@pytest.mark.anyio
async def test_redis_gateway_task_inference_telemetry_ignores_failed() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    telem_failed = InferenceExecutionTelemetry(
        provider_name="cloudflare",
        attempt_index=0,
        latency_ms=120.0,
        status="fallback_eligible_failure",
        failure_reason="provider_unavailable_error",
    )
    gateway._client.xrevrange = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            (
                "1000-1",
                {
                    "task_id": "task-test-failed",
                    "inference_telemetry": json.dumps(telem_failed.to_dict()),
                },
            )
        ]
    )

    result = await gateway.task_inference_telemetry("task-test-failed")
    assert result is None


@pytest.mark.anyio
async def test_redis_gateway_task_inference_telemetry_ignores_other_tasks() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    telem_success = InferenceExecutionTelemetry(
        provider_name="gemini",
        attempt_index=0,
        latency_ms=90.0,
        status="succeeded",
        model_name="gemini-2.5-flash",
    )
    gateway._client.xrevrange = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            (
                "1000-1",
                {
                    "task_id": "task-other",
                    "inference_telemetry": json.dumps(telem_success.to_dict()),
                },
            )
        ]
    )

    result = await gateway.task_inference_telemetry("task-target")
    assert result is None


@pytest.mark.anyio
async def test_redis_gateway_task_inference_telemetry_handles_malformed_and_invalid() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.xrevrange = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            # Malformed JSON
            (
                "1000-3",
                {
                    "task_id": "task-malformed",
                    "inference_telemetry": "{invalid json",
                },
            ),
            # Non-dict JSON (e.g. JSON array or int)
            (
                "1000-2",
                {
                    "task_id": "task-malformed",
                    "inference_telemetry": json.dumps(["not", "an", "object"]),
                },
            ),
            # Dict missing required fields according to InferenceExecutionTelemetry
            (
                "1000-1",
                {
                    "task_id": "task-malformed",
                    "inference_telemetry": json.dumps({"incomplete": "dict"}),
                },
            ),
        ]
    )

    result = await gateway.task_inference_telemetry("task-malformed")
    assert result is None


@pytest.mark.anyio
async def test_redis_gateway_task_inference_telemetry_returns_none_when_empty() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.xrevrange = AsyncMock(return_value=[])  # type: ignore[method-assign]

    result = await gateway.task_inference_telemetry("task-any")
    assert result is None


