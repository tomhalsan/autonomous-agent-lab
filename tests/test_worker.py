import asyncio
import json
from typing import NoReturn
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import redis.asyncio as redis
from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    FallbackTransition,
    TaskRequested,
    TaskResult,
)
from orchestrator.infrastructure.redis_adapter import RedisGateway
from worker.application.processor import TaskExecutionError, TaskProcessor
from worker.infrastructure.cloudflare_inference import (
    DEFAULT_CLOUDFLARE_AI_MODEL,
    CloudflareInferenceProvider,
)
from worker.infrastructure.gemini_inference import GeminiInferenceProvider
from worker.infrastructure.inference_factory import create_inference_provider
from worker.infrastructure.redis_bus import RedisBus, retry_with_backoff
from worker.infrastructure.stub_inference import StubInferenceProvider


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
        assert metadata.providers == {"inference_provider": "stub"}

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
        assert isinstance(kwargs.get("inference_provider"), StubInferenceProvider)


# --- Inference Provider Tests ---


@pytest.mark.anyio
async def test_stub_inference_provider_generate_default() -> None:
    provider = StubInferenceProvider()
    res = await provider.generate("Summarize daily activity")
    assert res == "Generated response for: Summarize daily activity"


@pytest.mark.anyio
async def test_stub_inference_provider_generate_custom_response() -> None:
    provider = StubInferenceProvider(custom_response="Deterministic output")
    res = await provider.generate("Any prompt")
    assert res == "Deterministic output"


@pytest.mark.anyio
async def test_stub_inference_provider_failure_trigger_raises() -> None:
    provider = StubInferenceProvider(failure_trigger="crash")
    with pytest.raises(RuntimeError, match="Simulated inference failure"):
        await provider.generate("Please crash now")


# --- TaskProcessor Inference Execution Tests ---


@pytest.mark.anyio
async def test_task_processor_uses_inference_provider_output() -> None:
    class CustomInferenceProvider:
        async def generate(self, prompt: str) -> str:
            return f"Custom inference: {prompt}"

    bus = FakeMessageBus()
    store = FakeResultStore()
    processor = TaskProcessor(
        "worker-1",
        bus,
        result_store=store,
        inference_provider=CustomInferenceProvider(),
    )
    task = TaskRequested.create("Optimize database queries")

    await processor.run_once(task)

    assert task.task_id in store.saved
    assert store.saved[task.task_id].output == f"Custom inference: {task.prompt}"
    assert store.saved[task.task_id].status == "completed"


@pytest.mark.anyio
async def test_task_processor_persists_generated_output_to_result_store() -> None:
    bus = FakeMessageBus()
    store = FakeResultStore()
    processor = TaskProcessor(
        "worker-1",
        bus,
        result_store=store,
        inference_provider=StubInferenceProvider(),
    )
    task = TaskRequested.create("Analyze queue throughput")

    await processor.run_once(task)

    assert task.task_id in store.saved
    saved = store.saved[task.task_id]
    assert saved.output == f"Generated response for: {task.prompt}"
    assert saved.status == "completed"


@pytest.mark.anyio
async def test_task_processor_inference_failure_lifecycle_and_no_persistence() -> None:
    class FailingInferenceProvider:
        async def generate(self, prompt: str) -> str:
            raise RuntimeError("Inference engine offline")

    bus = FakeMessageBus()
    store = FakeResultStore()
    processor = TaskProcessor(
        "worker-1",
        bus,
        result_store=store,
        inference_provider=FailingInferenceProvider(),
    )
    task = TaskRequested.create("Failing inference task")

    with pytest.raises(TaskExecutionError) as exc_info:
        await processor.run_once(task)

    assert "Inference engine offline" in str(exc_info.value.original_error)
    assert len(store.saved) == 0

    # Lifecycle: working -> error -> idle
    assert len(bus.published) == 3
    assert bus.published[0].state == "working"
    assert bus.published[0].task_id == task.task_id
    assert bus.published[1].state == "error"
    assert bus.published[1].task_id == task.task_id
    assert "Inference engine offline" in bus.published[1].detail
    assert bus.published[2].state == "idle"
    assert bus.published[2].detail == "Waiting for work"


# --- Factory Selection Tests ---


def test_create_inference_provider_default_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERENCE_PROVIDER", raising=False)
    provider = create_inference_provider()
    assert isinstance(provider, StubInferenceProvider)


def test_create_inference_provider_explicit_stub() -> None:
    provider = create_inference_provider("stub")
    assert isinstance(provider, StubInferenceProvider)


def test_create_inference_provider_env_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "stub")
    provider = create_inference_provider()
    assert isinstance(provider, StubInferenceProvider)


def test_create_inference_provider_case_insensitive() -> None:
    provider = create_inference_provider("  STuB  ")
    assert isinstance(provider, StubInferenceProvider)


def test_create_inference_provider_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported inference provider: 'unsupported'"):
        create_inference_provider("unsupported")


def test_create_inference_provider_cloudflare_missing_account_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "cloudflare")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.setenv("CLOUDFLARE_AI_API_TOKEN", "tok-1")

    with pytest.raises(
        ValueError, match="CLOUDFLARE_ACCOUNT_ID environment variable is required"
    ):
        create_inference_provider()


def test_create_inference_provider_cloudflare_missing_api_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "cloudflare")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc-1")
    monkeypatch.delenv("CLOUDFLARE_AI_API_TOKEN", raising=False)

    with pytest.raises(
        ValueError, match="CLOUDFLARE_AI_API_TOKEN environment variable is required"
    ):
        create_inference_provider()


def test_create_inference_provider_cloudflare_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "cloudflare")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc-env")
    monkeypatch.setenv("CLOUDFLARE_AI_API_TOKEN", "tok-env")
    monkeypatch.delenv("CLOUDFLARE_AI_MODEL", raising=False)

    provider = create_inference_provider()
    assert isinstance(provider, CloudflareInferenceProvider)
    assert provider._account_id == "acc-env"
    assert provider._api_token == "tok-env"
    assert provider.model == DEFAULT_CLOUDFLARE_AI_MODEL


def test_create_inference_provider_cloudflare_explicit_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_AI_API_TOKEN", raising=False)

    provider = create_inference_provider(
        name="cloudflare",
        account_id="acc-exp",
        api_token="tok-exp",
        model="@cf/meta/llama-3.3-70b-instruct",
        max_tokens=512,
    )
    assert isinstance(provider, CloudflareInferenceProvider)
    assert provider._account_id == "acc-exp"
    assert provider._api_token == "tok-exp"
    assert provider.model == "@cf/meta/llama-3.3-70b-instruct"
    assert provider.max_tokens == 512


# --- Cloudflare Adapter Tests ---


def test_cloudflare_provider_creation_default_and_custom_model() -> None:
    provider = CloudflareInferenceProvider(account_id="acc-1", api_token="tok-1")
    assert provider.model == DEFAULT_CLOUDFLARE_AI_MODEL

    custom = CloudflareInferenceProvider(
        account_id="acc-1", api_token="tok-1", model="@cf/custom/model"
    )
    assert custom.model == "@cf/custom/model"


def test_cloudflare_provider_secret_redaction_repr() -> None:
    secret = "secret-super-sensitive-token-123"
    provider = CloudflareInferenceProvider(account_id="acc-1", api_token=secret)
    rep = repr(provider)
    assert secret not in rep
    assert "[REDACTED]" in rep


@pytest.mark.anyio
async def test_cloudflare_provider_successful_generate() -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {"response": "This is the generated answer from Cloudflare."},
            "success": True,
            "errors": [],
            "messages": [],
        },
        request=httpx.Request(
            "POST",
            "https://api.cloudflare.com/client/v4/accounts/test-acc/ai/run/@cf/meta/llama-3.1-8b-instruct",
        ),
    )
    mock_client.post.return_value = mock_response

    provider = CloudflareInferenceProvider(
        account_id="test-acc",
        api_token="test-tok",
        client=mock_client,
    )

    prompt = "Explain quantum computing briefly"
    res = await provider.generate(prompt)

    assert res == "This is the generated answer from Cloudflare."
    mock_client.post.assert_called_once()
    url, kwargs = mock_client.post.call_args[0][0], mock_client.post.call_args[1]
    assert (
        "https://api.cloudflare.com/client/v4/accounts/test-acc/ai/run/@cf/meta/llama-3.1-8b-instruct"
        in url
    )
    assert kwargs["headers"]["Authorization"] == "Bearer test-tok"
    messages = kwargs["json"]["messages"]
    assert len(messages) == 2
    assert "You are a helpful and versatile AI assistant" in messages[0]["content"]
    assert messages[1]["content"] == prompt


@pytest.mark.anyio
async def test_cloudflare_provider_http_error_redacts_secret() -> None:
    mock_client = AsyncMock()
    secret_token = "secret-token-to-hide"
    mock_response = httpx.Response(
        status_code=500,
        json={"success": False, "errors": [{"message": f"Server error with {secret_token}"}]},
        request=httpx.Request(
            "POST",
            "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/model",
        ),
    )
    mock_client.post.return_value = mock_response

    provider = CloudflareInferenceProvider(
        account_id="acc",
        api_token=secret_token,
        client=mock_client,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await provider.generate("Any prompt")

    err_str = str(exc_info.value)
    assert secret_token not in err_str
    assert "[REDACTED]" in err_str


@pytest.mark.anyio
async def test_cloudflare_provider_malformed_json_raises() -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        content=b"not a valid json",
        request=httpx.Request(
            "POST",
            "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/model",
        ),
    )
    mock_client.post.return_value = mock_response

    provider = CloudflareInferenceProvider(
        account_id="acc",
        api_token="tok",
        client=mock_client,
    )

    with pytest.raises(RuntimeError, match="invalid JSON"):
        await provider.generate("prompt")


# --- Composition Root and Observability Tests ---


@pytest.mark.anyio
async def test_run_worker_registers_custom_inference_provider_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_ID", "worker-cf")
    monkeypatch.setenv("INFERENCE_PROVIDER", "cloudflare")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc-cf")
    monkeypatch.setenv("CLOUDFLARE_AI_API_TOKEN", "tok-cf")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_bus.register.assert_awaited_once()
        metadata = mock_bus.register.call_args[0][0]
        assert metadata.providers == {"inference_provider": "cloudflare"}


@pytest.mark.anyio
async def test_run_worker_unsupported_inference_provider_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "unknown-provider")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        from worker.main import run_worker

        with pytest.raises(ValueError, match="Unsupported inference provider"):
            await run_worker()

        mock_bus_cls.assert_not_called()


@pytest.mark.anyio
async def test_run_worker_multi_provider_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_ID", "worker-multi")
    monkeypatch.setenv("INFERENCE_PROVIDER", "stub,cloudflare")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc-multi")
    monkeypatch.setenv("CLOUDFLARE_AI_API_TOKEN", "tok-multi")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_bus.register.assert_awaited_once()
        metadata = mock_bus.register.call_args[0][0]
        assert metadata.providers == {"inference_provider": "stub,cloudflare"}


@pytest.mark.anyio
async def test_run_worker_missing_secondary_credentials_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "stub,cloudflare")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_AI_API_TOKEN", raising=False)

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        from worker.main import run_worker

        with pytest.raises(
            ValueError, match="CLOUDFLARE_ACCOUNT_ID environment variable is required"
        ):
            await run_worker()

        mock_bus_cls.assert_not_called()


@pytest.mark.anyio
async def test_run_worker_unsupported_secondary_provider_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "stub,invalid_vendor")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        from worker.main import run_worker

        with pytest.raises(
            ValueError, match="Unsupported inference provider: 'invalid_vendor'"
        ):
            await run_worker()

        mock_bus_cls.assert_not_called()


@pytest.mark.anyio
async def test_worker_redis_bus_serializes_fallback_transition_as_json() -> None:
    bus = RedisBus("redis://localhost:6379/0", queue_name="agent.tasks.general")
    bus.client.sadd = AsyncMock()  # type: ignore[method-assign]
    bus.client.hset = AsyncMock()  # type: ignore[method-assign]
    bus.client.xadd = AsyncMock()  # type: ignore[method-assign]

    transition = FallbackTransition(
        failed_provider="cloudflare",
        target_provider="gemini",
        reason="rate_limited",
        attempt_index=0,
    )
    event = AgentEvent.create(
        agent_id="worker-1",
        state="fallback",
        detail="Inference fallback: cloudflare -> gemini (rate_limited)",
        task_id="task-json-1",
        fallback_transition=transition,
    )

    await bus.publish(event)

    # Check xadd mapping
    bus.client.xadd.assert_awaited_once()
    _, call_mapping = bus.client.xadd.await_args[0]
    assert "fallback_transition" in call_mapping
    raw_ft = call_mapping["fallback_transition"]
    parsed = json.loads(raw_ft)
    assert parsed == transition.to_dict()
    assert parsed["failed_provider"] == "cloudflare"
    assert parsed["target_provider"] == "gemini"

    # Check hset mapping
    bus.client.hset.assert_awaited_once()
    kwargs = bus.client.hset.await_args[1]
    hset_mapping = kwargs.get("mapping", {})
    assert "fallback_transition" in hset_mapping
    assert json.loads(hset_mapping["fallback_transition"]) == transition.to_dict()


@pytest.mark.anyio
async def test_general_worker_with_inference_provider_retrievable_in_orchestrator() -> None:
    redis_storage: dict[str, str] = {}

    worker_bus = RedisBus("redis://localhost:6379/0", queue_name="agent.tasks.general")
    worker_bus.client.sadd = AsyncMock()  # type: ignore[method-assign]
    worker_bus.client.hset = AsyncMock()  # type: ignore[method-assign]
    worker_bus.client.xadd = AsyncMock()  # type: ignore[method-assign]

    async def mock_set(key: str, value: str) -> None:
        redis_storage[key] = value

    worker_bus.client.set = AsyncMock(side_effect=mock_set)  # type: ignore[method-assign]

    inference_provider = StubInferenceProvider()
    processor = TaskProcessor(
        "worker-1",
        worker_bus,
        result_store=worker_bus,
        inference_provider=inference_provider,
    )
    task = TaskRequested.create("Generate release summary")

    await processor.run_once(task)

    gateway = RedisGateway("redis://localhost:6379/0")

    async def mock_get(key: str) -> str | None:
        return redis_storage.get(key)

    gateway._client.get = AsyncMock(side_effect=mock_get)  # type: ignore[method-assign]

    retrieved = await gateway.get_result(task.task_id)
    assert retrieved is not None
    assert retrieved["task_id"] == task.task_id
    assert retrieved["status"] == "completed"
    assert retrieved["output"] == f"Generated response for: {task.prompt}"
    assert retrieved["acknowledged"] is True



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


# --- Gemini Adapter Tests ---


def test_create_inference_provider_gemini_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is required"):
        create_inference_provider()


def test_create_inference_provider_gemini_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-env")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    provider = create_inference_provider()
    assert isinstance(provider, GeminiInferenceProvider)
    assert provider.model == "gemini-2.5-flash"
    assert provider.provider_name == "gemini"


def test_create_inference_provider_gemini_explicit() -> None:
    provider = create_inference_provider(
        name="gemini",
        api_token="explicit-gemini-key",
        model="gemini-2.5-flash",
        max_tokens=256,
    )
    assert isinstance(provider, GeminiInferenceProvider)
    assert provider.model == "gemini-2.5-flash"
    assert provider.max_tokens == 256


def test_create_inference_provider_stub_and_cloudflare_unaffected_when_gemini_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    # Stub succeeds without Gemini env
    stub = create_inference_provider("stub")
    assert isinstance(stub, StubInferenceProvider)

    # Cloudflare succeeds without Gemini env
    cf = create_inference_provider(
        name="cloudflare",
        account_id="acc",
        api_token="tok",
    )
    assert isinstance(cf, CloudflareInferenceProvider)


@pytest.mark.anyio
async def test_gemini_worker_provider_successful_generate() -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Generated Gemini response."}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        },
    )
    mock_client.post.return_value = mock_response

    provider = GeminiInferenceProvider(
        api_key="test-gemini-key",
        client=mock_client,
    )

    result = await provider.generate("Explain machine learning")
    assert result == "Generated Gemini response."
    mock_client.post.assert_called_once()
    url, kwargs = mock_client.post.call_args[0][0], mock_client.post.call_args[1]
    assert "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" in url
    assert kwargs["headers"]["x-goog-api-key"] == "test-gemini-key"
    assert kwargs["json"]["contents"][0]["parts"][0]["text"] == "Explain machine learning"


@pytest.mark.anyio
async def test_gemini_worker_provider_error_redacts_secret() -> None:
    mock_client = AsyncMock()
    secret_key = "super-secret-gemini-api-key"
    mock_response = httpx.Response(
        status_code=400,
        json={"error": {"message": f"Bad request using {secret_key}"}},
    )
    mock_client.post.return_value = mock_response

    provider = GeminiInferenceProvider(
        api_key=secret_key,
        client=mock_client,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await provider.generate("test prompt")

    err_text = str(exc_info.value)
    assert secret_key not in err_text
    assert "[REDACTED]" in err_text


def test_gemini_worker_provider_configuration_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Invalid model
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        GeminiInferenceProvider(api_key="test-key", model="")

    with pytest.raises(ValueError, match="model must be a non-empty string"):
        GeminiInferenceProvider(api_key="test-key", model="   ")

    with pytest.raises(ValueError, match="model must be a non-empty string after normalization"):
        GeminiInferenceProvider(api_key="test-key", model="models/")

    # Invalid max_tokens
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        GeminiInferenceProvider(api_key="test-key", max_tokens="256")  # type: ignore

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        GeminiInferenceProvider(api_key="test-key", max_tokens=True)  # type: ignore

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        GeminiInferenceProvider(api_key="test-key", max_tokens=False)  # type: ignore

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        GeminiInferenceProvider(api_key="test-key", max_tokens=0)

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        GeminiInferenceProvider(api_key="test-key", max_tokens=-5)

    # Invalid temperature
    with pytest.raises(ValueError, match="temperature must be a number"):
        GeminiInferenceProvider(api_key="test-key", temperature=True)  # type: ignore

    with pytest.raises(ValueError, match="temperature must be a number"):
        GeminiInferenceProvider(api_key="test-key", temperature="0.7")  # type: ignore

    with pytest.raises(ValueError, match="temperature must be a number between 0.0 and 2.0"):
        GeminiInferenceProvider(api_key="test-key", temperature=-0.1)

    with pytest.raises(ValueError, match="temperature must be a number between 0.0 and 2.0"):
        GeminiInferenceProvider(api_key="test-key", temperature=2.1)

    # Valid temperature
    valid_provider = GeminiInferenceProvider(api_key="test-key", temperature=0.7)
    assert valid_provider.temperature == 0.7

    # Invalid GEMINI_MAX_TOKENS from env
    monkeypatch.setenv("GEMINI_MAX_TOKENS", "invalid_number")
    with pytest.raises(ValueError, match="GEMINI_MAX_TOKENS must be a positive integer"):
        GeminiInferenceProvider(api_key="test-key")

    monkeypatch.setenv("GEMINI_MAX_TOKENS", "0")
    with pytest.raises(ValueError, match="GEMINI_MAX_TOKENS must be a positive integer"):
        GeminiInferenceProvider(api_key="test-key")
