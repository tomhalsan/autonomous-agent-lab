"""Automated regression coverage for inference provider interchangeability (Issue #77).

Validates that:
1. Switching inference providers requires configuration changes only, with no Worker
   or Research Agent application/domain code changes.
2. Provider-specific response formats and implementation details remain strictly behind
   the existing inference adapter boundary.
3. Application-level behavior and results continue to satisfy the same provider-neutral contract.
4. Provider substitution is demonstrated by parameterized automated regression coverage.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from agent_contracts import (
    AgentEvent,
    InferenceResponse,
    TaskRequested,
    TaskResult,
)
from research_agent.application.processor import ResearchProcessor
from research_agent.domain.models import ResearchResult, SynthesisResult
from research_agent.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as ResearchCloudflareProvider,
)
from research_agent.infrastructure.gemini_inference import (
    GeminiInferenceProvider as ResearchGeminiProvider,
)
from research_agent.infrastructure.inference_factory import (
    create_inference_provider as create_research_inference_provider,
)
from research_agent.infrastructure.stub_executor import StubResearchExecutor
from research_agent.infrastructure.stub_inference import (
    StubInferenceProvider as ResearchStubProvider,
)
from worker.application.processor import TaskProcessor
from worker.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as WorkerCloudflareProvider,
)
from worker.infrastructure.gemini_inference import (
    GeminiInferenceProvider as WorkerGeminiProvider,
)
from worker.infrastructure.inference_factory import (
    create_inference_provider as create_worker_inference_provider,
)
from worker.infrastructure.stub_inference import (
    StubInferenceProvider as WorkerStubProvider,
)

# ==============================================================================
# Test Fakes for Worker and Research Agent
# ==============================================================================


class FakeMessageBus:
    def __init__(self) -> None:
        self.published: list[AgentEvent] = []

    async def publish(self, event: AgentEvent) -> None:
        self.published.append(event)

    async def next_task(self) -> TaskRequested:
        raise NotImplementedError


class FakeWorkerResultStore:
    def __init__(self) -> None:
        self.saved: dict[str, TaskResult] = {}

    async def save(self, result: TaskResult) -> None:
        self.saved[result.task_id] = result


class FakeResearchResultStore:
    def __init__(self) -> None:
        self.saved: dict[str, ResearchResult] = {}

    async def save(self, result: ResearchResult) -> None:
        self.saved[result.task_id] = result


# ==============================================================================
# Helper Mock Factories
# ==============================================================================


def _make_mock_cloudflare_client(
    response_text: str = "Cloudflare generated content.",
) -> AsyncMock:
    client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {"response": response_text},
            "success": True,
            "errors": [],
            "messages": [],
        },
        request=httpx.Request("POST", "https://api.cloudflare.com/client/v4/accounts/test-acc/ai/run/model"),
    )
    client.post.return_value = mock_response
    return client


def _make_mock_gemini_client(
    response_text: str = "Gemini generated content.",
) -> AsyncMock:
    client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": response_text}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 8},
        },
        request=httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        ),
    )
    client.post.return_value = mock_response
    return client


def _create_worker_provider(provider_type: str) -> Any:
    if provider_type == "stub":
        return WorkerStubProvider(
            provider_name="stub",
            custom_response=None,
        )
    if provider_type == "cloudflare":
        mock_client = _make_mock_cloudflare_client(
            response_text="Standardized worker output from Cloudflare.",
        )
        return WorkerCloudflareProvider(
            account_id="test-acc",
            api_token="test-token",
            client=mock_client,
        )
    if provider_type == "gemini":
        mock_client = _make_mock_gemini_client(
            response_text="Standardized worker output from Gemini.",
        )
        return WorkerGeminiProvider(
            api_key="test-gemini-key",
            client=mock_client,
        )
    raise ValueError(f"Unknown provider type: {provider_type}")


def _create_research_provider(provider_type: str) -> Any:
    if provider_type == "stub":
        return ResearchStubProvider(
            provider_name="stub",
        )
    if provider_type == "cloudflare":
        mock_client = _make_mock_cloudflare_client(
            response_text="Standardized synthesis from Cloudflare.\n\nSources:\n- https://example.com/source",
        )
        return ResearchCloudflareProvider(
            account_id="test-acc",
            api_token="test-token",
            client=mock_client,
        )
    if provider_type == "gemini":
        mock_client = _make_mock_gemini_client(
            response_text="Standardized synthesis from Gemini.\n\nSources:\n- https://example.com/source",
        )
        return ResearchGeminiProvider(
            api_key="test-gemini-key",
            client=mock_client,
        )
    raise ValueError(f"Unknown provider type: {provider_type}")


# ==============================================================================
# Suite 1: Worker Provider Interchangeability
# ==============================================================================


@pytest.mark.anyio
@pytest.mark.parametrize("provider_type", ["stub", "cloudflare", "gemini"])
async def test_worker_inference_provider_contract_compliance(provider_type: str) -> None:
    """Verifies all worker inference providers satisfy the same generate() contract."""
    provider = _create_worker_provider(provider_type)

    prompt = "Summarize queue metrics across nodes"
    output = await provider.generate(prompt)

    # 1. Output must be an InferenceResponse satisfying the worker port contract
    assert isinstance(output, InferenceResponse)
    assert isinstance(output.text, str)
    assert len(output.text.strip()) > 0
    assert output.provider_name == provider_type

    # 2. Provider implementation details must not leak into output
    assert "status_code" not in output.text
    assert "candidates" not in output.text
    assert "test-token" not in output.text
    assert "test-gemini-key" not in output.text


@pytest.mark.anyio
@pytest.mark.parametrize("provider_type", ["stub", "cloudflare", "gemini"])
async def test_worker_task_processor_substitution(provider_type: str) -> None:
    """Verifies TaskProcessor executes identically when substituting inference providers."""
    provider = _create_worker_provider(provider_type)
    bus = FakeMessageBus()
    store = FakeWorkerResultStore()

    processor = TaskProcessor(
        agent_id="worker-test",
        bus=bus,
        result_store=store,
        inference_provider=provider,
    )
    task = TaskRequested.create("Optimize database queries")

    await processor.run_once(task)

    # 1. ResultStore receives valid TaskResult satisfying provider-neutral contract
    assert task.task_id in store.saved
    saved = store.saved[task.task_id]
    assert isinstance(saved, TaskResult)
    assert saved.task_id == task.task_id
    assert saved.status == "completed"
    assert isinstance(saved.output, str) and len(saved.output.strip()) > 0

    # 2. Lifecycle event sequence is identical across providers (working -> idle)
    assert len(bus.published) == 2
    assert bus.published[0].state == "working"
    assert bus.published[0].task_id == task.task_id
    assert bus.published[0].detail == "Processing task"
    assert task.prompt not in bus.published[0].detail

    assert bus.published[1].state == "idle"
    assert bus.published[1].task_id == task.task_id
    assert bus.published[1].detail == "Task completed"
    assert task.prompt not in bus.published[1].detail


@pytest.mark.anyio
@pytest.mark.parametrize("provider_name", ["stub", "cloudflare", "gemini"])
async def test_worker_configuration_switching_without_code_changes(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
) -> None:
    """Verifies switching worker providers requires environment configuration only."""
    monkeypatch.setenv("INFERENCE_PROVIDER", provider_name)

    mock_client: AsyncMock | None = None
    if provider_name == "cloudflare":
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc-env-1")
        monkeypatch.setenv("CLOUDFLARE_AI_API_TOKEN", "tok-env-1")
        mock_client = _make_mock_cloudflare_client("Output from env-configured Cloudflare")
    elif provider_name == "gemini":
        monkeypatch.setenv("GEMINI_API_KEY", "key-env-1")
        mock_client = _make_mock_gemini_client("Output from env-configured Gemini")

    # Instantiate provider purely via factory and env config (passing mock_client for offline safety)
    provider = create_worker_inference_provider(client=mock_client)
    assert provider.provider_name == provider_name

    # Plug into unchanged application processor
    bus = FakeMessageBus()
    store = FakeWorkerResultStore()
    processor = TaskProcessor("worker-env", bus, result_store=store, inference_provider=provider)
    task = TaskRequested.create("Run health audit")

    await processor.run_once(task)

    assert task.task_id in store.saved
    assert store.saved[task.task_id].status == "completed"
    assert len(store.saved[task.task_id].output) > 0


# ==============================================================================
# Suite 2: Research Agent Provider Interchangeability
# ==============================================================================


@pytest.mark.anyio
@pytest.mark.parametrize("provider_type", ["stub", "cloudflare", "gemini"])
async def test_research_agent_inference_provider_contract_compliance(provider_type: str) -> None:
    """Verifies all research agent inference providers satisfy the same synthesize() contract."""
    provider = _create_research_provider(provider_type)

    question = "What is Byzantine fault tolerance?"
    evidence = "BFT systems tolerate arbitrary node failures up to (n-1)/3."

    result = await provider.synthesize(question, evidence)

    # 1. Output must be a valid SynthesisResult
    assert isinstance(result, SynthesisResult)
    assert result.question == question
    assert result.evidence == evidence
    assert isinstance(result.content, str)
    assert len(result.content.strip()) > 0
    assert isinstance(result.truncated, bool)
    assert result.provider == provider_type

    # 2. Provider implementation details must not leak
    assert "status_code" not in result.content
    assert "test-token" not in result.content
    assert "test-gemini-key" not in result.content


@pytest.mark.anyio
@pytest.mark.parametrize("provider_type", ["stub", "cloudflare", "gemini"])
async def test_research_processor_substitution(provider_type: str) -> None:
    """Verifies ResearchProcessor executes identically when substituting inference providers."""
    provider = _create_research_provider(provider_type)
    bus = FakeMessageBus()
    store = FakeResearchResultStore()
    executor = StubResearchExecutor()

    processor = ResearchProcessor(
        agent_id="research-test",
        bus=bus,
        executor=executor,
        result_store=store,
        inference_provider=provider,
    )
    task = TaskRequested.create("Investigate consensus algorithms")

    await processor.run_once(task)

    # 1. ResultStore receives valid ResearchResult satisfying provider-neutral contract
    assert task.task_id in store.saved
    saved = store.saved[task.task_id]
    assert isinstance(saved, ResearchResult)
    assert saved.task_id == task.task_id
    assert saved.acknowledged is True
    assert isinstance(saved.truncated, bool)
    assert isinstance(saved.output, str) and len(saved.output.strip()) > 0
    # Sources from executor are preserved across all providers
    assert isinstance(saved.sources, (tuple, list))

    # 2. Lifecycle event sequence is identical across providers (working -> idle)
    assert len(bus.published) == 2
    assert bus.published[0].state == "working"
    assert bus.published[0].task_id == task.task_id
    assert bus.published[0].detail == "Processing task"
    assert task.prompt not in bus.published[0].detail

    assert bus.published[1].state == "idle"
    assert bus.published[1].task_id == task.task_id
    assert bus.published[1].detail == "Task completed"
    assert saved.output not in bus.published[1].detail


@pytest.mark.anyio
@pytest.mark.parametrize("provider_name", ["stub", "cloudflare", "gemini"])
async def test_research_agent_configuration_switching_without_code_changes(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
) -> None:
    """Verifies switching research agent providers requires environment configuration only."""
    monkeypatch.setenv("INFERENCE_PROVIDER", provider_name)

    mock_client: AsyncMock | None = None
    if provider_name == "cloudflare":
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc-env-2")
        monkeypatch.setenv("CLOUDFLARE_AI_API_TOKEN", "tok-env-2")
        mock_client = _make_mock_cloudflare_client("Synthesis from env-configured Cloudflare")
    elif provider_name == "gemini":
        monkeypatch.setenv("GEMINI_API_KEY", "key-env-2")
        mock_client = _make_mock_gemini_client("Synthesis from env-configured Gemini")

    # Instantiate provider purely via factory and env config (passing mock_client for offline safety)
    provider = create_research_inference_provider(client=mock_client)
    assert provider.provider_name == provider_name

    # Plug into unchanged application processor
    bus = FakeMessageBus()
    store = FakeResearchResultStore()
    executor = StubResearchExecutor()
    processor = ResearchProcessor(
        "research-env",
        bus,
        executor,
        result_store=store,
        inference_provider=provider,
    )
    task = TaskRequested.create("Synthesize distributed tracing survey")

    await processor.run_once(task)

    assert task.task_id in store.saved
    assert store.saved[task.task_id].acknowledged is True
    assert len(store.saved[task.task_id].output) > 0
