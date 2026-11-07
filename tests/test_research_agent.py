import asyncio
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from agent_contracts import AgentCapabilityMetadata, AgentEvent, TaskRequested, TaskResult
from orchestrator.application.queries import GetAgentStates, GetRecentEvents
from research_agent.application.processor import ResearchExecutionError, ResearchProcessor
from research_agent.domain.evaluator import EvidenceEvaluator
from research_agent.domain.evidence import (
    AuthorityTier,
    CandidateEvidence,
    EvaluatedEvidence,
    FilterReason,
    SourceType,
)
from research_agent.domain.models import ResearchResult, SynthesisResult
from research_agent.infrastructure.cloudflare_inference import (
    DEFAULT_CLOUDFLARE_AI_MODEL,
    CloudflareInferenceProvider,
)
from research_agent.infrastructure.executor_factory import create_research_executor
from research_agent.infrastructure.inference_factory import create_inference_provider
from research_agent.infrastructure.redis_bus import RedisBus
from research_agent.infrastructure.stub_executor import StubResearchExecutor
from research_agent.infrastructure.stub_inference import StubInferenceProvider, StubSynthesizer
from research_agent.infrastructure.tavily_executor import TavilyResearchExecutor


class FakeMessageBus:
    def __init__(self) -> None:
        self.published: list[AgentEvent] = []
        self.registered_metadata: AgentCapabilityMetadata | None = None

    async def register(self, metadata: AgentCapabilityMetadata) -> None:
        self.registered_metadata = metadata

    async def publish(self, event: AgentEvent) -> None:
        self.published.append(event)

    async def next_task(self) -> TaskRequested:
        raise NotImplementedError


class FakeResultStore:
    def __init__(self) -> None:
        self.saved: dict[str, ResearchResult] = {}

    async def save(self, result: ResearchResult) -> None:
        self.saved[result.task_id] = result


class FakeCompatibilityStateReader:
    """Simulates RedisGateway reading state hashes and event streams from Redis."""

    def __init__(self, states: dict[str, dict[str, str]], events: list[dict[str, str]]) -> None:
        self._states = states
        self._events = events

    async def agents(self) -> list[dict[str, str]]:
        return [self._states[agent_id] for agent_id in sorted(self._states.keys())]

    async def events(self, limit: int = 50) -> list[dict[str, str]]:
        return self._events[:limit]


# --- Domain Tests ---


def test_research_result_valid() -> None:
    result = ResearchResult(task_id="task-123", output="Completed deterministic analysis")
    assert result.task_id == "task-123"
    assert result.output == "Completed deterministic analysis"
    assert result.acknowledged is True


def test_research_result_rejects_empty_task_id() -> None:
    with pytest.raises(ValueError, match="task_id must not be empty"):
        ResearchResult(task_id="   ", output="any")


def test_synthesis_result_valid() -> None:
    result = SynthesisResult(
        question="What is consensus?",
        content="Consensus ensures agreement across distributed nodes.",
        evidence="Raft and Paxos are consensus algorithms.",
        provider="stub",
        model="stub",
    )
    assert result.question == "What is consensus?"
    assert result.content == "Consensus ensures agreement across distributed nodes."
    assert result.evidence == "Raft and Paxos are consensus algorithms."
    assert result.provider == "stub"
    assert result.model == "stub"


def test_synthesis_result_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        SynthesisResult(
            question="   ",
            content="synthesized content",
            evidence="evidence",
        )


# --- Infrastructure Adapter Tests: StubResearchExecutor ---


@pytest.mark.anyio
async def test_stub_research_executor_deterministic_output() -> None:
    executor = StubResearchExecutor()
    task = TaskRequested.create("Investigate distributed state synchronization")

    result1 = await executor.execute(task)
    result2 = await executor.execute(task)

    assert result1.task_id == task.task_id
    assert result2.task_id == task.task_id
    assert result1.output == result2.output
    assert "Deterministic execution completed for task:" in result1.output
    # Must not contain fake fabricated findings
    assert "findings" not in result1.output.lower()


@pytest.mark.anyio
async def test_stub_research_executor_deterministic_failure() -> None:
    executor = StubResearchExecutor(failure_trigger="trigger_error")
    failing_task = TaskRequested.create("Task containing trigger_error in prompt")

    with pytest.raises(RuntimeError, match="Simulated execution failure"):
        await executor.execute(failing_task)


# --- Application Processor Lifecycle Tests ---


@pytest.mark.anyio
async def test_research_processor_success_transitions() -> None:
    bus = FakeMessageBus()
    executor = StubResearchExecutor()
    processor = ResearchProcessor("research-agent-1", bus, executor)
    task = TaskRequested.create("Research consensus algorithms")

    await processor.run_once(task)

    assert len(bus.published) == 2
    working_event, idle_event = bus.published

    # Transition 1: working
    assert working_event.state == "working"
    assert working_event.agent_id == "research-agent-1"
    assert working_event.task_id == task.task_id
    assert "Processing: Research consensus algorithms" in working_event.detail

    # Transition 2: idle
    assert idle_event.state == "idle"
    assert idle_event.agent_id == "research-agent-1"
    assert idle_event.task_id == task.task_id
    assert "Completed: Deterministic execution completed" in idle_event.detail


@pytest.mark.anyio
async def test_research_processor_failure_lifecycle_preserves_error_state() -> None:
    bus = FakeMessageBus()
    executor = StubResearchExecutor(failure_trigger="crash_task")
    processor = ResearchProcessor("research-agent-1", bus, executor)
    task = TaskRequested.create("Execute task with crash_task trigger")

    with pytest.raises(ResearchExecutionError) as exc_info:
        await processor.run_once(task)

    assert exc_info.value.task_id == task.task_id
    assert "Simulated execution failure" in str(exc_info.value.original_error)

    # Required lifecycle: working -> error (MUST NOT immediately overwrite error with idle)
    assert len(bus.published) == 2
    working_event, error_event = bus.published

    assert working_event.state == "working"
    assert working_event.task_id == task.task_id

    assert error_event.state == "error"
    assert error_event.task_id == task.task_id
    assert "Simulated execution failure" in error_event.detail


@pytest.mark.anyio
async def test_research_processor_recovers_to_working_on_subsequent_task() -> None:
    bus = FakeMessageBus()
    executor = StubResearchExecutor(failure_trigger="fail_first")
    processor = ResearchProcessor("research-agent-1", bus, executor)

    task1 = TaskRequested.create("fail_first task")
    with pytest.raises(ResearchExecutionError):
        await processor.run_once(task1)

    assert bus.published[-1].state == "error"

    # Subsequent task transitions from error to working then idle
    task2 = TaskRequested.create("good task")
    await processor.run_once(task2)

    assert len(bus.published) == 4
    # Event sequence: task1 working -> task1 error -> task2 working -> task2 idle
    states = [e.state for e in bus.published]
    assert states == ["working", "error", "working", "idle"]


# --- Orchestrator State Reader & Dashboard Compatibility Tests ---


@pytest.mark.anyio
async def test_orchestrator_state_reader_and_dashboard_compatibility() -> None:
    """Verify that research agent events conform to what orchestrator queries

    and dashboard HTML expect (agent_id, state, detail, task_id).
    """
    bus = FakeMessageBus()
    executor = StubResearchExecutor(failure_trigger="fail_here")
    processor = ResearchProcessor("research-agent-1", bus, executor)

    # Initial registration
    initial_event = AgentEvent.create("research-agent-1", "idle", "Waiting for work")
    await bus.publish(initial_event)

    # Execute a task
    task = TaskRequested.create("Analyze data portability")
    await processor.run_once(task)

    # Execute a failing task
    failing_task = TaskRequested.create("Trigger fail_here")
    with pytest.raises(ResearchExecutionError):
        await processor.run_once(failing_task)

    # Transform published events into Redis state representations exactly like RedisBus does
    stored_states: dict[str, dict[str, str]] = {}
    stored_events: list[dict[str, str]] = []

    for idx, event in enumerate(bus.published):
        mapping = {k: "" if v is None else str(v) for k, v in event.to_dict().items()}
        stored_states[event.agent_id] = mapping
        stored_events.append(dict(mapping, event_id=f"1700000000000-{idx}"))

    reader = FakeCompatibilityStateReader(stored_states, stored_events)
    agent_states_query = GetAgentStates(reader)
    recent_events_query = GetRecentEvents(reader)

    agents = await agent_states_query.execute()
    events = await recent_events_query.execute()

    assert len(agents) == 1
    research_agent_state = agents[0]

    # Verify keys required by presentation/index.html:
    # <b>${esc(x.agent_id)}</b> <span class="ok">${esc(x.state)}</span><br>${esc(x.detail)}
    assert research_agent_state["agent_id"] == "research-agent-1"
    assert research_agent_state["state"] == "error"  # Preserved in error after failed task
    assert "Simulated execution failure" in research_agent_state["detail"]
    assert "task_id" in research_agent_state
    assert research_agent_state["task_id"] == failing_task.task_id

    # Verify events list compatibility:
    # <b>${esc(x.state)}</b> ${esc(x.agent_id)}<br>${esc(x.detail)}
    assert len(events) == 5
    for event in events:
        assert "event_id" in event
        assert "agent_id" in event
        assert "state" in event
        assert "detail" in event
        assert event["state"] in ("idle", "working", "error")


# --- RedisBus & Main Wiring Tests ---


def test_research_agent_redis_bus_default_queue() -> None:
    bus = RedisBus("redis://localhost:6379/0")
    assert bus.queue_name == "agent.tasks.research"


def test_research_agent_redis_bus_custom_queue() -> None:
    bus = RedisBus("redis://localhost:6379/0", queue_name="custom.research.queue")
    assert bus.queue_name == "custom.research.queue"


@pytest.mark.anyio
async def test_run_research_agent_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_ID", raising=False)
    monkeypatch.delenv("TASK_QUEUE", raising=False)

    with (
        patch("research_agent.main.RedisBus") as mock_bus_cls,
        patch("research_agent.main.ResearchProcessor"),
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_bus_cls.assert_called_once_with(
            "redis://localhost:6379/0",
            queue_name="agent.tasks.research",
        )


@pytest.mark.anyio
async def test_run_research_agent_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ID", "custom-research-agent")
    monkeypatch.setenv("TASK_QUEUE", "custom.tasks.research")

    with (
        patch("research_agent.main.RedisBus") as mock_bus_cls,
        patch("research_agent.main.ResearchProcessor"),
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_bus_cls.assert_called_once_with(
            "redis://localhost:6379/0",
            queue_name="custom.tasks.research",
        )


# --- ResearchExecutor Factory & Selection Tests ---


def test_create_research_executor_default_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_EXECUTOR", raising=False)
    executor = create_research_executor()
    assert isinstance(executor, StubResearchExecutor)


def test_create_research_executor_explicit_stub() -> None:
    executor = create_research_executor("stub")
    assert isinstance(executor, StubResearchExecutor)

    # Verify case and whitespace insensitivity
    executor_caps = create_research_executor("  STUB  ")
    assert isinstance(executor_caps, StubResearchExecutor)


def test_create_research_executor_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EXECUTOR", "stub")
    executor = create_research_executor()
    assert isinstance(executor, StubResearchExecutor)


def test_create_research_executor_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported research executor: 'llm'"):
        create_research_executor("llm")

    with pytest.raises(ValueError, match="Unsupported research executor: 'unknown'"):
        create_research_executor("unknown")

    with pytest.raises(ValueError, match="Unsupported research executor: ''"):
        create_research_executor("")


def test_create_research_executor_unsupported_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_EXECUTOR", "unsupported_provider")
    with pytest.raises(ValueError, match="Unsupported research executor: 'unsupported_provider'"):
        create_research_executor()


@pytest.mark.anyio
async def test_run_research_agent_creates_stub_executor_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCH_EXECUTOR", raising=False)

    with (
        patch("research_agent.main.RedisBus") as mock_bus_cls,
        patch("research_agent.main.ResearchProcessor") as mock_processor_cls,
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_processor_cls.assert_called_once()
        _, _, executor_arg = mock_processor_cls.call_args[0]
        assert isinstance(executor_arg, StubResearchExecutor)


@pytest.mark.anyio
async def test_run_research_agent_explicit_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_EXECUTOR", "stub")

    with (
        patch("research_agent.main.RedisBus") as mock_bus_cls,
        patch("research_agent.main.ResearchProcessor") as mock_processor_cls,
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_processor_cls.assert_called_once()
        _, _, executor_arg = mock_processor_cls.call_args[0]
        assert isinstance(executor_arg, StubResearchExecutor)


@pytest.mark.anyio
async def test_run_research_agent_unsupported_executor_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_EXECUTOR", "unsupported_model")

    with patch("research_agent.main.RedisBus") as mock_bus_cls:
        from research_agent.main import run_research_agent

        with pytest.raises(ValueError, match="Unsupported research executor: 'unsupported_model'"):
            await run_research_agent()

        # Fails before attempting any Redis connection
        mock_bus_cls.assert_not_called()


# --- TavilyResearchExecutor Tests ---


def test_create_research_executor_tavily_with_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_EXECUTOR", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-12345")
    executor = create_research_executor()
    assert isinstance(executor, TavilyResearchExecutor)


def test_create_research_executor_tavily_explicit_name_and_key() -> None:
    executor = create_research_executor("tavily", api_key="tvly-test-12345")
    assert isinstance(executor, TavilyResearchExecutor)

    # Verify case and whitespace insensitivity
    executor_caps = create_research_executor("  TAVILY  ", api_key="tvly-test-12345")
    assert isinstance(executor_caps, TavilyResearchExecutor)


def test_create_research_executor_tavily_missing_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(
        ValueError,
        match="TAVILY_API_KEY environment variable is required when RESEARCH_EXECUTOR is 'tavily'",
    ):
        create_research_executor("tavily")


def test_create_research_executor_tavily_empty_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "   ")
    with pytest.raises(
        ValueError,
        match="TAVILY_API_KEY environment variable is required when RESEARCH_EXECUTOR is 'tavily'",
    ):
        create_research_executor("tavily")


@pytest.mark.anyio
async def test_run_research_agent_tavily_missing_key_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_EXECUTOR", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with patch("research_agent.main.RedisBus") as mock_bus_cls:
        from research_agent.main import run_research_agent

        with pytest.raises(
            ValueError,
            match="TAVILY_API_KEY environment variable is required when RESEARCH_EXECUTOR is 'tavily'",
        ):
            await run_research_agent()

        mock_bus_cls.assert_not_called()


@pytest.mark.anyio
async def test_run_research_agent_tavily_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_EXECUTOR", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-12345")

    with (
        patch("research_agent.main.RedisBus") as mock_bus_cls,
        patch("research_agent.main.ResearchProcessor") as mock_processor_cls,
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_processor_cls.assert_called_once()
        _, _, executor_arg = mock_processor_cls.call_args[0]
        assert isinstance(executor_arg, TavilyResearchExecutor)


@pytest.mark.anyio
async def test_tavily_executor_successful_search() -> None:
    mock_client = AsyncMock()
    mock_client.search.return_value = {
        "query": "Compare local and cloud LLMs",
        "answer": "Local LLMs prioritize privacy while cloud LLMs offer superior compute scale.",
        "results": [
            {
                "title": "Local vs Cloud Comparison",
                "url": "https://example.com/compare",
                "content": "Benchmark shows 10x latency variance.",
            },
            {
                "title": "Cloud AI Scaling",
                "url": "https://example.com/cloud",
                "content": "Scale-out architectures simplify operations.",
            },
        ],
    }

    executor = TavilyResearchExecutor(api_key="tvly-mock", client=mock_client)
    task = TaskRequested.create("Compare local and cloud LLMs")

    result = await executor.execute(task)

    assert result.task_id == task.task_id
    assert "Research query: Compare local and cloud LLMs" in result.output
    assert "Summary: Local LLMs prioritize privacy while cloud LLMs offer superior compute scale." in result.output
    assert "[1] Local vs Cloud Comparison (https://example.com/compare)" in result.output
    assert "Benchmark shows 10x latency variance." in result.output
    assert "[2] Cloud AI Scaling (https://example.com/cloud)" in result.output
    assert "Scale-out architectures simplify operations." in result.output
    mock_client.search.assert_awaited_once_with(query="Compare local and cloud LLMs")


@pytest.mark.anyio
async def test_tavily_executor_empty_results_no_fabrication() -> None:
    mock_client = AsyncMock()
    mock_client.search.return_value = {
        "query": "obscure nonexistent topic",
        "results": [],
    }

    executor = TavilyResearchExecutor(api_key="tvly-mock", client=mock_client)
    task = TaskRequested.create("obscure nonexistent topic")

    result = await executor.execute(task)

    assert result.task_id == task.task_id
    assert result.output == "No research results found for query: obscure nonexistent topic"
    mock_client.search.assert_awaited_once_with(query="obscure nonexistent topic")


@pytest.mark.anyio
async def test_tavily_executor_api_failure_handling() -> None:
    mock_client = AsyncMock()
    mock_client.search.side_effect = RuntimeError("Rate limit exceeded")

    executor = TavilyResearchExecutor(api_key="tvly-mock", client=mock_client)
    task = TaskRequested.create("rate limited query")

    with pytest.raises(RuntimeError, match="Tavily search failed for task"):
        await executor.execute(task)


@pytest.mark.anyio
async def test_tavily_executor_redacts_api_key_on_failure() -> None:
    secret_key = "tvly-super-secret-key-99999"
    mock_client = AsyncMock()
    mock_client.search.side_effect = RuntimeError(f"HTTP 401 Unauthorized for {secret_key}")

    executor = TavilyResearchExecutor(api_key=secret_key, client=mock_client)
    task = TaskRequested.create("auth fail query")

    with pytest.raises(RuntimeError) as exc_info:
        await executor.execute(task)

    assert secret_key not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


@pytest.mark.anyio
async def test_research_processor_with_tavily_executor_integration() -> None:
    mock_client = AsyncMock()
    mock_client.search.return_value = {
        "query": "Survey consensus models",
        "results": [
            {
                "title": "Raft vs Paxos",
                "url": "https://example.com/raft",
                "content": "Leader election mechanisms",
            }
        ],
    }

    executor = TavilyResearchExecutor(api_key="tvly-mock", client=mock_client)
    bus = FakeMessageBus()
    processor = ResearchProcessor("research-agent-1", bus, executor)
    task = TaskRequested.create("Survey consensus models")

    await processor.run_once(task)

    assert len(bus.published) == 2
    assert bus.published[0].state == "working"
    assert bus.published[1].state == "idle"
    assert "Completed:" in bus.published[1].detail
    assert "Raft vs Paxos" in bus.published[1].detail


@pytest.mark.anyio
async def test_research_processor_stores_result_on_success() -> None:
    bus = FakeMessageBus()
    store = FakeResultStore()
    executor = StubResearchExecutor()
    processor = ResearchProcessor("research-agent-1", bus, executor, result_store=store)
    task = TaskRequested.create("Investigate consensus algorithms")

    await processor.run_once(task)

    assert task.task_id in store.saved
    saved = store.saved[task.task_id]
    assert saved.task_id == task.task_id
    assert f"Deterministic execution completed for task: {task.task_id}" in saved.output
    assert saved.acknowledged is True
    # Operational events intact
    assert len(bus.published) == 2
    assert bus.published[0].state == "working"
    assert bus.published[1].state == "idle"


@pytest.mark.anyio
async def test_research_processor_error_does_not_store_result() -> None:
    bus = FakeMessageBus()
    store = FakeResultStore()
    executor = StubResearchExecutor(failure_trigger="trigger_fail")
    processor = ResearchProcessor("research-agent-1", bus, executor, result_store=store)
    task = TaskRequested.create("Task containing trigger_fail in prompt")

    with pytest.raises(ResearchExecutionError):
        await processor.run_once(task)

    assert len(store.saved) == 0
    assert len(bus.published) == 2
    assert bus.published[0].state == "working"
    assert bus.published[1].state == "error"


@pytest.mark.anyio
async def test_research_processor_with_tavily_stores_grounded_result() -> None:
    mock_client = AsyncMock()
    mock_client.search.return_value = {
        "query": "Compare local and cloud LLMs",
        "answer": "Local LLMs offer privacy; cloud LLMs scale seamlessly.",
        "results": [
            {
                "title": "Local vs Cloud",
                "url": "https://example.com/llm-analysis",
                "content": "Benchmark shows 10x compute throughput on cloud.",
            }
        ],
    }
    executor = TavilyResearchExecutor(api_key="tvly-mock", client=mock_client)
    bus = FakeMessageBus()
    store = FakeResultStore()
    processor = ResearchProcessor("research-agent-1", bus, executor, result_store=store)
    task = TaskRequested.create("Compare local and cloud LLMs")

    await processor.run_once(task)

    assert task.task_id in store.saved
    saved = store.saved[task.task_id]
    assert saved.task_id == task.task_id
    assert "Research query: Compare local and cloud LLMs" in saved.output
    assert "Summary: Local LLMs offer privacy; cloud LLMs scale seamlessly." in saved.output
    assert "[1] Local vs Cloud (https://example.com/llm-analysis)" in saved.output
    assert "Benchmark shows 10x compute throughput on cloud." in saved.output
    assert len(bus.published) == 2


@pytest.mark.anyio
async def test_run_research_agent_wires_result_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_EXECUTOR", "stub")

    with (
        patch("research_agent.main.RedisBus") as mock_bus_cls,
        patch("research_agent.main.ResearchProcessor") as mock_processor_cls,
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_processor_cls.assert_called_once()
        _, kwargs = mock_processor_cls.call_args
        assert kwargs.get("result_store") == mock_bus


# --- Infrastructure Adapter Tests: StubInferenceProvider ---


@pytest.mark.anyio
async def test_stub_inference_provider_deterministic_synthesis() -> None:
    provider = StubInferenceProvider()
    question = "Survey Raft and Paxos"
    evidence = "Raft is leader-based. Paxos uses two-phase commits."

    res1 = await provider.synthesize(question=question, evidence=evidence)
    res2 = await provider.synthesize(question=question, evidence=evidence)

    assert res1.question == question
    assert res1.evidence == evidence
    assert res1.provider == "stub"
    assert res1.model == "stub"
    assert res1.content == res2.content
    assert "Synthesized research for: Survey Raft and Paxos" in res1.content
    assert "Grounded evidence:" in res1.content
    assert evidence in res1.content


@pytest.mark.anyio
async def test_stub_inference_provider_empty_evidence() -> None:
    provider = StubInferenceProvider()
    question = "Unknown concept search"
    evidence = "   "

    res = await provider.synthesize(question=question, evidence=evidence)

    assert res.question == question
    assert "No grounded evidence available." in res.content
    assert "Grounded evidence:" not in res.content


@pytest.mark.anyio
async def test_stub_inference_provider_simulated_failure() -> None:
    provider = StubInferenceProvider(failure_trigger="trigger_inference_error")
    question = "Prompt containing trigger_inference_error"

    with pytest.raises(RuntimeError, match="Simulated synthesis failure"):
        await provider.synthesize(question=question, evidence="Some evidence")


def test_stub_synthesizer_alias() -> None:
    assert StubSynthesizer is StubInferenceProvider


# --- Inference Provider Factory Tests ---


def test_create_inference_provider_default(monkeypatch: pytest.MonkeyPatch) -> None:
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
    provider = create_inference_provider("  STUB  ")
    assert isinstance(provider, StubInferenceProvider)


def test_create_inference_provider_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported inference provider: 'openai'"):
        create_inference_provider("openai")

    with pytest.raises(ValueError, match="Unsupported inference provider: 'gemini'"):
        create_inference_provider("gemini")


# --- Processor Integration Tests with InferenceProvider ---


@pytest.mark.anyio
async def test_research_processor_with_inference_provider_synthesizes_output() -> None:
    bus = FakeMessageBus()
    store = FakeResultStore()
    executor = StubResearchExecutor()
    inference_provider = StubInferenceProvider()
    processor = ResearchProcessor(
        "research-agent-1",
        bus,
        executor,
        result_store=store,
        inference_provider=inference_provider,
    )
    task = TaskRequested.create("Investigate raft log compaction")

    await processor.run_once(task)

    assert task.task_id in store.saved
    saved = store.saved[task.task_id]
    assert saved.task_id == task.task_id
    assert "Synthesized research for: Investigate raft log compaction" in saved.output
    assert f"Deterministic execution completed for task: {task.task_id}" in saved.output
    assert saved.acknowledged is True

    # Event stream verification
    assert len(bus.published) == 2
    assert bus.published[0].state == "working"
    assert bus.published[1].state == "idle"
    assert (
        "Completed: Synthesized research for: Investigate raft log compaction"
        in bus.published[1].detail
    )


@pytest.mark.anyio
async def test_research_processor_with_tavily_and_inference_provider() -> None:
    mock_client = AsyncMock()
    mock_client.search.return_value = {
        "query": "Evaluate state machine replication",
        "answer": "SMR requires consistent state across nodes.",
        "results": [
            {
                "title": "SMR Overview",
                "url": "https://example.com/smr",
                "content": "Consensus protocols implement state machine replication.",
            }
        ],
    }
    executor = TavilyResearchExecutor(api_key="tvly-mock", client=mock_client)
    inference_provider = StubInferenceProvider()
    bus = FakeMessageBus()
    store = FakeResultStore()
    processor = ResearchProcessor(
        "research-agent-1",
        bus,
        executor,
        result_store=store,
        inference_provider=inference_provider,
    )
    task = TaskRequested.create("Evaluate state machine replication")

    await processor.run_once(task)

    assert task.task_id in store.saved
    saved = store.saved[task.task_id]
    assert "Synthesized research for: Evaluate state machine replication" in saved.output
    assert "Grounded evidence:" in saved.output
    assert "Research query: Evaluate state machine replication" in saved.output
    assert "Summary: SMR requires consistent state across nodes." in saved.output
    assert "https://example.com/smr" in saved.output
    assert len(bus.published) == 2
    assert bus.published[1].state == "idle"


@pytest.mark.anyio
async def test_research_processor_inference_error_publishes_error_event() -> None:
    bus = FakeMessageBus()
    store = FakeResultStore()
    executor = StubResearchExecutor()
    inference_provider = StubInferenceProvider(failure_trigger="trigger_synthesis_fail")
    processor = ResearchProcessor(
        "research-agent-1",
        bus,
        executor,
        result_store=store,
        inference_provider=inference_provider,
    )
    task = TaskRequested.create("Task containing trigger_synthesis_fail in prompt")

    with pytest.raises(ResearchExecutionError, match="Simulated synthesis failure"):
        await processor.run_once(task)

    assert len(store.saved) == 0
    assert len(bus.published) == 2
    assert bus.published[0].state == "working"
    assert bus.published[1].state == "error"
    assert "Simulated synthesis failure" in bus.published[1].detail


# --- Main Service Wiring Tests for InferenceProvider ---


@pytest.mark.anyio
async def test_run_research_agent_wires_inference_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_EXECUTOR", "stub")
    monkeypatch.setenv("INFERENCE_PROVIDER", "stub")

    with (
        patch("research_agent.main.RedisBus") as mock_bus_cls,
        patch("research_agent.main.ResearchProcessor") as mock_processor_cls,
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_processor_cls.assert_called_once()
        _, kwargs = mock_processor_cls.call_args
        assert kwargs.get("result_store") == mock_bus
        assert isinstance(kwargs.get("inference_provider"), StubInferenceProvider)


@pytest.mark.anyio
async def test_run_research_agent_unsupported_inference_provider_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "unsupported_llm")

    with patch("research_agent.main.RedisBus") as mock_bus_cls:
        from research_agent.main import run_research_agent

        with pytest.raises(ValueError, match="Unsupported inference provider: 'unsupported_llm'"):
            await run_research_agent()

        mock_bus_cls.assert_not_called()


# --- Infrastructure Adapter Tests: CloudflareInferenceProvider ---


def test_cloudflare_provider_creation_default_model() -> None:
    provider = CloudflareInferenceProvider(account_id="acc-123", api_token="tok-456")
    assert provider._account_id == "acc-123"
    assert provider._api_token == "tok-456"
    assert provider.model == DEFAULT_CLOUDFLARE_AI_MODEL


def test_cloudflare_provider_creation_custom_model() -> None:
    provider = CloudflareInferenceProvider(
        account_id="acc-123",
        api_token="tok-456",
        model="@cf/meta/llama-3.3-70b-instruct",
    )
    assert provider.model == "@cf/meta/llama-3.3-70b-instruct"


def test_cloudflare_provider_missing_account_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(
        ValueError,
        match="CLOUDFLARE_ACCOUNT_ID environment variable is required",
    ):
        CloudflareInferenceProvider(api_token="tok-456")


def test_cloudflare_provider_missing_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_AI_API_TOKEN", raising=False)
    with pytest.raises(
        ValueError,
        match="CLOUDFLARE_AI_API_TOKEN environment variable is required",
    ):
        CloudflareInferenceProvider(account_id="acc-123")


def test_cloudflare_provider_secret_redaction_repr() -> None:
    secret = "secret-super-sensitive-token-999"
    provider = CloudflareInferenceProvider(account_id="acc-123", api_token=secret)
    rep = repr(provider)
    assert secret not in rep
    assert "[REDACTED]" in rep


# --- Factory Selection Tests: Cloudflare ---


def test_create_inference_provider_cloudflare_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "cloudflare")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc-env-1")
    monkeypatch.setenv("CLOUDFLARE_AI_API_TOKEN", "tok-env-1")
    monkeypatch.delenv("CLOUDFLARE_AI_MODEL", raising=False)

    provider = create_inference_provider()
    assert isinstance(provider, CloudflareInferenceProvider)
    assert provider._account_id == "acc-env-1"
    assert provider._api_token == "tok-env-1"
    assert provider.model == DEFAULT_CLOUDFLARE_AI_MODEL


def test_create_inference_provider_cloudflare_explicit_and_custom_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_AI_API_TOKEN", raising=False)

    provider = create_inference_provider(
        name="cloudflare",
        account_id="acc-custom",
        api_token="tok-custom",
        model="@cf/meta/llama-3.3-70b-instruct",
    )
    assert isinstance(provider, CloudflareInferenceProvider)
    assert provider._account_id == "acc-custom"
    assert provider._api_token == "tok-custom"
    assert provider.model == "@cf/meta/llama-3.3-70b-instruct"


def test_create_inference_provider_cloudflare_missing_account_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "cloudflare")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.setenv("CLOUDFLARE_AI_API_TOKEN", "tok-1")

    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID environment variable is required"):
        create_inference_provider()


def test_create_inference_provider_cloudflare_missing_api_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "cloudflare")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc-1")
    monkeypatch.delenv("CLOUDFLARE_AI_API_TOKEN", raising=False)

    with pytest.raises(ValueError, match="CLOUDFLARE_AI_API_TOKEN environment variable is required"):
        create_inference_provider()


# --- Synthesis Execution, Grounding, and Error Handling Tests ---


@pytest.mark.anyio
async def test_cloudflare_provider_successful_synthesis() -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": "Raft achieves consensus via leader election and log replication. Sourced from https://raft.github.io"
            },
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

    question = "How does Raft maintain consensus?"
    evidence = "Raft uses leader election and replicated logs (https://raft.github.io)."
    result = await provider.synthesize(question=question, evidence=evidence)

    assert result.question == question
    assert "Raft achieves consensus" in result.content
    assert result.evidence == evidence  # Grounded evidence preserved
    assert result.provider == "cloudflare"
    assert result.model == DEFAULT_CLOUDFLARE_AI_MODEL

    # Verify request payload
    mock_client.post.assert_called_once()
    url, kwargs = mock_client.post.call_args[0][0], mock_client.post.call_args[1]
    assert (
        "https://api.cloudflare.com/client/v4/accounts/test-acc/ai/run/@cf/meta/llama-3.1-8b-instruct"
        in url
    )
    assert kwargs["headers"]["Authorization"] == "Bearer test-tok"
    assert kwargs["headers"]["Content-Type"] == "application/json"

    messages = kwargs["json"]["messages"]
    assert len(messages) == 2
    system_msg = messages[0]["content"]
    assert "Answer the research question directly" in system_msg
    assert "Rely exclusively on the provided grounded evidence" in system_msg
    assert "Clearly identify key findings" in system_msg
    assert "Preserve and source relevant URLs" in system_msg
    assert "insufficient" in system_msg

    user_msg = messages[1]["content"]
    assert question in user_msg
    assert evidence in user_msg


@pytest.mark.anyio
async def test_cloudflare_provider_empty_evidence_formatting() -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": "The supplied evidence is insufficient to answer the question."
            },
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

    result = await provider.synthesize(question="What is Quantum Consensus?", evidence="   ")
    assert result.evidence == "   "
    assert "insufficient" in result.content

    _, kwargs = mock_client.post.call_args
    user_msg = kwargs["json"]["messages"][1]["content"]
    assert "[No grounded evidence provided]" in user_msg


@pytest.mark.anyio
async def test_cloudflare_provider_api_failure_error_status() -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=401,
        json={
            "result": None,
            "success": False,
            "errors": [{"code": 1000, "message": "Invalid API Token"}],
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

    with pytest.raises(
        RuntimeError,
        match=r"Cloudflare Workers AI API error \(status 401\): Invalid API Token",
    ):
        await provider.synthesize(question="What is consensus?", evidence="Evidence")


@pytest.mark.anyio
async def test_cloudflare_provider_network_failure() -> None:
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ConnectTimeout("Connection timed out")

    provider = CloudflareInferenceProvider(
        account_id="test-acc",
        api_token="test-tok",
        client=mock_client,
    )

    with pytest.raises(
        RuntimeError,
        match="Cloudflare Workers AI request failed: Connection timed out",
    ):
        await provider.synthesize(question="What is consensus?", evidence="Evidence")


@pytest.mark.anyio
async def test_cloudflare_provider_malformed_json_response() -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        content=b"not-json-content",
        headers={"Content-Type": "text/html"},
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

    with pytest.raises(RuntimeError, match="Cloudflare Workers AI returned invalid JSON"):
        await provider.synthesize(question="What is consensus?", evidence="Evidence")


@pytest.mark.anyio
async def test_cloudflare_provider_malformed_response_missing_result() -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={"success": True, "result": None},
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

    with pytest.raises(RuntimeError, match="missing or invalid 'result' object"):
        await provider.synthesize(question="What is consensus?", evidence="Evidence")


@pytest.mark.anyio
async def test_cloudflare_provider_malformed_response_missing_response_string() -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={"success": True, "result": {"other_field": 123}},
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

    with pytest.raises(RuntimeError, match="missing 'result.response' string"):
        await provider.synthesize(question="What is consensus?", evidence="Evidence")


@pytest.mark.anyio
async def test_cloudflare_provider_redacts_api_token_on_failure() -> None:
    secret_token = "secret-cf-super-token-xyz"
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.RequestError(
        f"Failed to connect using token {secret_token}"
    )

    provider = CloudflareInferenceProvider(
        account_id="test-acc",
        api_token=secret_token,
        client=mock_client,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await provider.synthesize(question="Test question", evidence="Test evidence")

    err_str = str(exc_info.value)
    assert secret_token not in err_str
    assert "[REDACTED]" in err_str


@pytest.mark.anyio
async def test_cloudflare_provider_normal_completion(caplog: pytest.LogCaptureFixture) -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": "Raft achieves consensus via leader election and log replication. Sourced from https://raft.github.io.",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 45,
                    "completion_tokens": 78,
                    "total_tokens": 123,
                },
            },
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

    with caplog.at_level(logging.INFO):
        result = await provider.synthesize(
            question="How does Raft maintain consensus?",
            evidence="Raft uses leader election and replicated logs (https://raft.github.io).",
        )

    assert result.truncated is False
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 45, "completion_tokens": 78, "total_tokens": 123}
    assert "[Response truncated" not in result.content
    assert result.content.endswith(".")
    # Safe metadata logged without secret exposure
    assert "Cloudflare synthesis completed" in caplog.text
    assert "finish_reason=stop" in caplog.text
    assert "prompt_tokens=45" in caplog.text
    assert "completion_tokens=78" in caplog.text
    assert "test-tok" not in caplog.text


@pytest.mark.anyio
async def test_cloudflare_provider_token_limit_completion(caplog: pytest.LogCaptureFixture) -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": "This approach can improve performance by assigning specialized agents to specific tasks, reducing latency across nodes.",
                "finish_reason": "length",
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 1024,
                    "total_tokens": 1144,
                },
            },
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

    evidence_text = "Multi-agent systems route specialized workloads (https://arxiv.org/abs/2308.08155)."
    with caplog.at_level(logging.WARNING):
        result = await provider.synthesize(
            question="Investigate multi-agent consensus",
            evidence=evidence_text,
        )

    # Must detect as truncated and not silently treat as complete
    assert result.truncated is True
    assert result.finish_reason == "length"
    assert "[Response truncated: token limit reached]" in result.content
    # Grounded evidence preserved
    assert result.evidence == evidence_text
    # Source attribution preserved
    assert "https://arxiv.org/abs/2308.08155" in result.content
    # Safe warning logged
    assert "Cloudflare synthesis output detected as truncated" in caplog.text
    assert "finish_reason=length" in caplog.text
    assert "test-tok" not in caplog.text


@pytest.mark.anyio
async def test_cloudflare_provider_response_ending_mid_sentence(caplog: pytest.LogCaptureFixture) -> None:
    mock_client = AsyncMock()
    # Response terminates abruptly mid-sentence with no trailing punctuation
    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": "This approach can improve performance by assigning specialized agents to specific",
                "finish_reason": "stop",
            },
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

    with caplog.at_level(logging.WARNING):
        result = await provider.synthesize(
            question="Analyze multi-agent partitioning",
            evidence="Evidence text (https://example.com/multi-agent).",
        )

    # Must be detected as truncated despite finish_reason='stop' because it terminated mid-sentence
    assert result.truncated is True
    assert "[Response truncated: token limit reached]" in result.content
    assert "https://example.com/multi-agent" in result.content
    assert "ends_mid_sentence=True" in caplog.text


@pytest.mark.anyio
async def test_cloudflare_provider_payload_enforces_requested_number_of_findings() -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": "1. Finding one.\n2. Finding two.\n3. Finding three.\n\nSources:\n- https://example.com",
                "finish_reason": "stop",
            },
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
        max_tokens=800,
    )

    question = "List 3 key findings on distributed consensus protocols"
    evidence = "Protocols include Raft, Paxos, PBFT (https://example.com)."
    result = await provider.synthesize(question=question, evidence=evidence)

    assert result.truncated is False
    assert "[Response truncated" not in result.content

    # Inspect payload sent to Cloudflare
    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args[1]["json"]
    assert payload["max_tokens"] == 800

    system_prompt = payload["messages"][0]["content"]
    assert "Answer only the requested number of findings" in system_prompt
    assert "include one concise explanation per finding" in system_prompt
    assert "include a short key finding per item only if space permits" in system_prompt
    assert "Avoid adding extra findings beyond the requested count" in system_prompt
    assert "Sources section" in system_prompt

    user_prompt = payload["messages"][1]["content"]
    assert "Answer only the requested number of findings" in user_prompt
    assert question in user_prompt


@pytest.mark.anyio
async def test_cloudflare_provider_payload_forbids_unrequested_additionally_section() -> None:
    mock_client = AsyncMock()
    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": "1. Key finding.\n\nSources:\n- https://example.com",
                "finish_reason": "stop",
            },
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

    await provider.synthesize(
        question="Explain Raft heartbeats",
        evidence="Leaders send periodic heartbeats (https://example.com).",
    )

    payload = mock_client.post.call_args[1]["json"]
    system_prompt = payload["messages"][0]["content"]
    user_prompt = payload["messages"][1]["content"]

    # Verify explicit ban on unrequested "Additionally" sections
    assert "Additionally" in system_prompt
    assert "do not add an unrequested 'Additionally' section" in system_prompt
    assert "Additionally" in user_prompt


# --- End-to-end Integration: Processor with Tavily + Cloudflare ---


@pytest.mark.anyio
async def test_research_processor_with_tavily_and_cloudflare_inference() -> None:
    mock_tavily_client = AsyncMock()
    mock_tavily_client.search.return_value = {
        "query": "Survey state machine replication",
        "answer": "SMR relies on consensus.",
        "results": [
            {
                "title": "SMR Research",
                "url": "https://example.com/smr-paper",
                "content": "Consensus engines replicate operations across state machines.",
            }
        ],
    }
    executor = TavilyResearchExecutor(api_key="tvly-mock", client=mock_tavily_client)

    mock_cf_client = AsyncMock()
    mock_cf_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": "Key Findings: State machine replication requires deterministic execution and consensus (https://example.com/smr-paper)."
            },
            "success": True,
            "errors": [],
            "messages": [],
        },
        request=httpx.Request(
            "POST",
            "https://api.cloudflare.com/client/v4/accounts/acc-1/ai/run/@cf/meta/llama-3.1-8b-instruct",
        ),
    )
    mock_cf_client.post.return_value = mock_cf_response

    inference_provider = CloudflareInferenceProvider(
        account_id="acc-1",
        api_token="tok-1",
        client=mock_cf_client,
    )

    bus = FakeMessageBus()
    store = FakeResultStore()
    processor = ResearchProcessor(
        "research-agent-1",
        bus,
        executor,
        result_store=store,
        inference_provider=inference_provider,
    )
    task = TaskRequested.create("Survey state machine replication")

    await processor.run_once(task)

    assert task.task_id in store.saved
    saved = store.saved[task.task_id]
    assert "Key Findings: State machine replication" in saved.output
    assert "https://example.com/smr-paper" in saved.output
    assert saved.acknowledged is True
    assert len(bus.published) == 2
    assert bus.published[0].state == "working"
    assert bus.published[1].state == "idle"
    assert "Completed: Key Findings" in bus.published[1].detail


# --- Main Service Wiring Tests for Cloudflare ---


@pytest.mark.anyio
async def test_run_research_agent_cloudflare_missing_credentials_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "cloudflare")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_AI_API_TOKEN", raising=False)

    with patch("research_agent.main.RedisBus") as mock_bus_cls:
        from research_agent.main import run_research_agent

        with pytest.raises(
            ValueError, match="CLOUDFLARE_ACCOUNT_ID environment variable is required"
        ):
            await run_research_agent()

        mock_bus_cls.assert_not_called()


@pytest.mark.anyio
async def test_run_research_agent_cloudflare_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_EXECUTOR", "stub")
    monkeypatch.setenv("INFERENCE_PROVIDER", "cloudflare")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc-main-test")
    monkeypatch.setenv("CLOUDFLARE_AI_API_TOKEN", "tok-main-test")

    with (
        patch("research_agent.main.RedisBus") as mock_bus_cls,
        patch("research_agent.main.ResearchProcessor") as mock_processor_cls,
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_processor_cls.assert_called_once()
        _, kwargs = mock_processor_cls.call_args
        assert kwargs.get("result_store") == mock_bus
        provider = kwargs.get("inference_provider")
        assert isinstance(provider, CloudflareInferenceProvider)
        assert provider._account_id == "acc-main-test"
        assert provider._api_token == "tok-main-test"
        assert provider.model == DEFAULT_CLOUDFLARE_AI_MODEL


# --- Capability Metadata Registration Tests ---


@pytest.mark.anyio
async def test_research_agent_redis_bus_register() -> None:
    bus = RedisBus("redis://localhost:6379/0", queue_name="agent.tasks.research")
    bus.client.sadd = AsyncMock()  # type: ignore[method-assign]
    bus.client.hset = AsyncMock()  # type: ignore[method-assign]

    metadata = AgentCapabilityMetadata(
        agent_id="research-agent-1",
        capabilities=["research"],
        agent_type="research-agent",
        providers={"research_executor": "tavily", "inference_provider": "cloudflare"},
    )

    await bus.register(metadata)

    bus.client.sadd.assert_awaited_once_with("agent.ids", "research-agent-1")
    bus.client.hset.assert_awaited_once()
    key, kwargs = bus.client.hset.call_args[0][0], bus.client.hset.call_args[1]
    assert key == "agent.state:research-agent-1"
    mapping = kwargs["mapping"]
    assert mapping["agent_id"] == "research-agent-1"
    assert mapping["agent_type"] == "research-agent"
    assert mapping["capabilities"] == '["research"]'
    assert mapping["research_executor"] == "tavily"
    assert mapping["inference_provider"] == "cloudflare"


@pytest.mark.anyio
async def test_run_research_agent_registers_capability_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_ID", "research-agent-custom")
    monkeypatch.setenv("RESEARCH_EXECUTOR", "stub")
    monkeypatch.setenv("INFERENCE_PROVIDER", "stub")

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
        metadata = mock_bus.register.call_args[0][0]
        assert isinstance(metadata, AgentCapabilityMetadata)
        assert metadata.agent_id == "research-agent-custom"
        assert metadata.capabilities == ["research"]
        assert metadata.agent_type == "research-agent"
        assert metadata.providers["research_executor"] == "stub"
        assert metadata.providers["inference_provider"] == "stub"

        mock_bus.publish.assert_awaited_once()
        event = mock_bus.publish.call_args[0][0]
        assert event.agent_id == "research-agent-custom"
        assert event.state == "idle"


@pytest.mark.anyio
async def test_research_agent_metadata_does_not_expose_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_tavily = "tvly-secret-key-12345"
    secret_cf_token = "cf-token-secret-67890"
    secret_cf_acc = "cf-account-id-abcdef"

    monkeypatch.setenv("AGENT_ID", "research-agent-secure")
    monkeypatch.setenv("RESEARCH_EXECUTOR", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", secret_tavily)
    monkeypatch.setenv("INFERENCE_PROVIDER", "cloudflare")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", secret_cf_acc)
    monkeypatch.setenv("CLOUDFLARE_AI_API_TOKEN", secret_cf_token)

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
        metadata = mock_bus.register.call_args[0][0]

        # Ensure values in metadata.providers are strictly provider names
        assert metadata.providers == {
            "research_executor": "tavily",
            "inference_provider": "cloudflare",
        }

        # Serialized dict and json must not contain any of the secrets
        meta_dict = metadata.to_dict()
        meta_json = metadata.to_json()
        for secret in (secret_tavily, secret_cf_token, secret_cf_acc):
            assert secret not in meta_json
            assert secret not in str(meta_dict)


# --- Issue #9: Research Evidence Quality and Authority Evaluation Tests ---


def test_evidence_evaluator_prioritizes_high_authority_domains() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        CandidateEvidence(
            url="https://medium.com/@random_user/consensus-post",
            title="My thoughts on consensus",
            content="Raft and Paxos are consensus algorithms used in distributed databases.",
            raw_score=0.7,
        ),
        CandidateEvidence(
            url="https://docs.python.org/3/library/consensus.html",
            title="Asyncio Distributed Consensus Documentation",
            content="Official Python documentation for distributed consensus and state verification primitives.",
            raw_score=0.75,
        ),
        CandidateEvidence(
            url="https://csail.mit.edu/research/consensus-study",
            title="Distributed Consensus Verification",
            content="Formal verification of Paxos and Raft consensus algorithms in academic distributed systems.",
            raw_score=0.8,
        ),
        CandidateEvidence(
            url="https://techblog.io/raft-explained",
            title="Raft Algorithm Explained",
            content="A clear explanation of how distributed nodes reach consensus with Raft leader election.",
            raw_score=0.72,
        ),
    ]

    results = evaluator.evaluate(candidates, query="distributed consensus verification")

    retained = [r for r in results if r.retained]
    assert len(retained) == 4

    # Top sources must be the institutional (MIT .edu) and official documentation (docs.python.org)
    top_domains = [r.domain for r in retained[:2]]
    assert "csail.mit.edu" in top_domains
    assert "docs.python.org" in top_domains

    # Check authority classifications
    mit_res = next(r for r in retained if r.domain == "csail.mit.edu")
    assert mit_res.authority_tier == AuthorityTier.HIGH
    assert mit_res.source_type == SourceType.ACADEMIC_GOV
    assert "institutional-domain" in mit_res.quality_signals

    doc_res = next(r for r in retained if r.domain == "docs.python.org")
    assert doc_res.authority_tier == AuthorityTier.HIGH
    assert doc_res.source_type == SourceType.OFFICIAL_DOCS
    assert "official-documentation" in doc_res.quality_signals

    # Neutral sources are lower rank but still retained
    neutral_res = next(r for r in retained if r.domain == "techblog.io")
    assert neutral_res.authority_tier == AuthorityTier.MEDIUM or neutral_res.authority_tier == AuthorityTier.NEUTRAL
    assert neutral_res.retained is True
    assert mit_res.rank_score > neutral_res.rank_score


def test_evidence_evaluator_filters_promotional_and_spam_content() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        # Promotional affiliate query parameter
        CandidateEvidence(
            url="https://reviewsite.com/best-tools?aff=affiliate123&utm_medium=affiliate",
            title="Best Distributed Tools Review",
            content="Here is a breakdown of top consensus tools for your engineering stack.",
        ),
        # Commercial promotional call-to-action in text
        CandidateEvidence(
            url="https://deals.example.com/cloud-special",
            title="Consensus Software - Buy Now for 50% Off",
            content="Exclusive discount! Buy now and save with our special promo code for distributed database licenses.",
        ),
        # Known low-value domain
        CandidateEvidence(
            url="https://coupons.com/deal/database-software",
            title="Coupons and Promo Codes for Distributed Databases",
            content="Find the lowest price guarantee and promotional discounts on database subscriptions.",
        ),
        # Legitimate technical content
        CandidateEvidence(
            url="https://ietf.org/rfc/rfc9999.txt",
            title="RFC 9999 - State Machine Replication Protocol",
            content="This document specifies state machine replication and consensus requirements for resilient networks.",
        ),
    ]

    results = evaluator.evaluate(candidates, query="state machine replication consensus")

    retained = [r for r in results if r.retained]
    filtered = [r for r in results if not r.retained]

    # Only the IETF standards RFC should be retained
    assert len(retained) == 1
    assert retained[0].domain == "ietf.org"
    assert retained[0].authority_tier == AuthorityTier.HIGH

    # All three promotional sources must be filtered with FilterReason.PROMOTIONAL
    assert len(filtered) == 3
    for f in filtered:
        assert f.retained is False
        assert f.filter_reason == FilterReason.PROMOTIONAL
        assert "Filtered: Detected promotional marketing" in f.rationale


def test_evidence_evaluator_preserves_neutral_and_unknown_domains() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        CandidateEvidence(
            url="https://indie-dev.net/post/distributed-systems",
            title="Notes on Building Distributed Systems",
            content="Personal engineering notes discussing quorum, consensus, and network partitions.",
        ),
        CandidateEvidence(
            url="https://software-journal.com/paxos-in-practice",
            title="Paxos in Practical Engineering",
            content="Case study of deploying Paxos consensus across multi-datacenter clusters.",
        ),
    ]

    results = evaluator.evaluate(candidates, query="distributed consensus quorum")

    # Neutral / unknown commercial domains must NOT be automatically rejected
    assert len(results) == 2
    for r in results:
        assert r.retained is True
        assert r.authority_tier == AuthorityTier.NEUTRAL or r.authority_tier == AuthorityTier.MEDIUM
        assert r.filter_reason is None
        assert "Retained" in r.rationale


def test_evidence_evaluator_handles_duplicate_urls_and_content() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        CandidateEvidence(
            url="https://example.com/raft-paper",
            title="Raft Consensus Paper",
            content="In Search of an Understandable Consensus Algorithm by Diego Ongaro and John Ousterhout.",
        ),
        # Duplicate URL (with trailing slash)
        CandidateEvidence(
            url="https://example.com/raft-paper/",
            title="Raft Consensus Paper (Copy)",
            content="Identical paper copy discussing Raft consensus algorithm.",
        ),
        # Different URL but duplicate snippet content
        CandidateEvidence(
            url="https://mirror.org/paper-mirror",
            title="Mirror of Raft Consensus Paper",
            content="In Search of an Understandable Consensus Algorithm by Diego Ongaro and John Ousterhout.",
        ),
        # Distinct content
        CandidateEvidence(
            url="https://example.org/paxos-paper",
            title="Paxos Algorithm Overview",
            content="Leslie Lamport's seminal work on Paxos consensus and fault tolerance in asynchronous systems.",
        ),
    ]

    results = evaluator.evaluate(candidates, query="raft consensus algorithm")

    retained = [r for r in results if r.retained]
    filtered = [r for r in results if not r.retained]

    # Two distinct papers retained
    assert len(retained) == 2
    # Two duplicates filtered with FilterReason.DUPLICATE
    assert len(filtered) == 2
    for f in filtered:
        assert f.filter_reason == FilterReason.DUPLICATE
        assert "Duplicate" in f.rationale


def test_evidence_evaluator_handles_missing_metadata_conservatively() -> None:
    evaluator = EvidenceEvaluator()
    candidate = CandidateEvidence(
        url="https://unknown-engineering.org/consensus-notes",
        title="Consensus Notes",
        content="Overview of leader election and log replication in distributed state machines.",
        raw_score=None,
        published_date=None,
    )

    results = evaluator.evaluate([candidate], query="consensus leader election")
    assert len(results) == 1
    res = results[0]

    assert res.retained is True
    assert res.recency_indicator == "unknown"
    assert 0.0 < res.rank_score < 1.0
    assert "recency:unknown" in res.quality_signals
    # Must not crash or invent a published date
    assert "published" not in res.rationale.lower()


def test_evidence_evaluator_filters_malformed_and_unsafe_urls() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        CandidateEvidence(
            url="javascript:alert('malicious')",
            title="Malicious Link",
            content="Some technical content here describing distributed consensus.",
        ),
        CandidateEvidence(
            url="https://admin:supersecret@cluster.internal/status",
            title="Private Cluster Status",
            content="Internal status of node consensus and leader heartbeats.",
        ),
        CandidateEvidence(
            url="http://127.0.0.1:8080/data",
            title="Localhost Endpoint",
            content="Local data dump on node consensus.",
        ),
        CandidateEvidence(
            url="https://valid-site.org/consensus",
            title="Valid Technical Analysis",
            content="Discussion of consensus protocols and network partitions.",
        ),
    ]

    results = evaluator.evaluate(candidates, query="consensus protocol")

    retained = [r for r in results if r.retained]
    filtered = [r for r in results if not r.retained]

    assert len(retained) == 1
    assert retained[0].domain == "valid-site.org"

    assert len(filtered) == 3
    for f in filtered:
        assert f.filter_reason == FilterReason.MALFORMED_URL

    # Verify userinfo was redacted
    credential_item = next(f for f in filtered if "supersecret" not in f.url)
    assert "supersecret" not in credential_item.url
    assert "supersecret" not in credential_item.rationale


def test_evidence_evaluator_diversity_capping() -> None:
    evaluator = EvidenceEvaluator(max_per_root_domain=2)
    candidates = [
        CandidateEvidence(
            url=f"https://blog.techcorp.com/article-{i}",
            title=f"TechCorp Consensus Guide Part {i}",
            content=f"Detailed article {i} covering consensus mechanisms, raft, and fault tolerance.",
        )
        for i in range(1, 5)
    ]

    results = evaluator.evaluate(candidates, query="consensus raft")

    retained = [r for r in results if r.retained]
    filtered = [r for r in results if not r.retained]

    # Exactly 2 retained from techcorp.com, 2 capped
    assert len(retained) == 2
    assert len(filtered) == 2
    for f in filtered:
        assert f.filter_reason == FilterReason.DOMAIN_CAP_EXCEEDED
        assert "techcorp.com" in f.rationale


def test_evidence_evaluator_cross_domain_keyword_overlap_detection() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        CandidateEvidence(
            url="https://first-lab.org/raft",
            title="Raft Algorithm Specification",
            content="Raft solves consensus using replicated logs and leader election.",
        ),
        CandidateEvidence(
            url="https://second-institute.edu/distributed-systems",
            title="Replicated Logs and Consensus",
            content="Study of replicated logs and leader election in distributed state machines.",
        ),
    ]

    results = evaluator.evaluate(candidates, query="replicated logs leader election consensus")

    assert len(results) == 2
    for r in results:
        assert r.retained is True
        assert r.cross_domain_overlap_count > 0
        assert any("cross-domain-keyword-overlap" in sig for sig in r.quality_signals)
        assert "cross-domain keyword overlap" in r.rationale


def test_evidence_evaluator_deterministic_ordering_for_equivalent_evidence() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        CandidateEvidence(
            url="https://b-source.com/consensus",
            title="Consensus Guide B",
            content="First distinct engineering overview of distributed consensus algorithms.",
        ),
        CandidateEvidence(
            url="https://a-source.com/consensus",
            title="Consensus Guide A",
            content="Second distinct engineering overview of distributed consensus algorithms.",
        ),
    ]

    results1 = evaluator.evaluate(candidates, query="distributed consensus")
    results2 = evaluator.evaluate(candidates, query="distributed consensus")

    assert [r.url for r in results1] == [r.url for r in results2]
    # Domain tie-break puts a-source.com before b-source.com
    assert results1[0].domain == "a-source.com"
    assert results1[1].domain == "b-source.com"


def test_evaluated_evidence_serialization_and_credential_sanitization() -> None:
    item = EvaluatedEvidence(
        url="https://docs.python.org/3/library/asyncio.html",
        title="Asyncio Docs",
        content="Python asynchronous framework.",
        domain="docs.python.org",
        authority_tier=AuthorityTier.HIGH,
        source_type=SourceType.OFFICIAL_DOCS,
        quality_signals=("official-documentation", "high-relevance"),
        relevance_signal="high",
        recency_indicator="recent",
        rank_score=0.92,
        retained=True,
        filter_reason=None,
        rationale="Retained: Official documentation.",
        cross_domain_overlap_count=2,
    )

    data = item.to_dict()
    assert data["authority_tier"] == "high"
    assert data["source_type"] == "official_docs"
    assert data["rank_score"] == 0.92
    assert data["retained"] is True
    assert data["cross_domain_overlap_count"] == 2

    # Ensure no credentials or disallowed tokens in serialized dictionary
    disallowed_keys = {"api_key", "secret", "token", "password", "auth"}
    assert not any(k in disallowed_keys for k in data)

    # Round-trip through from_dict
    restored = EvaluatedEvidence.from_dict(data)
    assert restored.url == item.url
    assert restored.domain == item.domain
    assert restored.authority_tier == AuthorityTier.HIGH
    assert restored.source_type == SourceType.OFFICIAL_DOCS
    assert restored.quality_signals == item.quality_signals
    assert restored.rank_score == item.rank_score
    assert restored.retained is True
    assert restored.cross_domain_overlap_count == 2


@pytest.mark.anyio
async def test_tavily_executor_filters_promotional_and_prioritizes_authority() -> None:
    mock_client = AsyncMock()
    mock_client.search.return_value = {
        "query": "Research consensus protocols",
        "results": [
            {
                "title": "Special Deal - 50% Off Database Licenses",
                "url": "https://promo-deals.com/buy-now?aff=affiliate_partner",
                "content": "Buy now and save! Best price guarantee on database tools with our promo code.",
            },
            {
                "title": "IETF RFC - State Machine Replication Protocol",
                "url": "https://ietf.org/rfc/rfc9999.txt",
                "content": "Standard specification for fault-tolerant consensus protocols and replicated state machines.",
                "score": 0.88,
            },
            {
                "title": "Independent Developer Blog on Consensus",
                "url": "https://tech-notes.io/distributed-consensus",
                "content": "Practical analysis of Paxos and Raft consensus protocols under latency and network partitions.",
                "score": 0.75,
            },
            {
                "title": "Duplicate Link to RFC",
                "url": "https://ietf.org/rfc/rfc9999.txt",
                "content": "Standard specification for fault-tolerant consensus protocols.",
            },
        ],
    }

    executor = TavilyResearchExecutor(api_key="tvly-mock", client=mock_client)
    task = TaskRequested.create("Research consensus protocols")

    result = await executor.execute(task)

    # 1. Output must contain only retained sources
    assert "promo-deals.com" not in result.output
    assert "Buy now" not in result.output
    assert "ietf.org" in result.output
    assert "tech-notes.io" in result.output

    # 2. Output must format domain-quality indicators alongside URLs
    assert "Authority: high (standards)" in result.output
    assert "Domain: ietf.org" in result.output
    assert "Domain: tech-notes.io" in result.output

    # 3. High-authority source is prioritized as [1]
    lines = result.output.split("\n")
    source_lines = [l for l in lines if l.startswith(("[1]", "[2]"))]
    assert "[1] IETF RFC" in source_lines[0]
    assert "[2] Independent Developer Blog" in source_lines[1]

    # 4. Structured sources are preserved on ResearchResult
    assert len(result.sources) == 4
    retained_sources = result.retained_sources
    assert len(retained_sources) == 2
    assert retained_sources[0].domain == "ietf.org"
    assert retained_sources[0].authority_tier == AuthorityTier.HIGH

    filtered_sources = [s for s in result.sources if not s.retained]
    assert len(filtered_sources) == 2
    reasons = {s.filter_reason for s in filtered_sources}
    assert FilterReason.PROMOTIONAL in reasons
    assert FilterReason.DUPLICATE in reasons


@pytest.mark.anyio
async def test_tavily_executor_all_sources_filtered_yields_explanatory_message() -> None:
    mock_client = AsyncMock()
    mock_client.search.return_value = {
        "query": "Research best database",
        "results": [
            {
                "title": "Affiliate Review Site",
                "url": "https://spammydeals.com/best-databases?aff=partner_tag",
                "content": "Order now with our discount code for maximum savings on databases.",
            }
        ],
    }

    executor = TavilyResearchExecutor(api_key="tvly-mock", client=mock_client)
    task = TaskRequested.create("Research best database")

    result = await executor.execute(task)

    assert "No verified research evidence passed quality criteria" in result.output
    assert len(result.sources) == 1
    assert result.sources[0].retained is False
    assert result.sources[0].filter_reason == FilterReason.PROMOTIONAL


@pytest.mark.anyio
async def test_research_processor_preserves_sources_through_synthesis_and_saves_to_redis() -> None:
    mock_tavily = AsyncMock()
    mock_tavily.search.return_value = {
        "query": "Compare Raft and Paxos",
        "results": [
            {
                "title": "Raft Consensus Algorithm",
                "url": "https://raft.github.io",
                "content": "Raft is an understandable consensus algorithm designed as an alternative to Paxos.",
                "score": 0.85,
            },
            {
                "title": "Buy Raft License",
                "url": "https://deals.example.com/buy-now?aff=test",
                "content": "Special promo code for Raft book purchase.",
            },
        ],
    }

    mock_inference = AsyncMock()
    mock_inference.synthesize.return_value = SynthesisResult(
        question="Compare Raft and Paxos",
        content="Key Findings:\n1. Raft uses leader election and replicated logs.\n\nSources:\n- https://raft.github.io",
        evidence="Grounded evidence snippet",
        provider="cloudflare",
        model="@cf/meta/llama-3.1-8b-instruct",
        truncated=False,
    )

    fake_bus = FakeMessageBus()
    fake_store = FakeResultStore()
    executor = TavilyResearchExecutor(api_key="tvly-mock", client=mock_tavily)

    processor = ResearchProcessor(
        agent_id="research-agent-1",
        bus=fake_bus,
        executor=executor,
        result_store=fake_store,
        inference_provider=mock_inference,
    )

    task = TaskRequested.create("Compare Raft and Paxos")
    await processor.run_once(task)

    # 1. Synthesis only received evidence containing retained sources (no promotional content)
    mock_inference.synthesize.assert_awaited_once()
    evidence_sent_to_llm = mock_inference.synthesize.call_args[1]["evidence"]
    assert "https://raft.github.io" in evidence_sent_to_llm
    assert "deals.example.com" not in evidence_sent_to_llm
    assert "Authority: " in evidence_sent_to_llm

    # 2. ResultStore saved ResearchResult with output AND structured evaluated sources
    assert task.task_id in fake_store.saved
    saved_result = fake_store.saved[task.task_id]
    assert "Key Findings:" in saved_result.output
    assert len(saved_result.sources) == 2
    assert len(saved_result.retained_sources) == 1
    assert saved_result.retained_sources[0].domain == "raft.github.io"


def test_task_result_contract_round_trip_with_sources() -> None:
    sources = [
        {
            "url": "https://raft.github.io",
            "domain": "raft.github.io",
            "authority_tier": "high",
            "source_type": "official_docs",
            "quality_signals": ["official-documentation", "high-relevance"],
            "rank_score": 0.88,
            "retained": True,
            "rationale": "Retained: High-authority official documentation.",
        }
    ]

    original = TaskResult(
        task_id="task-research-1",
        output="Synthesized research findings",
        status="completed",
        acknowledged=True,
        sources=sources,
    )

    serialized = original.to_json()
    restored = TaskResult.from_json(serialized)

    assert restored.task_id == original.task_id
    assert restored.output == original.output
    assert restored.status == original.status
    assert restored.sources == sources


def test_evidence_evaluator_uses_per_item_raw_score_in_phase_3_regardless_of_order() -> None:
    evaluator = EvidenceEvaluator()
    cand_high = CandidateEvidence(
        url="https://alpha-site.com/consensus",
        title="Consensus Overview Alpha",
        content="Alpha lab findings on distributed consensus protocols and state replication.",
        raw_score=0.95,
    )
    cand_low = CandidateEvidence(
        url="https://beta-site.com/consensus",
        title="Consensus Overview Beta",
        content="Beta engineering analysis of distributed consensus protocols and cluster nodes.",
        raw_score=0.10,
    )

    # In [cand_high, cand_low], cand_low is evaluated last in Phase 1
    res_hl = evaluator.evaluate([cand_high, cand_low], query="distributed consensus protocols")
    # In [cand_low, cand_high], cand_high is evaluated last in Phase 1
    res_lh = evaluator.evaluate([cand_low, cand_high], query="distributed consensus protocols")

    # Both items must be retained (distinct content)
    assert len(res_hl) == 2
    assert len(res_lh) == 2

    # cand_high must have a higher rank_score than cand_low due to its own raw_score
    item_high = next(r for r in res_hl if r.url == cand_high.url)
    item_low = next(r for r in res_hl if r.url == cand_low.url)
    assert item_high.rank_score > item_low.rank_score

    # Crucially, evaluation must be strictly order-invariant (Phase 3 must not use stale loop variable)
    assert [r.url for r in res_hl] == [r.url for r in res_lh]
    assert [r.rank_score for r in res_hl] == [r.rank_score for r in res_lh]
    assert res_hl[0].url == cand_high.url


def test_evidence_evaluator_does_not_filter_legitimate_ref_query_params() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        CandidateEvidence(
            url="https://github.com/kubernetes/kubernetes/blob/main/README.md?ref=badge_shield",
            title="Kubernetes Readme",
            content="Production-grade container orchestration and cluster management documentation.",
        ),
        CandidateEvidence(
            url="https://stackoverflow.com/questions/12345/consensus?ref=hot",
            title="StackOverflow Consensus Discussion",
            content="Detailed community explanation of Raft leader election and split votes.",
        ),
        CandidateEvidence(
            url="https://docs.python.org/3/library/asyncio.html?ref=sidebar",
            title="Asyncio Documentation",
            content="Official documentation for asynchronous tasks and event loops in Python.",
        ),
    ]

    results = evaluator.evaluate(candidates, query="kubernetes consensus asyncio")
    retained = [r for r in results if r.retained]
    filtered = [r for r in results if not r.retained]

    # All three legitimate sources with ?ref= must be retained (not flagged as promotional)
    assert len(retained) == 3
    assert len(filtered) == 0


def test_evidence_evaluator_rejects_untrusted_docs_and_developer_subdomains_from_high_authority() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        CandidateEvidence(
            url="https://docs.unknown-startup.xyz/api",
            title="Startup API",
            content="Overview of startup API endpoints for data storage.",
        ),
        CandidateEvidence(
            url="https://developer.untrusted-example.com/guide",
            title="Untrusted Developer Guide",
            content="Developer guide for distributed database connections.",
        ),
        CandidateEvidence(
            url="https://docs.python.org/3/library/sqlite3.html",
            title="Python SQLite Documentation",
            content="Official documentation for the sqlite3 module in standard library.",
        ),
        CandidateEvidence(
            url="https://developer.mozilla.org/en-US/docs/Web/API",
            title="MDN Web Docs API",
            content="Official web standards documentation for browser APIs.",
        ),
    ]

    results = evaluator.evaluate(candidates, query="api documentation distributed")

    startup_res = next(r for r in results if r.domain == "docs.unknown-startup.xyz")
    untrusted_res = next(r for r in results if r.domain == "developer.untrusted-example.com")
    python_res = next(r for r in results if r.domain == "docs.python.org")
    mdn_res = next(r for r in results if r.domain == "developer.mozilla.org")

    # Untrusted subdomains must receive NEUTRAL authority (not elevated to HIGH)
    assert startup_res.authority_tier == AuthorityTier.NEUTRAL
    assert startup_res.source_type == SourceType.GENERAL_EDITORIAL
    assert untrusted_res.authority_tier == AuthorityTier.NEUTRAL
    assert untrusted_res.source_type == SourceType.GENERAL_EDITORIAL

    # Explicit trusted official docs domains must still receive HIGH authority
    assert python_res.authority_tier == AuthorityTier.HIGH
    assert python_res.source_type == SourceType.OFFICIAL_DOCS
    assert mdn_res.authority_tier == AuthorityTier.HIGH
    assert mdn_res.source_type == SourceType.OFFICIAL_DOCS


def test_evidence_evaluator_does_not_infer_recency_from_content_numbers() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        CandidateEvidence(
            url="https://standards.org/rfc2025",
            title="RFC 2025",
            content="RFC 2025 specifies state machine replication protocols for clusters.",
            published_date=None,
        ),
        CandidateEvidence(
            url="https://network-guide.org/ports",
            title="Port Numbers",
            content="Port 2025 is reserved for consensus heartbeats across nodes.",
            published_date=None,
        ),
        CandidateEvidence(
            url="https://benchmarks.org/nodes",
            title="Benchmark 2025 Nodes",
            content="We benchmarked 2025 concurrent connections under peak cluster load.",
            published_date=None,
        ),
        CandidateEvidence(
            url="https://actual-date.org/paper",
            title="Current Research Paper",
            content="Discussion of state machine replication.",
            published_date="2025-06-15",
        ),
        CandidateEvidence(
            url="https://historical-date.org/paper",
            title="Historical Research Paper",
            content="Foundational Paxos consensus discussion.",
            published_date="1998-05-01",
        ),
    ]

    results = evaluator.evaluate(candidates, query="state machine replication consensus")

    rfc_res = next(r for r in results if "rfc2025" in r.url)
    port_res = next(r for r in results if "ports" in r.url)
    bench_res = next(r for r in results if "benchmarks" in r.url)
    recent_res = next(r for r in results if "actual-date" in r.url)
    hist_res = next(r for r in results if "historical-date" in r.url)

    # Content-based numbers like RFC 2025, Port 2025 must NOT produce "recent"
    assert rfc_res.recency_indicator == "unknown"
    assert port_res.recency_indicator == "unknown"
    assert bench_res.recency_indicator == "unknown"

    # Explicit metadata must be parsed correctly
    assert recent_res.recency_indicator == "recent"
    assert hist_res.recency_indicator == "historical"


def test_evidence_evaluator_does_not_elevate_unlisted_org_io_dev_tlds() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        CandidateEvidence(
            url="https://unlisted-blog.org/consensus",
            title="Unlisted Org Blog",
            content="First distinct perspective on consensus protocols and state machine replication.",
        ),
        CandidateEvidence(
            url="https://unlisted-project.io/consensus",
            title="Unlisted IO Project",
            content="Second distinct perspective on consensus protocols and network partitions.",
        ),
        CandidateEvidence(
            url="https://unlisted-developer.dev/consensus",
            title="Unlisted Dev Portfolio",
            content="Third distinct perspective on distributed consensus implementations.",
        ),
        CandidateEvidence(
            url="https://standard-blog.com/consensus",
            title="Standard Com Blog",
            content="Fourth distinct perspective on raft and paxos algorithms.",
        ),
    ]

    results = evaluator.evaluate(candidates, query="consensus protocols state machine replication")

    for r in results:
        # None of these unlisted TLDs should be elevated to MEDIUM or TECHNICAL_REFERENCE
        assert r.authority_tier == AuthorityTier.NEUTRAL
        assert r.source_type == SourceType.GENERAL_EDITORIAL
        assert "developer-web-tld" not in r.quality_signals


def test_evidence_evaluator_compound_domain_diversity_capping() -> None:
    evaluator = EvidenceEvaluator(max_per_root_domain=2)
    candidates = [
        # Two distinct .co.uk domains (must not be collapsed into "co.uk")
        CandidateEvidence(
            url="https://www.bbc.co.uk/news/technology-1",
            title="BBC Tech Article 1",
            content="UK national broadcaster reporting on distributed database consensus.",
        ),
        CandidateEvidence(
            url="https://www.bbc.co.uk/news/technology-2",
            title="BBC Tech Article 2",
            content="UK national broadcaster reporting on distributed system consensus reliability.",
        ),
        CandidateEvidence(
            url="https://www.bbc.co.uk/news/technology-3",
            title="BBC Tech Article 3",
            content="Third UK article on consensus protocols.",
        ),
        CandidateEvidence(
            url="https://www.theguardian.co.uk/technology/article-1",
            title="Guardian Article 1",
            content="Separate UK newspaper reporting on distributed database consensus.",
        ),
        # Two distinct .ac.uk academic domains
        CandidateEvidence(
            url="https://research.ox.ac.uk/consensus-paper",
            title="Oxford Consensus Research",
            content="Oxford University research into replicated state machine consensus algorithms.",
        ),
        CandidateEvidence(
            url="https://cl.cam.ac.uk/consensus-spec",
            title="Cambridge Consensus Research",
            content="Cambridge Computer Lab study on Paxos and Raft consensus performance.",
        ),
    ]

    results = evaluator.evaluate(candidates, query="distributed database consensus")

    retained = [r for r in results if r.retained]
    filtered = [r for r in results if not r.retained]

    # bbc.co.uk has 2 retained and 1 capped
    bbc_retained = [r for r in retained if r.domain == "bbc.co.uk"]
    assert len(bbc_retained) == 2

    # theguardian.co.uk has its own independent cap slot and is retained
    guardian_retained = [r for r in retained if r.domain == "theguardian.co.uk"]
    assert len(guardian_retained) == 1

    # ox.ac.uk and cam.ac.uk each have their own independent cap slots and are both retained
    ox_retained = [r for r in retained if r.domain == "research.ox.ac.uk"]
    cam_retained = [r for r in retained if r.domain == "cl.cam.ac.uk"]
    assert len(ox_retained) == 1
    assert len(cam_retained) == 1

    # The 3rd bbc.co.uk article was capped for exceeding max_per_root_domain
    assert len(filtered) == 1
    assert filtered[0].filter_reason == FilterReason.DOMAIN_CAP_EXCEEDED
    assert "bbc.co.uk" in filtered[0].rationale


def test_evaluated_evidence_from_dict_handles_unknown_enum_values_gracefully() -> None:
    data = {
        "url": "https://example.com/test",
        "title": "Example",
        "content": "Content snippet.",
        "domain": "example.com",
        "authority_tier": "unheard_of_authority",
        "source_type": "alien_source_type",
        "filter_reason": "unknown_filter_reason",
    }

    restored = EvaluatedEvidence.from_dict(data)
    assert restored.authority_tier == AuthorityTier.NEUTRAL
    assert restored.source_type == SourceType.UNKNOWN
    assert restored.filter_reason is None


def test_evaluated_evidence_from_dict_backward_compatible_with_corroboration_count_key() -> None:
    data = {
        "url": "https://example.com/test",
        "title": "Example",
        "content": "Content snippet.",
        "domain": "example.com",
        "corroboration_count": 4,
    }

    restored = EvaluatedEvidence.from_dict(data)
    assert restored.cross_domain_overlap_count == 4


def test_evidence_evaluator_preserves_conflicting_perspectives_without_forcing_consensus() -> None:
    evaluator = EvidenceEvaluator()
    candidates = [
        CandidateEvidence(
            url="https://pro-raft-research.org/study",
            title="Why Raft is Preferable to Paxos",
            content="Raft is superior for practical systems because leader election and log replication are understandable and maintainable.",
        ),
        CandidateEvidence(
            url="https://pro-paxos-research.org/study",
            title="Why Multi-Paxos Outperforms Raft",
            content="Paxos is superior for high throughput systems because leader-less variants avoid leader bottleneck and minimize latency.",
        ),
    ]

    results = evaluator.evaluate(candidates, query="raft versus paxos consensus superiority")

    # Both conflicting viewpoints must be retained without filtering
    retained = [r for r in results if r.retained]
    assert len(retained) == 2
    domains = {r.domain for r in retained}
    assert "pro-raft-research.org" in domains
    assert "pro-paxos-research.org" in domains


def test_task_result_backward_compatibility_with_pre_issue_9_payload() -> None:
    legacy_json = (
        '{"task_id": "task-legacy-1", "output": "Legacy output without sources", '
        '"status": "completed", "acknowledged": true}'
    )

    restored = TaskResult.from_json(legacy_json)
    assert restored.task_id == "task-legacy-1"
    assert restored.output == "Legacy output without sources"
    assert restored.status == "completed"
    assert restored.acknowledged is True
    assert restored.sources is None


