import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from agent_contracts import AgentEvent, TaskRequested
from orchestrator.application.queries import GetAgentStates, GetRecentEvents
from research_agent.application.processor import ResearchExecutionError, ResearchProcessor
from research_agent.domain.models import ResearchResult, SynthesisResult
from research_agent.infrastructure.executor_factory import create_research_executor
from research_agent.infrastructure.inference_factory import create_inference_provider
from research_agent.infrastructure.redis_bus import RedisBus
from research_agent.infrastructure.stub_executor import StubResearchExecutor
from research_agent.infrastructure.stub_inference import StubInferenceProvider, StubSynthesizer
from research_agent.infrastructure.tavily_executor import TavilyResearchExecutor


class FakeMessageBus:
    def __init__(self) -> None:
        self.published: list[AgentEvent] = []

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

    with pytest.raises(ValueError, match="Unsupported inference provider: 'cloudflare'"):
        create_inference_provider("cloudflare")


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




