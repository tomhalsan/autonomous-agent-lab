import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from agent_contracts import AgentEvent, TaskRequested
from orchestrator.application.queries import GetAgentStates, GetRecentEvents
from research_agent.application.processor import ResearchExecutionError, ResearchProcessor
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.redis_bus import RedisBus
from research_agent.infrastructure.stub_executor import StubResearchExecutor


class FakeMessageBus:
    def __init__(self) -> None:
        self.published: list[AgentEvent] = []

    async def publish(self, event: AgentEvent) -> None:
        self.published.append(event)

    async def next_task(self) -> TaskRequested:
        raise NotImplementedError


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
