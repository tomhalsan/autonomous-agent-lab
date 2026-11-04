import pytest
from agent_contracts import TaskRequested
from orchestrator.application.queries import GetAgentStates, GetRecentEvents
from orchestrator.application.use_cases import SubmitTask
from orchestrator.domain.capability import AgentCapability


class FakeTaskQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[TaskRequested, AgentCapability]] = []

    async def enqueue(
        self,
        task: TaskRequested,
        capability: AgentCapability = AgentCapability.GENERAL,
    ) -> None:
        self.enqueued.append((task, capability))

class FakeStateReader:
    def __init__(
        self,
        agents_data: list[dict[str, str]] | None = None,
        events_data: list[dict[str, str]] | None = None,
    ) -> None:
        self.agents_data = agents_data or []
        self.events_data = events_data or []
        self.last_events_limit: int | None = None

    async def agents(self) -> list[dict[str, str]]:
        return self.agents_data

    async def events(self, limit: int = 50) -> list[dict[str, str]]:
        self.last_events_limit = limit
        return self.events_data[:limit]


@pytest.mark.anyio
async def test_submit_task_use_case() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(queue)
    task = await use_case.execute("test task prompt")

    assert task.prompt == "test task prompt"
    assert len(queue.enqueued) == 1

    enqueued_task, capability = queue.enqueued[0]

    assert enqueued_task.prompt == "test task prompt"
    assert capability == AgentCapability.GENERAL

@pytest.mark.anyio
async def test_get_agent_states_query() -> None:
    expected = [{"agent_id": "worker-1", "state": "idle", "detail": "Waiting"}]
    reader = FakeStateReader(agents_data=expected)
    query = GetAgentStates(reader)

    result = await query.execute()
    assert result == expected


@pytest.mark.anyio
async def test_get_recent_events_bounds_limits() -> None:
    sample_events = [{"event_id": str(i), "detail": f"event {i}"} for i in range(150)]
    reader = FakeStateReader(events_data=sample_events)
    query = GetRecentEvents(reader)

    # Test default
    await query.execute()
    assert reader.last_events_limit == 50

    # Test lower bound (<= 0 maps to 1)
    await query.execute(limit=0)
    assert reader.last_events_limit == 1

    await query.execute(limit=-10)
    assert reader.last_events_limit == 1

    # Test upper bound (> 100 maps to 100)
    await query.execute(limit=250)
    assert reader.last_events_limit == 100

    # Test valid within-range limit
    await query.execute(limit=25)
    assert reader.last_events_limit == 25

@pytest.mark.anyio
async def test_submit_research_task_routes_to_research_capability() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(queue)

    task = await use_case.execute(
        "Compare local and cloud LLMs",
        capability=AgentCapability.RESEARCH,
    )

    assert len(queue.enqueued) == 1

    enqueued_task, capability = queue.enqueued[0]

    assert enqueued_task == task
    assert capability == AgentCapability.RESEARCH
