from typing import Any

import pytest
from agent_contracts import TaskRequested
from orchestrator.application.queries import (
    GetAgentStates,
    GetRecentEvents,
    GetTaskResult,
)
from orchestrator.application.use_cases import SubmitTask
from orchestrator.domain.capability import AgentCapability


class FakeTaskQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[TaskRequested, AgentCapability]] = []
        self.dispatches: list[tuple[TaskRequested, str]] = []

    async def enqueue(
        self,
        task: TaskRequested,
        capability: AgentCapability = AgentCapability.GENERAL,
        target: str | None = None,
    ) -> None:
        self.enqueued.append((task, capability))
        if target:
            self.dispatches.append((task, target))

    async def dispatch(self, task: TaskRequested, target: str) -> None:
        self.dispatches.append((task, target))
        capability = (
            AgentCapability.RESEARCH if "research" in target else AgentCapability.GENERAL
        )
        self.enqueued.append((task, capability))

class FakeStateReader:
    def __init__(
        self,
        agents_data: list[dict[str, str]] | None = None,
        events_data: list[dict[str, str]] | None = None,
        routing_decisions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.agents_data = agents_data or []
        self.events_data = events_data or []
        self.routing_decisions = routing_decisions or {}
        self.last_events_limit: int | None = None

    async def agents(self) -> list[dict[str, str]]:
        return self.agents_data

    async def events(self, limit: int = 50) -> list[dict[str, str]]:
        self.last_events_limit = limit
        return self.events_data[:limit]

    async def routing_decision(self, task_id: str) -> dict[str, Any] | None:
        if task_id in self.routing_decisions:
            return self.routing_decisions[task_id]
        for event in self.events_data:
            if event.get("task_id") == task_id and event.get("routing_decision"):
                rd = event["routing_decision"]
                import json

                return json.loads(rd) if isinstance(rd, str) else rd
        return None

    async def task_inference_telemetry(self, task_id: str) -> dict[str, Any] | None:
        for event in self.events_data:
            if event.get("task_id") == task_id and event.get("inference_telemetry"):
                raw_telem = event["inference_telemetry"]
                import json

                try:
                    telem = (
                        json.loads(raw_telem) if isinstance(raw_telem, str) else raw_telem
                    )
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(telem, dict):
                    continue
                from agent_contracts import InferenceExecutionTelemetry

                validated = None
                try:
                    validated = InferenceExecutionTelemetry.from_dict(telem)
                except (KeyError, TypeError, ValueError):
                    pass
                if validated is not None and validated.status == "succeeded":
                    return {
                        "task_id": task_id,
                        "inference_telemetry": validated.to_dict(),
                    }
        return None


class FakeResultReader:
    def __init__(
        self,
        results: dict[str, dict[str, Any]] | None = None,
        pending: set[str] | None = None,
    ) -> None:
        self.results = results or {}
        self.pending = pending or set()

    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        if task_id in self.results:
            return self.results[task_id]
        if task_id in self.pending:
            return {"task_id": task_id, "status": "pending", "output": None}
        return None


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


@pytest.mark.anyio
async def test_submit_task_automatic_routing_research() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(queue)

    # Under Phase B routing policy, genuine research prompts require external retrieval/freshness/source evidence rather than weak verbs alone
    task = await use_case.execute("Compare current features and pricing of local and cloud LLMs")

    assert len(queue.enqueued) == 1
    enqueued_task, capability = queue.enqueued[0]
    assert enqueued_task == task
    assert capability == AgentCapability.RESEARCH


@pytest.mark.anyio
async def test_submit_task_automatic_routing_general() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(queue)

    task = await use_case.execute("build agent")

    assert len(queue.enqueued) == 1
    enqueued_task, capability = queue.enqueued[0]
    assert enqueued_task == task
    assert capability == AgentCapability.GENERAL


@pytest.mark.anyio
async def test_submit_task_routing_v2_freshness_and_explanatory() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(queue)

    # Freshness prompt automatically resolves to RESEARCH
    task_research = await use_case.execute(
        "What are the latest developments in battery technology?"
    )
    assert queue.enqueued[0] == (task_research, AgentCapability.RESEARCH)

    # General explanatory prompt automatically resolves to GENERAL
    task_general = await use_case.execute(
        "Explain the difference between containers and virtual machines."
    )
    assert queue.enqueued[1] == (task_general, AgentCapability.GENERAL)



@pytest.mark.anyio
async def test_submit_task_preserves_explicit_capability_override() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(queue)

    # Research prompt explicitly directed to GENERAL
    task1 = await use_case.execute(
        "Investigate system logs",
        capability=AgentCapability.GENERAL,
    )
    assert queue.enqueued[0] == (task1, AgentCapability.GENERAL)

    # General prompt explicitly directed to RESEARCH
    task2 = await use_case.execute(
        "build agent",
        capability=AgentCapability.RESEARCH,
    )
    assert queue.enqueued[1] == (task2, AgentCapability.RESEARCH)


class FakeCapabilityRouter:
    def __init__(self, capability_to_return: AgentCapability) -> None:
        self.capability_to_return = capability_to_return
        self.routed_prompts: list[str] = []

    def route(self, prompt: str) -> AgentCapability:
        self.routed_prompts.append(prompt)
        return self.capability_to_return


@pytest.mark.anyio
async def test_submit_task_uses_injected_router() -> None:
    queue = FakeTaskQueue()
    fake_router = FakeCapabilityRouter(AgentCapability.RESEARCH)
    use_case = SubmitTask(queue, router=fake_router)

    # Non-research prompt routed to RESEARCH via injected router
    task = await use_case.execute("simple task")

    assert fake_router.routed_prompts == ["simple task"]
    assert queue.enqueued[0] == (task, AgentCapability.RESEARCH)


@pytest.mark.anyio
async def test_get_task_result_query_completed() -> None:
    expected = {
        "task_id": "task-1",
        "status": "completed",
        "output": "research findings",
    }
    reader = FakeResultReader(results={"task-1": expected})
    query = GetTaskResult(reader)

    result = await query.execute("task-1")
    assert result == expected


@pytest.mark.anyio
async def test_get_task_result_query_pending() -> None:
    reader = FakeResultReader(pending={"task-pending-1"})
    query = GetTaskResult(reader)

    result = await query.execute("task-pending-1")
    assert result == {"task_id": "task-pending-1", "status": "pending", "output": None}


@pytest.mark.anyio
async def test_get_task_result_query_not_found() -> None:
    reader = FakeResultReader()
    query = GetTaskResult(reader)

    result = await query.execute("non-existent-task")
    assert result is None


@pytest.mark.anyio
async def test_get_task_inference_telemetry_query_success() -> None:
    from orchestrator.application.queries import GetTaskInferenceTelemetry

    expected_telem = {
        "provider_name": "cloudflare",
        "attempt_index": 0,
        "latency_ms": 94.3,
        "status": "succeeded",
        "model_name": "@cf/meta/llama-3.1-8b-instruct",
        "finish_reason": "stop",
    }
    reader = FakeStateReader(
        events_data=[
            {
                "task_id": "task-telemetry-1",
                "inference_telemetry": expected_telem,
            }
        ]
    )
    query = GetTaskInferenceTelemetry(reader)
    result = await query.execute("task-telemetry-1")
    assert result is not None
    assert result["task_id"] == "task-telemetry-1"
    assert result["inference_telemetry"]["provider_name"] == "cloudflare"
    assert result["inference_telemetry"]["status"] == "succeeded"


@pytest.mark.anyio
async def test_get_task_inference_telemetry_query_not_found() -> None:
    from orchestrator.application.queries import GetTaskInferenceTelemetry

    reader = FakeStateReader(events_data=[])
    query = GetTaskInferenceTelemetry(reader)
    result = await query.execute("task-missing")
    assert result is None


@pytest.mark.anyio
async def test_get_task_inference_telemetry_query_empty_task_id_raises() -> None:
    from orchestrator.application.queries import GetTaskInferenceTelemetry

    reader = FakeStateReader()
    query = GetTaskInferenceTelemetry(reader)
    with pytest.raises(ValueError, match="task_id must not be empty"):
        await query.execute("")
    with pytest.raises(ValueError, match="task_id must not be empty"):
        await query.execute("   ")

