import json
from unittest.mock import AsyncMock

import pytest
from agent_contracts import (
    AgentEvent,
    RoutingDecision,
    TaskRequested,
)
from orchestrator.application.queries import GetRoutingDecision
from orchestrator.application.use_cases import SubmitTask
from orchestrator.config import Settings
from orchestrator.domain.capability import (
    CapabilityUnavailableError,
)
from orchestrator.infrastructure.redis_adapter import RedisGateway
from orchestrator.presentation.api import create_app, task_routing
from starlette.exceptions import HTTPException
from starlette.requests import Request

from tests.test_application import FakeStateReader, FakeTaskQueue
from tests.test_runtime_dispatch import FakeCandidateFinder, FakeEventPublisher, make_candidate


class FakeRoutingRecorder:
    def __init__(self) -> None:
        self.recorded: list[RoutingDecision] = []

    async def record_routing_decision(self, decision: RoutingDecision) -> None:
        self.recorded.append(decision)


# ==============================================================================
# 1. RoutingDecision Contract & Validation Tests
# ==============================================================================


def test_routing_decision_attributes() -> None:
    candidate_summary = {
        "agent_id": "research-1",
        "agent_type": "research-agent",
        "capabilities": ["research"],
        "execution_state": "idle",
        "availability_state": "ready",
        "dispatch_target": "agent.tasks.research",
        "is_eligible": True,
    }
    decision = RoutingDecision(
        task_id="task-123",
        prompt_classification="research",
        required_capabilities=["research"],
        candidates_evaluated=[candidate_summary],
        selected_agent_id="research-1",
        dispatch_target="agent.tasks.research",
        rationale="Selected research-1 targeting agent.tasks.research",
    )

    assert decision.task_id == "task-123"
    assert decision.prompt_classification == "research"
    assert decision.required_capabilities == ["research"]
    assert len(decision.candidates_evaluated) == 1
    assert decision.candidates_evaluated[0]["agent_id"] == "research-1"
    assert decision.selected_agent_id == "research-1"
    assert decision.dispatch_target == "agent.tasks.research"
    assert "Selected research-1" in decision.rationale
    assert decision.version == 1
    assert decision.decision_timestamp != ""


def test_routing_decision_serialization_round_trip() -> None:
    decision = RoutingDecision(
        task_id="task-abc",
        prompt_classification="general",
        required_capabilities=["general"],
        candidates_evaluated=[],
        selected_agent_id="worker-1",
        dispatch_target="agent.tasks.general",
        rationale="Selected worker-1 via deterministic selection",
    )

    data = decision.to_dict()
    assert data["task_id"] == "task-abc"
    assert data["selected_agent_id"] == "worker-1"

    restored = RoutingDecision.from_dict(data)
    assert restored == decision

    json_str = decision.to_json()
    from_json_restored = RoutingDecision.from_json(json_str)
    assert from_json_restored == decision


def test_routing_decision_validation_rejects_empty_task_id() -> None:
    with pytest.raises(ValueError, match="task_id must not be empty"):
        RoutingDecision(
            task_id="   ",
            prompt_classification="general",
            required_capabilities=["general"],
            candidates_evaluated=[],
            selected_agent_id=None,
            dispatch_target=None,
            rationale="No task",
        )


def test_routing_decision_validation_rejects_empty_prompt_classification() -> None:
    with pytest.raises(ValueError, match="prompt_classification must not be empty"):
        RoutingDecision(
            task_id="task-1",
            prompt_classification="   ",
            required_capabilities=["general"],
            candidates_evaluated=[],
            selected_agent_id=None,
            dispatch_target=None,
            rationale="No classification",
        )


def test_routing_decision_validation_rejects_empty_required_capabilities() -> None:
    with pytest.raises(ValueError, match="required_capabilities must be a non-empty list"):
        RoutingDecision(
            task_id="task-1",
            prompt_classification="general",
            required_capabilities=[],
            candidates_evaluated=[],
            selected_agent_id=None,
            dispatch_target=None,
            rationale="Empty capabilities",
        )


@pytest.mark.parametrize(
    "sensitive_key",
    ["api_key", "tavily_api_key", "token", "auth_token", "account_id", "secret", "user_password", "credential"],
)
def test_routing_decision_rejects_sensitive_keys(sensitive_key: str) -> None:
    with pytest.raises(ValueError, match="must not be exposed in routing decision"):
        RoutingDecision(
            task_id="task-sec",
            prompt_classification="research",
            required_capabilities=["research"],
            candidates_evaluated=[
                {
                    "agent_id": "research-1",
                    sensitive_key: "leaked_secret_value",
                }
            ],
            selected_agent_id="research-1",
            dispatch_target="agent.tasks.research",
            rationale="Selected research agent",
        )


# ==============================================================================
# 2. AgentEvent Carrying Routing Decision Tests
# ==============================================================================


def test_agent_event_carries_routing_decision() -> None:
    decision_dict = {
        "task_id": "task-evt",
        "prompt_classification": "research",
        "required_capabilities": ["research"],
        "candidates_evaluated": [],
        "selected_agent_id": "research-1",
        "dispatch_target": "agent.tasks.research",
        "rationale": "Routed to research specialist",
    }
    event = AgentEvent.create(
        agent_id="orchestrator",
        state="routed",
        detail="Routed to research specialist",
        task_id="task-evt",
        routing_decision=decision_dict,
    )

    assert event.routing_decision == decision_dict
    event_dict = event.to_dict()
    assert event_dict["routing_decision"] == decision_dict

    restored = AgentEvent.from_json(event.to_json())
    assert restored == event
    assert restored.routing_decision == decision_dict


def test_agent_event_omits_routing_decision_when_none() -> None:
    event = AgentEvent.create(
        agent_id="worker-1",
        state="idle",
        detail="Waiting for work",
    )
    assert event.routing_decision is None
    data = event.to_dict()
    assert "routing_decision" not in data


# ==============================================================================
# 3. SubmitTask Decision Recording Tests
# ==============================================================================


@pytest.mark.anyio
async def test_submit_task_records_routing_decision_on_dynamic_dispatch() -> None:
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()
    candidate = make_candidate("worker-1", capability="general", dispatch_target="agent.tasks.general")
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(
        queue=queue,
        candidate_finder=finder,
        routing_recorder=recorder,
    )

    task = await submit_task.execute("Process standard batch queue item")

    assert isinstance(task, TaskRequested)
    assert len(recorder.recorded) == 1
    decision = recorder.recorded[0]

    assert decision.task_id == task.task_id
    assert decision.prompt_classification == "general"
    assert decision.required_capabilities == ["general"]
    assert len(decision.candidates_evaluated) == 1
    assert decision.candidates_evaluated[0]["agent_id"] == "worker-1"
    assert decision.selected_agent_id == "worker-1"
    assert decision.dispatch_target == "agent.tasks.general"
    assert "worker-1" in decision.rationale


@pytest.mark.anyio
async def test_submit_task_records_routing_decision_for_research_intent() -> None:
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()
    candidate = make_candidate("research-1", capability="research", dispatch_target="agent.tasks.research")
    finder = FakeCandidateFinder({"research": [candidate]})

    submit_task = SubmitTask(
        queue=queue,
        candidate_finder=finder,
        routing_recorder=recorder,
    )

    task = await submit_task.execute("Research quantum key distribution breakthroughs")

    assert isinstance(task, TaskRequested)
    assert len(recorder.recorded) == 1
    decision = recorder.recorded[0]

    assert decision.task_id == task.task_id
    assert decision.prompt_classification in ("research", "research_and_investigation")
    assert decision.required_capabilities == ["research"]
    assert len(decision.candidates_evaluated) == 1
    assert decision.candidates_evaluated[0]["agent_id"] == "research-1"
    assert decision.selected_agent_id == "research-1"
    assert decision.dispatch_target == "agent.tasks.research"


@pytest.mark.anyio
async def test_submit_task_records_routing_decision_on_unavailable_capability() -> None:
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()
    events = FakeEventPublisher()
    finder = FakeCandidateFinder({"research": []})

    submit_task = SubmitTask(
        queue=queue,
        candidate_finder=finder,
        events=events,
        routing_recorder=recorder,
    )

    with pytest.raises(CapabilityUnavailableError):
        await submit_task.execute("Research deep space navigation")

    assert len(recorder.recorded) == 1
    decision = recorder.recorded[0]

    assert decision.selected_agent_id is None
    assert decision.dispatch_target is None
    assert decision.candidates_evaluated == []
    assert "No eligible agent is currently available" in decision.rationale

    assert len(events.published) == 1
    error_event = events.published[0]
    assert error_event.state == "error"
    assert error_event.routing_decision == decision.to_dict()


@pytest.mark.anyio
async def test_submit_task_records_routing_decision_on_legacy_fallback() -> None:
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()

    submit_task = SubmitTask(
        queue=queue,
        candidate_finder=None,
        routing_recorder=recorder,
    )

    task = await submit_task.execute("Legacy task prompt")

    assert len(recorder.recorded) == 1
    decision = recorder.recorded[0]
    assert decision.task_id == task.task_id
    assert decision.required_capabilities == ["general"]


# ==============================================================================
# 4. RedisGateway Persistence & StateReader Query Tests
# ==============================================================================


@pytest.mark.anyio
async def test_redis_gateway_records_and_reads_routing_decision() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.set = AsyncMock()
    gateway._client.get = AsyncMock()
    gateway._client.xadd = AsyncMock()

    decision = RoutingDecision(
        task_id="task-persist-1",
        prompt_classification="research",
        required_capabilities=["research"],
        candidates_evaluated=[],
        selected_agent_id="research-1",
        dispatch_target="agent.tasks.research",
        rationale="Selected research-1",
    )

    await gateway.record_routing_decision(decision)

    gateway._client.set.assert_awaited_once_with(
        "task.routing:task-persist-1",
        decision.to_json(),
    )

    # Read back
    gateway._client.get.return_value = decision.to_json()
    result = await gateway.routing_decision("task-persist-1")

    assert result is not None
    assert result["task_id"] == "task-persist-1"
    assert result["selected_agent_id"] == "research-1"


@pytest.mark.anyio
async def test_redis_gateway_routing_decision_falls_back_to_events_stream() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.get = AsyncMock(return_value=None)
    # Stream returns an event with routing_decision
    gateway.events = AsyncMock(
        return_value=[
            {
                "event_id": "1",
                "task_id": "task-fallback-1",
                "state": "routed",
                "routing_decision": json.dumps(
                    {
                        "task_id": "task-fallback-1",
                        "prompt_classification": "general",
                        "selected_agent_id": "worker-1",
                        "rationale": "Event stream fallback",
                    }
                ),
            }
        ]
    )

    result = await gateway.routing_decision("task-fallback-1")
    assert result is not None
    assert result["task_id"] == "task-fallback-1"
    assert result["selected_agent_id"] == "worker-1"
    assert result["rationale"] == "Event stream fallback"


# ==============================================================================
# 5. GetRoutingDecision Query & Presentation Endpoint Tests
# ==============================================================================


@pytest.mark.anyio
async def test_get_routing_decision_query() -> None:
    fake_reader = FakeStateReader(
        routing_decisions={
            "task-q-1": {
                "task_id": "task-q-1",
                "prompt_classification": "research",
                "selected_agent_id": "research-1",
                "rationale": "Query test rationale",
            }
        }
    )

    query = GetRoutingDecision(fake_reader)
    decision = await query.execute("task-q-1")

    assert decision is not None
    assert decision["task_id"] == "task-q-1"
    assert decision["selected_agent_id"] == "research-1"

    # Unknown task returns None
    assert await query.execute("unknown-task") is None

    # Empty task_id raises ValueError
    with pytest.raises(ValueError, match="task_id must not be empty"):
        await query.execute("   ")


@pytest.mark.anyio
async def test_task_routing_endpoint_success_and_not_found() -> None:
    fake_reader = FakeStateReader(
        routing_decisions={
            "task-found": {
                "task_id": "task-found",
                "prompt_classification": "research",
                "required_capabilities": ["research"],
                "candidates_evaluated": [],
                "selected_agent_id": "research-agent-1",
                "dispatch_target": "agent.tasks.research",
                "rationale": "Routed to research agent",
            }
        }
    )

    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        state_reader=fake_reader,
    )
    req = Request({"type": "http", "app": app})

    # Success case
    res = await task_routing("task-found", request=req)
    assert res["task_id"] == "task-found"
    assert res["selected_agent_id"] == "research-agent-1"
    assert res["dispatch_target"] == "agent.tasks.research"

    # 404 case
    with pytest.raises(HTTPException) as exc_info:
        await task_routing("task-missing", request=req)
    assert exc_info.value.status_code == 404
    assert "Routing decision for task 'task-missing' not found" in exc_info.value.detail
