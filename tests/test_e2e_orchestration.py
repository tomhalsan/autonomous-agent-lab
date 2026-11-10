"""End-to-end integration and dynamic orchestration tests for Issue #10.

Validates the full dynamic orchestration lifecycle:
agent registration -> capability discovery -> availability evaluation ->
prompt routing -> candidate selection -> dynamic dispatch -> routing decision recording ->
unavailable capability handling -> runtime failover -> observability.
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    InferenceExecutionTelemetry,
)
from orchestrator.infrastructure.redis_adapter import RedisGateway
from orchestrator.presentation.api import create_app
from research_agent.infrastructure.redis_bus import RedisBus as ResearchRedisBus
from worker.infrastructure.redis_bus import RedisBus as WorkerRedisBus


class InMemoryRedis:
    """Deterministic, thread-safe in-memory async Redis simulation.

    Supports the exact subset of Redis data structures and operations used by
    RedisGateway and Agent RedisBus: sets, hashes, lists, strings, and streams.
    Requires no external services, network sockets, background threads, or sleeps.
    """

    def __init__(self) -> None:
        self._sets: dict[str, set[str]] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._lists: dict[str, list[str]] = {}
        self._strings: dict[str, str] = {}
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._counter = 0

    async def sadd(self, key: str, *members: str) -> int:
        s = self._sets.setdefault(key, set())
        added = len(set(members) - s)
        s.update(members)
        return added

    async def smembers(self, key: str) -> set[str]:
        return set(self._sets.get(key, set()))

    async def hset(
        self, key: str, mapping: dict[str, str] | None = None, **kwargs: str
    ) -> int:
        h = self._hashes.setdefault(key, {})
        m = dict(mapping or {})
        m.update(kwargs)
        h.update(m)
        return len(m)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    async def rpush(self, key: str, *values: str) -> int:
        l = self._lists.setdefault(key, [])
        l.extend(values)
        return len(l)

    async def blpop(
        self, keys: str | list[str], timeout: int = 0
    ) -> tuple[str, str] | None:
        key_list = [keys] if isinstance(keys, str) else list(keys)
        for k in key_list:
            l = self._lists.get(k, [])
            if l:
                return (k, l.pop(0))
        return None

    async def set(self, key: str, value: str) -> bool:
        self._strings[key] = str(value)
        return True

    async def get(self, key: str) -> str | None:
        return self._strings.get(key)

    async def xadd(
        self,
        key: str,
        fields: dict[str, str],
        maxlen: int | None = None,
        approximate: bool = False,
    ) -> str:
        self._counter += 1
        eid = f"{self._counter}-0"
        s = self._streams.setdefault(key, [])
        s.append((eid, dict(fields)))
        if maxlen is not None and len(s) > maxlen:
            self._streams[key] = s[-maxlen:]
        return eid

    async def xrevrange(
        self,
        key: str,
        max: str = "+",
        min: str = "-",
        count: int | None = None,
    ) -> list[tuple[str, dict[str, str]]]:
        s = list(reversed(self._streams.get(key, [])))
        return s[:count] if count is not None else s

    async def aclose(self) -> None:
        pass


def make_test_environment() -> tuple[Any, InMemoryRedis]:
    """Instantiate the production FastAPI orchestrator wired to an in-memory Redis store."""
    store = InMemoryRedis()
    gateway = RedisGateway("redis://fake:6379/0")
    gateway._client = store
    app = create_app(gateway=gateway, enforce_availability=True)
    return app, store


# ==============================================================================
# Scenario A: Runtime Dynamic Registration (Acceptance Criterion 1)
# ==============================================================================


@pytest.mark.anyio
async def test_e2e_runtime_registration_and_dynamic_routing() -> None:
    """Verifies a newly registered agent receives routed tasks without orchestrator reconfiguration."""
    app, store = make_test_environment()

    # Agent uses custom non-default dispatch queue
    target_queue = "agent.tasks.custom-worker-1"
    agent_bus = WorkerRedisBus("redis://fake:6379/0", queue_name=target_queue)
    agent_bus.client = store

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Step 1: Verify initially no candidates are discovered
        c_res = await client.get("/api/candidates", params={"capability": "general"})
        assert c_res.status_code == 200
        assert c_res.json() == []

        # Step 2: Agent dynamically registers at runtime
        metadata = AgentCapabilityMetadata(
            agent_id="worker-dynamic-1",
            agent_type="worker",
            capabilities=["general"],
            dispatch_target=target_queue,
            providers={"inference_provider": "stub"},
            version=1,
        )
        await agent_bus.register(metadata)
        await agent_bus.publish(
            AgentEvent.create("worker-dynamic-1", "idle", "Ready for tasks")
        )

        # Step 3: Candidate discovery immediately sees the newly registered agent
        c_res2 = await client.get("/api/candidates", params={"capability": "general"})
        assert c_res2.status_code == 200
        candidates = c_res2.json()
        assert len(candidates) == 1
        assert candidates[0]["agent_id"] == "worker-dynamic-1"
        assert candidates[0]["dispatch_target"] == target_queue
        assert candidates[0]["availability"]["is_eligible"] is True

        # Step 4: Submit a task requiring general capability via presentation API
        t_res = await client.post(
            "/api/tasks", json={"prompt": "Run database backup and health check"}
        )
        assert t_res.status_code == 202
        task_data = t_res.json()
        task_id = task_data["task_id"]
        assert task_data["status"] == "accepted"

        # Step 5: Agent dequeues the task from its advertised custom queue
        dispatched_task = await agent_bus.next_task()
        assert dispatched_task.task_id == task_id
        assert dispatched_task.prompt == "Run database backup and health check"

        # Step 6: Query recorded routing decision
        r_res = await client.get(f"/api/tasks/{task_id}/routing")
        assert r_res.status_code == 200
        decision = r_res.json()
        assert decision["task_id"] == task_id
        assert decision["prompt_classification"] == "general"
        assert decision["required_capabilities"] == ["general"]
        assert decision["selected_agent_id"] == "worker-dynamic-1"
        assert decision["dispatch_target"] == target_queue
        assert "worker-dynamic-1" in decision["rationale"]


@pytest.mark.anyio
async def test_e2e_specialized_research_agent_dynamic_routing() -> None:
    """Verifies prompt classification dynamically routes research tasks to specialized research agents."""
    app, store = make_test_environment()

    worker_queue = "agent.tasks.general-worker"
    worker_bus = WorkerRedisBus("redis://fake:6379/0", queue_name=worker_queue)
    worker_bus.client = store

    research_queue = "agent.tasks.deep-research"
    research_bus = ResearchRedisBus("redis://fake:6379/0", queue_name=research_queue)
    research_bus.client = store

    # Register both agents
    await worker_bus.register(
        AgentCapabilityMetadata(
            agent_id="worker-1",
            capabilities=["general"],
            dispatch_target=worker_queue,
        )
    )
    await worker_bus.publish(AgentEvent.create("worker-1", "idle", "Ready"))

    await research_bus.register(
        AgentCapabilityMetadata(
            agent_id="research-agent-1",
            capabilities=["research"],
            dispatch_target=research_queue,
        )
    )
    await research_bus.publish(AgentEvent.create("research-agent-1", "idle", "Ready"))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Submit a research query with clear research intent
        # Phase B requires external retrieval/freshness/source evidence rather than weak verbs alone
        res_task = await client.post(
            "/api/tasks",
            json={"prompt": "Research the latest consensus protocols in distributed state machines"},
        )
        assert res_task.status_code == 202
        research_task_id = res_task.json()["task_id"]

        # Dequeued specifically by the research agent on its custom queue
        task = await research_bus.next_task()
        assert task.task_id == research_task_id
        assert "consensus protocols" in task.prompt

        # Verify routing decision recorded research capability and selected target
        r_res = await client.get(f"/api/tasks/{research_task_id}/routing")
        assert r_res.status_code == 200
        decision = r_res.json()
        assert decision["prompt_classification"] == "research_and_investigation"
        assert decision["required_capabilities"] == ["research"]
        assert decision["selected_agent_id"] == "research-agent-1"
        assert decision["dispatch_target"] == research_queue


# ==============================================================================
# Scenario B: Multiple Candidates & Deterministic Selection
# ==============================================================================


@pytest.mark.anyio
async def test_e2e_multiple_candidates_evaluated_and_selected_deterministically() -> None:
    """Verifies when multiple eligible agents exist, all are evaluated and one is deterministically selected."""
    app, store = make_test_environment()

    # Register three eligible agents with distinct alphabetical IDs
    agents_info = [
        ("worker-charlie", "agent.tasks.charlie"),
        ("worker-alpha", "agent.tasks.alpha"),
        ("worker-bravo", "agent.tasks.bravo"),
    ]

    for agent_id, target in agents_info:
        bus = WorkerRedisBus("redis://fake:6379/0", queue_name=target)
        bus.client = store
        await bus.register(
            AgentCapabilityMetadata(
                agent_id=agent_id,
                capabilities=["general"],
                dispatch_target=target,
            )
        )
        await bus.publish(AgentEvent.create(agent_id, "idle", "Ready"))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Submit task
        res = await client.post("/api/tasks", json={"prompt": "Process batch transaction log"})
        assert res.status_code == 202
        task_id = res.json()["task_id"]

        # Query routing decision
        r_res = await client.get(f"/api/tasks/{task_id}/routing")
        assert r_res.status_code == 200
        decision = r_res.json()

        # All 3 candidates must be evaluated in the decision record
        evaluated = decision["candidates_evaluated"]
        assert len(evaluated) == 3
        evaluated_ids = {c["agent_id"] for c in evaluated}
        assert evaluated_ids == {"worker-alpha", "worker-bravo", "worker-charlie"}
        for c in evaluated:
            assert c["is_eligible"] is True
            assert c["availability_state"] == "ready"

        # Deterministic selection policy selects worker-alpha (lexicographical leader)
        assert decision["selected_agent_id"] == "worker-alpha"
        assert decision["dispatch_target"] == "agent.tasks.alpha"

        # Verify task is on alpha's queue, and not on bravo or charlie
        alpha_bus = WorkerRedisBus("redis://fake:6379/0", queue_name="agent.tasks.alpha")
        alpha_bus.client = store
        dispatched_task = await alpha_bus.next_task()
        assert dispatched_task.task_id == task_id

        bravo_bus = WorkerRedisBus("redis://fake:6379/0", queue_name="agent.tasks.bravo")
        bravo_bus.client = store
        bravo_item = await store.blpop("agent.tasks.bravo", timeout=0)
        assert bravo_item is None


# ==============================================================================
# Scenario C: Agent Unavailable / Unhealthy / Stale (Acceptance Criterion 2)
# ==============================================================================


@pytest.mark.anyio
async def test_e2e_stale_agent_rejected_with_graceful_unavailable_error() -> None:
    """Verifies that an agent with an expired heartbeat is marked ineligible, rejecting the task with 503."""
    app, store = make_test_environment()

    # Register an agent with a stale heartbeat (120 seconds in the past; threshold is 30s)
    stale_time = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
    await store.sadd("agent.ids", "worker-stale")
    await store.hset(
        "agent.state:worker-stale",
        mapping={
            "agent_id": "worker-stale",
            "capabilities": json.dumps(["general"]),
            "dispatch_target": "agent.tasks.stale",
            "state": "idle",
            "updated_at": stale_time,
        },
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Candidate endpoint reflects stale status
        c_res = await client.get("/api/candidates", params={"capability": "general"})
        assert c_res.status_code == 200
        assert c_res.json() == []

        # Task submission must gracefully fail with HTTP 503 capability_unavailable
        t_res = await client.post("/api/tasks", json={"prompt": "Run database migration"})
        assert t_res.status_code == 503
        detail = t_res.json()["detail"]
        assert detail["error"] == "capability_unavailable"
        assert detail["capability"] == "general"
        rejection_task_id = detail["task_id"]

        # The rejection decision must be recorded and queryable
        r_res = await client.get(f"/api/tasks/{rejection_task_id}/routing")
        assert r_res.status_code == 200
        decision = r_res.json()
        assert decision["task_id"] == rejection_task_id
        assert decision["selected_agent_id"] is None
        assert decision["dispatch_target"] is None
        assert "No eligible agent" in decision["rationale"]

        # An error event must have been published to the event stream
        e_res = await client.get("/api/events")
        assert e_res.status_code == 200
        events = e_res.json()
        error_events = [e for e in events if e.get("task_id") == rejection_task_id]
        assert len(error_events) >= 1
        assert error_events[0]["state"] == "error"


@pytest.mark.anyio
async def test_e2e_failover_to_secondary_candidate_when_primary_fails() -> None:
    """Verifies that when the primary candidate is offline or stale, routing automatically selects the secondary."""
    app, store = make_test_environment()

    # Register worker-alpha (offline) and worker-beta (idle/ready)
    now_str = datetime.now(UTC).isoformat()
    await store.sadd("agent.ids", "worker-alpha", "worker-beta")

    # Alpha is marked offline
    await store.hset(
        "agent.state:worker-alpha",
        mapping={
            "agent_id": "worker-alpha",
            "capabilities": json.dumps(["general"]),
            "dispatch_target": "agent.tasks.alpha",
            "state": "offline",
            "updated_at": now_str,
        },
    )

    # Beta is marked idle and ready
    await store.hset(
        "agent.state:worker-beta",
        mapping={
            "agent_id": "worker-beta",
            "capabilities": json.dumps(["general"]),
            "dispatch_target": "agent.tasks.beta",
            "state": "idle",
            "updated_at": now_str,
        },
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        t_res = await client.post("/api/tasks", json={"prompt": "Execute cluster failover check"})
        assert t_res.status_code == 202
        task_id = t_res.json()["task_id"]

        r_res = await client.get(f"/api/tasks/{task_id}/routing")
        assert r_res.status_code == 200
        decision = r_res.json()

        # worker-beta selected because worker-alpha is offline
        assert decision["selected_agent_id"] == "worker-beta"
        assert decision["dispatch_target"] == "agent.tasks.beta"

        # Verify task is enqueued to beta's queue
        beta_item = await store.blpop("agent.tasks.beta", timeout=0)
        assert beta_item is not None
        assert task_id in beta_item[1]


# ==============================================================================
# Scenario D: Runtime Change without Orchestrator Reconfiguration
# ==============================================================================


@pytest.mark.anyio
async def test_e2e_runtime_lifecycle_transitions_without_orchestrator_reconfiguration() -> None:
    """Verifies that the orchestrator adapts dynamically as an agent transitions states over time."""
    app, store = make_test_environment()
    target_queue = "agent.tasks.lifecycle-worker"
    bus = WorkerRedisBus("redis://fake:6379/0", queue_name=target_queue)
    bus.client = store

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Phase 1: No agent -> 503
        r1 = await client.post("/api/tasks", json={"prompt": "Process log files and generate statistics"})
        assert r1.status_code == 503

        # Phase 2: Agent registers and goes idle -> 202 accepted
        await bus.register(
            AgentCapabilityMetadata(
                agent_id="worker-lifecycle",
                capabilities=["general"],
                dispatch_target=target_queue,
            )
        )
        await bus.publish(AgentEvent.create("worker-lifecycle", "idle", "Ready"))

        r2 = await client.post("/api/tasks", json={"prompt": "Process log files and generate statistics"})
        assert r2.status_code == 202
        t2_id = r2.json()["task_id"]
        t2 = await bus.next_task()
        assert t2.task_id == t2_id

        # Phase 3: Agent transitions to working (busy) -> 503
        await bus.publish(AgentEvent.create("worker-lifecycle", "working", "Busy processing"))
        r3 = await client.post("/api/tasks", json={"prompt": "Process log files and generate statistics"})
        assert r3.status_code == 503

        # Phase 4: Agent transitions back to idle -> 202 accepted
        await bus.publish(AgentEvent.create("worker-lifecycle", "idle", "Finished and idle"))
        r4 = await client.post("/api/tasks", json={"prompt": "Process log files and generate statistics"})
        assert r4.status_code == 202
        t4_id = r4.json()["task_id"]
        t4 = await bus.next_task()
        assert t4.task_id == t4_id


# ==============================================================================
# Scenario E: Observability & Diagnostics
# ==============================================================================


@pytest.mark.anyio
async def test_e2e_observability_endpoints_and_secret_redaction() -> None:
    """Verifies that agents, candidates, routing decisions, and events are fully queryable and sanitize secrets."""
    app, store = make_test_environment()
    bus = WorkerRedisBus("redis://fake:6379/0", queue_name="agent.tasks.obs")
    bus.client = store

    # 0. Verify contracts actively reject sensitive credentials at creation time
    with pytest.raises(ValueError, match="Sensitive key 'api_key' must not be exposed"):
        AgentCapabilityMetadata(
            agent_id="worker-obs-1",
            agent_type="worker",
            capabilities=["general"],
            dispatch_target="agent.tasks.obs",
            providers={
                "api_key": "disallowed_secret_token",
            },
        )

    # Register legitimate agent with non-sensitive provider metadata
    await bus.register(
        AgentCapabilityMetadata(
            agent_id="worker-obs-1",
            agent_type="worker",
            capabilities=["general"],
            dispatch_target="agent.tasks.obs",
            providers={
                "inference_provider": "stub",
                "model_family": "open-weights",
            },
        )
    )
    # Directly inject an un-sanitized field into the raw Redis hash to verify defense-in-depth sanitization
    await store.hset("agent.state:worker-obs-1", mapping={"raw_secret_token": "leak_attempt"})
    await bus.publish(AgentEvent.create("worker-obs-1", "idle", "Observability test ready"))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. GET /api/agents
        agents_res = await client.get("/api/agents")
        assert agents_res.status_code == 200
        agents_list = agents_res.json()
        assert any(a["agent_id"] == "worker-obs-1" for a in agents_list)

        # 2. GET /api/candidates
        cand_res = await client.get("/api/candidates", params={"capability": "general"})
        assert cand_res.status_code == 200
        candidates = cand_res.json()
        assert len(candidates) == 1
        c = candidates[0]
        assert c["agent_id"] == "worker-obs-1"
        assert c["availability"]["is_eligible"] is True
        assert c["providers"]["inference_provider"] == "stub"
        assert "raw_secret_token" not in c["providers"]

        # 3. Submit task and check GET /api/tasks/{task_id}/routing
        t_res = await client.post("/api/tasks", json={"prompt": "Verify system metrics"})
        assert t_res.status_code == 202
        task_id = t_res.json()["task_id"]

        r_res = await client.get(f"/api/tasks/{task_id}/routing")
        assert r_res.status_code == 200
        decision = r_res.json()
        assert decision["task_id"] == task_id
        assert decision["prompt_classification"] == "general"
        assert decision["selected_agent_id"] == "worker-obs-1"
        assert decision["dispatch_target"] == "agent.tasks.obs"

        # 4. GET /api/events
        e_res = await client.get("/api/events")
        assert e_res.status_code == 200
        events = e_res.json()
        assert any(e.get("agent_id") == "worker-obs-1" for e in events)


@pytest.mark.anyio
async def test_get_api_events_exposes_inference_telemetry() -> None:
    """Verifies that events with inference_telemetry flow through RedisBus to GET /api/events."""
    app, store = make_test_environment()
    agent_bus = ResearchRedisBus("redis://fake:6379/0", queue_name="agent.tasks.research")
    agent_bus.client = store

    telemetry = InferenceExecutionTelemetry(
        provider_name="cloudflare",
        attempt_index=0,
        latency_ms=88.4,
        status="succeeded",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        finish_reason="stop",
        model_name="@cf/meta/llama-3.1-8b-instruct",
    )
    event = AgentEvent.create(
        agent_id="research-agent-e2e",
        state="working",
        detail="Inference attempt 0 succeeded (cloudflare)",
        task_id="task-telemetry-query-1",
        inference_telemetry=telemetry,
    )
    await agent_bus.publish(event)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        res = await client.get("/api/events")
        assert res.status_code == 200
        events = res.json()
        matching_events = [e for e in events if e.get("task_id") == "task-telemetry-query-1"]
        assert len(matching_events) == 1
        ev_data = matching_events[0]
        assert ev_data["agent_id"] == "research-agent-e2e"
        assert ev_data["state"] == "working"
        assert "inference_telemetry" in ev_data
        telem_data = (
            json.loads(ev_data["inference_telemetry"])
            if isinstance(ev_data["inference_telemetry"], str)
            else ev_data["inference_telemetry"]
        )
        assert telem_data["provider_name"] == "cloudflare"
        assert telem_data["attempt_index"] == 0
        assert telem_data["latency_ms"] == 88.4
        assert telem_data["status"] == "succeeded"
        assert telem_data["input_tokens"] == 100
        assert telem_data["output_tokens"] == 50
        assert telem_data["total_tokens"] == 150
        assert telem_data["model_name"] == "@cf/meta/llama-3.1-8b-instruct"
