from datetime import UTC, datetime, timedelta

import pytest
from agent_contracts import (
    AgentAvailabilityState,
    AgentCandidate,
    AgentExecutionState,
    LivenessPolicy,
)
from orchestrator.application import (
    FindCandidatesByCapability,
    QueryAgentsByCapability,
)
from orchestrator.domain import AgentCapability
from orchestrator.presentation.api import candidates as candidates_endpoint
from orchestrator.presentation.api import create_app
from starlette.requests import Request

from tests.test_application import FakeStateReader, FakeTaskQueue


@pytest.mark.anyio
async def test_matching_capability_and_ready_agent_is_returned() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-1",
                "agent_type": "worker",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": base.isoformat(),
                "providers": {"inference_provider": "stub"},
            }
        ]
    )
    query = FindCandidatesByCapability(reader)

    results = await query.execute(
        capability=AgentCapability.GENERAL,
        as_of=base + timedelta(seconds=5),
    )

    assert len(results) == 1
    candidate = results[0]
    assert candidate.agent_id == "worker-1"
    assert candidate.agent_type == "worker"
    assert candidate.capabilities == ["general"]
    assert candidate.execution_state == AgentExecutionState.IDLE
    assert candidate.availability.state == AgentAvailabilityState.READY
    assert candidate.availability.is_eligible is True
    assert candidate.providers == {"inference_provider": "stub"}


@pytest.mark.anyio
async def test_nonmatching_capability_is_excluded() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-1",
                "agent_type": "worker",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": base.isoformat(),
            }
        ]
    )
    query = FindCandidatesByCapability(reader)

    # Required capability is RESEARCH, but agent only advertises GENERAL
    results = await query.execute(
        capability=AgentCapability.RESEARCH,
        as_of=base + timedelta(seconds=5),
    )
    assert results == []


@pytest.mark.anyio
async def test_matching_but_busy_agent_is_excluded() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-busy",
                "agent_type": "worker",
                "capabilities": ["general"],
                "state": "working",  # Actively processing a task
                "updated_at": base.isoformat(),
            }
        ]
    )
    query = FindCandidatesByCapability(reader)

    results = await query.execute(
        capability=AgentCapability.GENERAL,
        as_of=base + timedelta(seconds=5),
    )
    # BUSY agents have is_eligible == False and must be excluded
    assert results == []


@pytest.mark.anyio
async def test_matching_but_stale_agent_is_excluded() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    policy = LivenessPolicy(stale_threshold_seconds=30.0, offline_threshold_seconds=90.0)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-lagged",
                "agent_type": "worker",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": base.isoformat(),
            }
        ]
    )
    query = FindCandidatesByCapability(reader, policy=policy)

    # 45 seconds elapsed exceeds stale threshold (30.0s)
    results = await query.execute(
        capability=AgentCapability.GENERAL,
        as_of=base + timedelta(seconds=45),
    )
    assert results == []


@pytest.mark.anyio
async def test_matching_but_offline_agent_is_excluded() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    policy = LivenessPolicy(stale_threshold_seconds=30.0, offline_threshold_seconds=90.0)
    reader = FakeStateReader(
        agents_data=[
            # 1. Timed out past offline threshold (100s)
            {
                "agent_id": "worker-dead",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": base.isoformat(),
            },
            # 2. In unrecovered ERROR execution state
            {
                "agent_id": "worker-error",
                "capabilities": ["general"],
                "state": "error",
                "updated_at": (base + timedelta(seconds=90)).isoformat(),
            },
        ]
    )
    query = FindCandidatesByCapability(reader, policy=policy)

    results = await query.execute(
        capability=AgentCapability.GENERAL,
        as_of=base + timedelta(seconds=100),
    )
    assert results == []


@pytest.mark.anyio
async def test_multiple_eligible_agents_advertising_same_capability_are_all_returned() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-1",
                "agent_type": "worker",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": base.isoformat(),
            },
            {
                "agent_id": "worker-2",
                "agent_type": "worker",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": base.isoformat(),
            },
            {
                "agent_id": "worker-3",
                "agent_type": "worker",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": base.isoformat(),
            },
        ]
    )
    query = FindCandidatesByCapability(reader)

    results = await query.execute(
        capability=AgentCapability.GENERAL,
        as_of=base + timedelta(seconds=5),
    )

    # Issue #4 must return all matching eligible candidates; selection belongs to Issue #5
    assert len(results) == 3
    candidate_ids = [c.agent_id for c in results]
    assert candidate_ids == ["worker-1", "worker-2", "worker-3"]


@pytest.mark.anyio
async def test_no_eligible_agents_returns_empty_candidate_list() -> None:
    # 1. Completely empty registry
    reader_empty = FakeStateReader(agents_data=[])
    query_empty = FindCandidatesByCapability(reader_empty)
    results_empty = await query_empty.execute(capability=AgentCapability.GENERAL)
    assert results_empty == []

    # 2. Registry with agents of other capabilities
    reader_other = FakeStateReader(
        agents_data=[
            {
                "agent_id": "research-1",
                "capabilities": ["research"],
                "state": "idle",
                "updated_at": datetime.now(UTC).isoformat(),
            }
        ]
    )
    query_other = FindCandidatesByCapability(reader_other)
    results_other = await query_other.execute(capability=AgentCapability.GENERAL)
    assert results_other == []


@pytest.mark.anyio
async def test_general_and_research_capabilities_handled_generically() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-gen",
                "agent_type": "worker",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": base.isoformat(),
            },
            {
                "agent_id": "research-1",
                "agent_type": "research-agent",
                "capabilities": ["research"],
                "state": "idle",
                "updated_at": base.isoformat(),
            },
            {
                "agent_id": "hybrid-1",
                "agent_type": "hybrid-agent",
                "capabilities": ["general", "research"],
                "state": "idle",
                "updated_at": base.isoformat(),
            },
        ]
    )
    query = FindCandidatesByCapability(reader)
    as_of = base + timedelta(seconds=5)

    # Query general (via enum and string)
    gen_candidates = await query.execute(AgentCapability.GENERAL, as_of=as_of)
    assert [c.agent_id for c in gen_candidates] == ["worker-gen", "hybrid-1"]

    gen_str_candidates = await query.execute("general", as_of=as_of)
    assert [c.agent_id for c in gen_str_candidates] == ["worker-gen", "hybrid-1"]

    # Query research (via enum and string)
    res_candidates = await query.execute(AgentCapability.RESEARCH, as_of=as_of)
    assert [c.agent_id for c in res_candidates] == ["research-1", "hybrid-1"]

    res_str_candidates = await query.execute("research", as_of=as_of)
    assert [c.agent_id for c in res_str_candidates] == ["research-1", "hybrid-1"]

    # Query custom capability
    custom_candidates = await query.execute("code-synthesis", as_of=as_of)
    assert custom_candidates == []


@pytest.mark.anyio
async def test_availability_evaluation_uses_deterministic_reference_time() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    policy = LivenessPolicy(stale_threshold_seconds=30.0, offline_threshold_seconds=90.0)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-timed",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": base.isoformat(),
            }
        ]
    )
    query = FindCandidatesByCapability(reader, policy=policy)

    # 1. Within fresh window (10s elapsed) -> READY -> candidate returned
    fresh_results = await query.execute(
        AgentCapability.GENERAL,
        as_of=base + timedelta(seconds=10),
    )
    assert len(fresh_results) == 1
    assert fresh_results[0].availability.state == AgentAvailabilityState.READY

    # 2. Within stale window (40s elapsed) -> STALE -> excluded
    stale_results = await query.execute(
        AgentCapability.GENERAL,
        as_of=base + timedelta(seconds=40),
    )
    assert stale_results == []

    # 3. Beyond offline threshold (100s elapsed) -> OFFLINE -> excluded
    offline_results = await query.execute(
        AgentCapability.GENERAL,
        as_of=base + timedelta(seconds=100),
    )
    assert offline_results == []


@pytest.mark.anyio
async def test_structured_candidate_metadata_and_serialization() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "research-node-1",
                "agent_type": "research-agent",
                "capabilities": ["research"],
                "state": "idle",
                "updated_at": base.isoformat(),
                "providers": {
                    "research_executor": "tavily",
                    "inference_provider": "cloudflare",
                },
            }
        ]
    )
    query = FindCandidatesByCapability(reader)

    results = await query.execute(
        AgentCapability.RESEARCH,
        as_of=base + timedelta(seconds=2),
    )

    assert len(results) == 1
    candidate = results[0]

    # Verify structured fields
    assert candidate.agent_id == "research-node-1"
    assert candidate.agent_type == "research-agent"
    assert candidate.capabilities == ["research"]
    assert candidate.execution_state == AgentExecutionState.IDLE
    assert candidate.availability.state == AgentAvailabilityState.READY
    assert candidate.providers == {
        "research_executor": "tavily",
        "inference_provider": "cloudflare",
    }

    # Serialization roundtrip
    c_dict = candidate.to_dict()
    assert c_dict["agent_id"] == "research-node-1"
    assert c_dict["capabilities"] == ["research"]
    assert c_dict["availability"]["state"] == "ready"
    assert c_dict["providers"]["research_executor"] == "tavily"

    restored_dict = AgentCandidate.from_dict(c_dict)
    assert restored_dict == candidate

    c_json = candidate.to_json()
    restored_json = AgentCandidate.from_json(c_json)
    assert restored_json == candidate


@pytest.mark.anyio
async def test_sensitive_metadata_is_not_exposed_in_candidate_providers() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-secure",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": base.isoformat(),
                "providers": {
                    "inference_provider": "cloudflare",
                    "api_key": "secret-token-value",
                    "auth_token": "token-xyz",
                    "account_password": "supersecretpassword",
                },
            }
        ]
    )
    query = FindCandidatesByCapability(reader)

    results = await query.execute(
        AgentCapability.GENERAL,
        as_of=base + timedelta(seconds=2),
    )

    assert len(results) == 1
    candidate = results[0]

    # Only safe provider keys are retained
    assert candidate.providers == {"inference_provider": "cloudflare"}
    assert "api_key" not in candidate.providers
    assert "auth_token" not in candidate.providers
    assert "account_password" not in candidate.providers


@pytest.mark.anyio
async def test_malformed_and_incomplete_registry_entries_handled_conservatively() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    reader = FakeStateReader(
        agents_data=[
            # Missing agent_id
            {"capabilities": ["general"], "state": "idle", "updated_at": base.isoformat()},
            # Empty agent_id
            {"agent_id": "   ", "capabilities": ["general"], "state": "idle", "updated_at": base.isoformat()},
            # Non-list capabilities
            {"agent_id": "bad-caps-1", "capabilities": 12345, "state": "idle", "updated_at": base.isoformat()},
            # Missing timestamp entirely (evaluated as OFFLINE)
            {"agent_id": "no-ts", "capabilities": ["general"], "state": "idle"},
            # Invalid timestamp string (evaluated as OFFLINE)
            {"agent_id": "bad-ts", "capabilities": ["general"], "state": "idle", "updated_at": "not-a-date"},
            # Non-dict entry in registry list
            "just-a-string-entry",  # type: ignore[list-item]
            # Valid entry to prove filtering isolates bad entries
            {
                "agent_id": "valid-worker",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": base.isoformat(),
            },
        ]
    )
    query = FindCandidatesByCapability(reader)

    results = await query.execute(
        AgentCapability.GENERAL,
        as_of=base + timedelta(seconds=5),
    )

    # Only the valid worker is returned, malformed entries are ignored safely
    assert len(results) == 1
    assert results[0].agent_id == "valid-worker"


@pytest.mark.anyio
async def test_invalid_query_parameters() -> None:
    reader = FakeStateReader()
    query = FindCandidatesByCapability(reader)

    with pytest.raises(ValueError, match="capability must not be empty"):
        await query.execute("")

    with pytest.raises(ValueError, match="capability must not be empty"):
        await query.execute("   ")

    with pytest.raises(TypeError, match="capability must be an AgentCapability or str"):
        await query.execute(123)  # type: ignore[arg-type]


def test_query_agents_by_capability_alias() -> None:
    assert QueryAgentsByCapability is FindCandidatesByCapability


@pytest.mark.anyio
async def test_api_candidates_endpoint() -> None:
    now = datetime.now(UTC)
    fake_reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "research-node-1",
                "agent_type": "research-agent",
                "capabilities": ["research"],
                "state": "idle",
                "updated_at": now.isoformat(),
                "providers": {"research_executor": "tavily"},
            }
        ]
    )
    app = create_app(
        task_queue=FakeTaskQueue(),
        state_reader=fake_reader,
    )
    req = Request({"type": "http", "app": app})

    res = await candidates_endpoint(capability="research", request=req)
    assert len(res) == 1
    assert res[0]["agent_id"] == "research-node-1"
    assert res[0]["capabilities"] == ["research"]
    assert res[0]["availability"]["state"] == "ready"
    assert res[0]["providers"] == {"research_executor": "tavily"}
