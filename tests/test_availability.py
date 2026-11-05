import math
from datetime import UTC, datetime, timedelta

import pytest
from agent_contracts import (
    AgentAvailability,
    AgentAvailabilityState,
    AgentEvent,
    AgentExecutionState,
    AgentHeartbeat,
    LivenessPolicy,
    evaluate_agent_record,
    evaluate_availability,
    parse_timestamp,
)
from orchestrator.domain import (
    AgentAvailability as DomainAgentAvailability,
)
from orchestrator.domain import (
    AgentAvailabilityState as DomainAgentAvailabilityState,
)
from orchestrator.domain import (
    AgentExecutionState as DomainAgentExecutionState,
)
from orchestrator.domain import (
    AgentHeartbeat as DomainAgentHeartbeat,
)
from orchestrator.domain import (
    LivenessPolicy as DomainLivenessPolicy,
)
from orchestrator.domain import (
    evaluate_agent_record as domain_evaluate_agent_record,
)
from orchestrator.domain import (
    evaluate_availability as domain_evaluate_availability,
)
from orchestrator.domain import (
    parse_timestamp as domain_parse_timestamp,
)


def test_availability_states_defined() -> None:
    assert AgentAvailabilityState.READY == "ready"
    assert AgentAvailabilityState.BUSY == "busy"
    assert AgentAvailabilityState.STALE == "stale"
    assert AgentAvailabilityState.OFFLINE == "offline"

    assert AgentAvailabilityState.READY.is_eligible is True
    assert AgentAvailabilityState.BUSY.is_eligible is False
    assert AgentAvailabilityState.STALE.is_eligible is False
    assert AgentAvailabilityState.OFFLINE.is_eligible is False

    assert AgentAvailabilityState.READY.is_reachable is True
    assert AgentAvailabilityState.BUSY.is_reachable is True
    assert AgentAvailabilityState.STALE.is_reachable is False
    assert AgentAvailabilityState.OFFLINE.is_reachable is False


def test_execution_states_defined() -> None:
    assert AgentExecutionState.IDLE == "idle"
    assert AgentExecutionState.WORKING == "working"
    assert AgentExecutionState.ERROR == "error"
    assert AgentExecutionState.UNKNOWN == "unknown"

    assert AgentExecutionState.from_str("idle") == AgentExecutionState.IDLE
    assert AgentExecutionState.from_str("  WORKING  ") == AgentExecutionState.WORKING
    assert AgentExecutionState.from_str("error") == AgentExecutionState.ERROR
    assert AgentExecutionState.from_str("unexpected_state") == AgentExecutionState.UNKNOWN

    with pytest.raises(TypeError):
        AgentExecutionState.from_str(123)  # type: ignore[arg-type]


def test_liveness_policy_defaults_and_validation() -> None:
    policy = LivenessPolicy()
    assert policy.stale_threshold_seconds == 30.0
    assert policy.offline_threshold_seconds == 90.0
    assert policy.clock_skew_tolerance_seconds == 5.0

    custom = LivenessPolicy(
        stale_threshold_seconds=10.0,
        offline_threshold_seconds=25.0,
        clock_skew_tolerance_seconds=2.0,
    )
    assert custom.stale_threshold_seconds == 10.0
    assert custom.offline_threshold_seconds == 25.0

    with pytest.raises(ValueError, match="stale_threshold_seconds must be positive"):
        LivenessPolicy(stale_threshold_seconds=0.0)

    with pytest.raises(ValueError, match="stale_threshold_seconds must be positive"):
        LivenessPolicy(stale_threshold_seconds=-5.0)

    with pytest.raises(ValueError, match="offline_threshold_seconds .* must be strictly greater"):
        LivenessPolicy(stale_threshold_seconds=30.0, offline_threshold_seconds=30.0)

    with pytest.raises(ValueError, match="offline_threshold_seconds .* must be strictly greater"):
        LivenessPolicy(stale_threshold_seconds=30.0, offline_threshold_seconds=20.0)

    with pytest.raises(ValueError, match="clock_skew_tolerance_seconds must not be negative"):
        LivenessPolicy(clock_skew_tolerance_seconds=-1.0)

    with pytest.raises(ValueError, match="must be a finite number"):
        LivenessPolicy(stale_threshold_seconds=float("nan"))

    with pytest.raises(ValueError, match="must be a finite number"):
        LivenessPolicy(offline_threshold_seconds=float("inf"))


def test_deterministic_fresh_ready_evaluation() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    last_seen = base
    as_of = base + timedelta(seconds=12)

    availability = evaluate_availability(
        last_seen_at=last_seen,
        execution_state=AgentExecutionState.IDLE,
        as_of=as_of,
        agent_id="worker-1",
    )

    assert availability.state == AgentAvailabilityState.READY
    assert availability.execution_state == AgentExecutionState.IDLE
    assert availability.is_eligible is True
    assert availability.is_reachable is True
    assert availability.is_stale is False
    assert availability.is_offline is False
    assert math.isclose(availability.heartbeat_age_seconds, 12.0)
    assert availability.agent_id == "worker-1"
    assert "fresh heartbeat" in availability.reason


def test_deterministic_busy_evaluation() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    last_seen = base
    as_of = base + timedelta(seconds=15)

    availability = evaluate_availability(
        last_seen_at=last_seen.isoformat(),
        execution_state="working",
        as_of=as_of.isoformat(),
        agent_id="research-1",
    )

    assert availability.state == AgentAvailabilityState.BUSY
    assert availability.execution_state == AgentExecutionState.WORKING
    assert availability.is_eligible is False
    assert availability.is_reachable is True
    assert availability.is_stale is False
    assert availability.is_offline is False
    assert math.isclose(availability.heartbeat_age_seconds, 15.0)
    assert availability.agent_id == "research-1"
    assert "actively working" in availability.reason


def test_stale_heartbeat_detection() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    policy = LivenessPolicy(stale_threshold_seconds=30.0, offline_threshold_seconds=90.0)

    # 45 seconds elapsed: past stale threshold (30.0s) but within offline threshold (90.0s)
    as_of = base + timedelta(seconds=45)

    # Idle agent with stale heartbeat
    idle_av = policy.evaluate(
        last_seen_at=base,
        execution_state=AgentExecutionState.IDLE,
        as_of=as_of,
        agent_id="worker-stale",
    )
    assert idle_av.state == AgentAvailabilityState.STALE
    assert idle_av.execution_state == AgentExecutionState.IDLE
    assert idle_av.is_eligible is False
    assert idle_av.is_reachable is False
    assert idle_av.is_stale is True
    assert idle_av.is_offline is False
    assert math.isclose(idle_av.heartbeat_age_seconds, 45.0)
    assert "exceeds stale threshold" in idle_av.reason

    # Working agent with stale heartbeat preserves execution_state=WORKING
    working_av = policy.evaluate(
        last_seen_at=base,
        execution_state=AgentExecutionState.WORKING,
        as_of=as_of,
        agent_id="research-stale",
    )
    assert working_av.state == AgentAvailabilityState.STALE
    assert working_av.execution_state == AgentExecutionState.WORKING
    assert working_av.is_eligible is False
    assert working_av.is_reachable is False
    assert working_av.is_stale is True


def test_offline_determination_by_timeout() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    policy = LivenessPolicy(stale_threshold_seconds=30.0, offline_threshold_seconds=90.0)

    # 95 seconds elapsed: past offline threshold
    as_of = base + timedelta(seconds=95)

    availability = policy.evaluate(
        last_seen_at=base,
        execution_state=AgentExecutionState.IDLE,
        as_of=as_of,
        agent_id="worker-offline",
    )

    assert availability.state == AgentAvailabilityState.OFFLINE
    assert availability.execution_state == AgentExecutionState.IDLE
    assert availability.is_eligible is False
    assert availability.is_reachable is False
    assert availability.is_stale is False
    assert availability.is_offline is True
    assert math.isclose(availability.heartbeat_age_seconds, 95.0)
    assert "exceeds offline threshold" in availability.reason


def test_fresh_agent_with_error_execution_state() -> None:
    """A fresh heartbeat means the agent process is reachable even if in ERROR execution state."""
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    # Heartbeat is completely fresh (3 seconds old)
    as_of_fresh = base + timedelta(seconds=3)

    availability = evaluate_availability(
        last_seen_at=base,
        execution_state=AgentExecutionState.ERROR,
        as_of=as_of_fresh,
        agent_id="worker-error",
    )

    # Truthful reachability: process is alive and communicating
    # Conservative dispatch eligibility: cannot be assigned new tasks while in error
    assert availability.state == AgentAvailabilityState.BUSY
    assert availability.execution_state == AgentExecutionState.ERROR
    assert availability.is_eligible is False
    assert availability.is_reachable is True
    assert availability.is_offline is False
    assert "error execution state" in availability.reason

    # If the error agent stops sending heartbeats and liveness expires (e.g. 95s):
    as_of_expired = base + timedelta(seconds=95)
    av_expired = evaluate_availability(
        last_seen_at=base,
        execution_state=AgentExecutionState.ERROR,
        as_of=as_of_expired,
        agent_id="worker-error",
    )
    assert av_expired.state == AgentAvailabilityState.OFFLINE
    assert av_expired.execution_state == AgentExecutionState.ERROR
    assert av_expired.is_reachable is False
    assert av_expired.is_offline is True


def test_offline_determination_by_explicit_signal() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    as_of = base + timedelta(seconds=2)

    availability = evaluate_availability(
        last_seen_at=base,
        execution_state=AgentExecutionState.IDLE,
        as_of=as_of,
        explicit_offline=True,
    )

    assert availability.state == AgentAvailabilityState.OFFLINE
    assert availability.is_eligible is False
    assert availability.is_reachable is False
    assert "explicitly reported offline" in availability.reason


def test_fresh_agent_with_unknown_execution_state() -> None:
    """A fresh heartbeat with unknown operational state is reachable but ineligible for dispatch."""
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    as_of = base + timedelta(seconds=2)

    availability = evaluate_availability(
        last_seen_at=base,
        execution_state="corrupted_state",
        as_of=as_of,
    )

    assert availability.state == AgentAvailabilityState.BUSY
    assert availability.execution_state == AgentExecutionState.UNKNOWN
    assert availability.is_eligible is False
    assert availability.is_reachable is True
    assert availability.is_offline is False
    assert "unknown execution state" in availability.reason


def test_threshold_boundary_conditions() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    policy = LivenessPolicy(stale_threshold_seconds=30.0, offline_threshold_seconds=90.0)

    # 1. Exactly 0 seconds: fresh
    av_0 = policy.evaluate(base, AgentExecutionState.IDLE, base)
    assert av_0.state == AgentAvailabilityState.READY
    assert av_0.heartbeat_age_seconds == 0.0

    # 2. Exactly at stale threshold (30.0s): still fresh
    av_30 = policy.evaluate(base, AgentExecutionState.IDLE, base + timedelta(seconds=30.0))
    assert av_30.state == AgentAvailabilityState.READY
    assert math.isclose(av_30.heartbeat_age_seconds, 30.0)

    # 3. Just beyond stale threshold (30.001s): stale
    av_30_eps = policy.evaluate(base, AgentExecutionState.IDLE, base + timedelta(seconds=30.001))
    assert av_30_eps.state == AgentAvailabilityState.STALE

    # 4. Exactly at offline threshold (90.0s): still stale
    av_90 = policy.evaluate(base, AgentExecutionState.IDLE, base + timedelta(seconds=90.0))
    assert av_90.state == AgentAvailabilityState.STALE
    assert math.isclose(av_90.heartbeat_age_seconds, 90.0)

    # 5. Just beyond offline threshold (90.001s): offline
    av_90_eps = policy.evaluate(base, AgentExecutionState.IDLE, base + timedelta(seconds=90.001))
    assert av_90_eps.state == AgentAvailabilityState.OFFLINE


def test_clock_skew_handling() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    policy = LivenessPolicy(clock_skew_tolerance_seconds=5.0)

    # last_seen is 3 seconds in the future relative to as_of (within 5.0s tolerance)
    last_seen_future_ok = base + timedelta(seconds=3)
    av_skew_ok = policy.evaluate(last_seen_future_ok, AgentExecutionState.IDLE, base)
    assert av_skew_ok.state == AgentAvailabilityState.READY
    assert av_skew_ok.heartbeat_age_seconds == 0.0

    # exactly at tolerance limit (5.0s future): clamped to 0.0s
    last_seen_limit = base + timedelta(seconds=5.0)
    av_limit = policy.evaluate(last_seen_limit, AgentExecutionState.IDLE, base)
    assert av_limit.heartbeat_age_seconds == 0.0

    # beyond tolerance limit (5.01s future): raises ValueError
    last_seen_too_far = base + timedelta(seconds=5.01)
    with pytest.raises(ValueError, match="in the future relative to .* beyond tolerance"):
        policy.evaluate(last_seen_too_far, AgentExecutionState.IDLE, base)


def test_parse_timestamp_valid_and_invalid() -> None:
    # ISO strings with Z
    dt1 = parse_timestamp("2026-11-05T13:00:00Z")
    assert dt1.tzinfo == UTC
    assert dt1.year == 2026

    # ISO strings with explicit offset
    dt2 = parse_timestamp("2026-11-05T15:00:00+02:00")
    assert dt2.tzinfo == UTC
    assert dt2.hour == 13  # Normalized to UTC

    # Datetime object (naive normalized to UTC)
    naive = datetime(2026, 11, 5, 13, 0, 0)
    dt3 = parse_timestamp(naive)
    assert dt3.tzinfo == UTC

    # Datetime object (already UTC)
    aware = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    dt4 = parse_timestamp(aware)
    assert dt4 == aware

    # Invalid string formats
    with pytest.raises(ValueError, match="Timestamp string must not be empty"):
        parse_timestamp("")

    with pytest.raises(ValueError, match="Timestamp string must not be empty"):
        parse_timestamp("   ")

    with pytest.raises(ValueError, match="Invalid ISO timestamp format"):
        parse_timestamp("not-a-timestamp")

    with pytest.raises(TypeError, match="Expected datetime or str"):
        parse_timestamp(12345)  # type: ignore[arg-type]


def test_execution_state_and_availability_state_separation() -> None:
    """Execution state answers what the agent is doing.

    Availability state answers whether it is eligible/reachable for orchestration.
    """
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)

    # 1. Agent is actively WORKING and fresh -> Availability is BUSY
    av_busy = evaluate_availability(
        base,
        execution_state=AgentExecutionState.WORKING,
        as_of=base + timedelta(seconds=5),
    )
    assert av_busy.execution_state == AgentExecutionState.WORKING
    assert av_busy.state == AgentAvailabilityState.BUSY
    assert av_busy.is_eligible is False
    assert av_busy.is_reachable is True

    # 2. Agent is WORKING but heartbeat decayed to STALE -> Availability is STALE, execution is WORKING
    av_stale_working = evaluate_availability(
        base,
        execution_state=AgentExecutionState.WORKING,
        as_of=base + timedelta(seconds=50),
    )
    assert av_stale_working.execution_state == AgentExecutionState.WORKING
    assert av_stale_working.state == AgentAvailabilityState.STALE
    assert av_stale_working.is_eligible is False
    assert av_stale_working.is_reachable is False

    # 3. Agent is IDLE and fresh -> Availability is READY
    av_ready = evaluate_availability(
        base,
        execution_state=AgentExecutionState.IDLE,
        as_of=base + timedelta(seconds=5),
    )
    assert av_ready.execution_state == AgentExecutionState.IDLE
    assert av_ready.state == AgentAvailabilityState.READY
    assert av_ready.is_eligible is True

    # 4. Agent is in ERROR state with fresh heartbeat -> Availability is BUSY (reachable, but ineligible)
    av_error_fresh = evaluate_availability(
        base,
        execution_state=AgentExecutionState.ERROR,
        as_of=base + timedelta(seconds=1),
    )
    assert av_error_fresh.execution_state == AgentExecutionState.ERROR
    assert av_error_fresh.state == AgentAvailabilityState.BUSY
    assert av_error_fresh.is_eligible is False
    assert av_error_fresh.is_reachable is True

    # When that error agent's heartbeat expires, availability transitions to OFFLINE
    av_error_expired = evaluate_availability(
        base,
        execution_state=AgentExecutionState.ERROR,
        as_of=base + timedelta(seconds=120),
    )
    assert av_error_expired.execution_state == AgentExecutionState.ERROR
    assert av_error_expired.state == AgentAvailabilityState.OFFLINE
    assert av_error_expired.is_eligible is False
    assert av_error_expired.is_reachable is False

    # 5. Existing AgentEvent state remains strictly execution-oriented
    event = AgentEvent.create(agent_id="test-agent", state="working", detail="Processing item")
    assert event.state == "working"


def test_agent_availability_dataclass_validation() -> None:
    with pytest.raises(ValueError, match="heartbeat_age_seconds must not be negative"):
        AgentAvailability(
            state=AgentAvailabilityState.READY,
            execution_state=AgentExecutionState.IDLE,
            heartbeat_age_seconds=-1.0,
            last_seen_at="2026-11-05T13:00:00Z",
            as_of="2026-11-05T13:00:05Z",
            reason="test",
        )

    with pytest.raises(ValueError, match="reason must not be empty"):
        AgentAvailability(
            state=AgentAvailabilityState.READY,
            execution_state=AgentExecutionState.IDLE,
            heartbeat_age_seconds=1.0,
            last_seen_at="2026-11-05T13:00:00Z",
            as_of="2026-11-05T13:00:05Z",
            reason="   ",
        )

    with pytest.raises(TypeError, match="state must be an AgentAvailabilityState"):
        AgentAvailability(
            state=123,  # type: ignore[arg-type]
            execution_state=AgentExecutionState.IDLE,
            heartbeat_age_seconds=1.0,
            last_seen_at="2026-11-05T13:00:00Z",
            as_of="2026-11-05T13:00:05Z",
            reason="test",
        )


def test_agent_availability_serialization_roundtrip() -> None:
    original = AgentAvailability(
        state=AgentAvailabilityState.READY,
        execution_state=AgentExecutionState.IDLE,
        heartbeat_age_seconds=4.5123,
        last_seen_at="2026-11-05T13:00:00+00:00",
        as_of="2026-11-05T13:00:04.512+00:00",
        reason="Agent is idle with fresh heartbeat",
        agent_id="worker-node-1",
        version=1,
    )

    as_dict = original.to_dict()
    assert as_dict["state"] == "ready"
    assert as_dict["execution_state"] == "idle"
    assert as_dict["heartbeat_age_seconds"] == 4.512
    assert as_dict["is_eligible"] is True
    assert as_dict["is_reachable"] is True
    assert as_dict["agent_id"] == "worker-node-1"

    as_json = original.to_json()
    assert isinstance(as_json, str)

    restored_from_dict = AgentAvailability.from_dict(as_dict)
    assert restored_from_dict.state == original.state
    assert restored_from_dict.execution_state == original.execution_state
    assert math.isclose(restored_from_dict.heartbeat_age_seconds, 4.512)
    assert restored_from_dict.last_seen_at == original.last_seen_at
    assert restored_from_dict.agent_id == original.agent_id
    assert restored_from_dict.is_eligible is True

    restored_from_json = AgentAvailability.from_json(as_json)
    assert restored_from_json == restored_from_dict


def test_agent_heartbeat_contract() -> None:
    hb = AgentHeartbeat.create(agent_id="research-node-1", execution_state="idle")
    assert hb.agent_id == "research-node-1"
    assert hb.execution_state == "idle"
    assert isinstance(hb.occurred_at, str)

    with pytest.raises(ValueError, match="agent_id must not be empty"):
        AgentHeartbeat(agent_id="", occurred_at="2026-11-05T13:00:00Z")

    with pytest.raises(ValueError, match="occurred_at must not be empty"):
        AgentHeartbeat(agent_id="worker-1", occurred_at="   ")

    hb_dict = hb.to_dict()
    assert hb_dict["agent_id"] == "research-node-1"

    hb_json = hb.to_json()
    restored = AgentHeartbeat.from_json(hb_json)
    assert restored == hb


def test_evaluate_agent_record_helper() -> None:
    base = datetime(2026, 11, 5, 13, 0, 0, tzinfo=UTC)
    as_of = base + timedelta(seconds=10)

    # 1. Normal record with updated_at
    record_idle = {
        "agent_id": "worker-1",
        "agent_type": "worker",
        "capabilities": ["general"],
        "state": "idle",
        "updated_at": base.isoformat(),
    }
    av1 = evaluate_agent_record(record_idle, as_of=as_of)
    assert av1.state == AgentAvailabilityState.READY
    assert av1.agent_id == "worker-1"
    assert av1.is_eligible is True

    # 2. Record with occurred_at and state=working
    record_working = {
        "agent_id": "research-1",
        "state": "working",
        "occurred_at": base.isoformat(),
    }
    av2 = evaluate_agent_record(record_working, as_of=as_of)
    assert av2.state == AgentAvailabilityState.BUSY
    assert av2.agent_id == "research-1"
    assert av2.is_reachable is True
    assert av2.is_eligible is False

    # 3. Record missing timestamp
    record_missing_ts = {
        "agent_id": "worker-missing",
        "state": "idle",
    }
    av3 = evaluate_agent_record(record_missing_ts, as_of=as_of)
    assert av3.state == AgentAvailabilityState.OFFLINE
    assert av3.agent_id == "worker-missing"
    assert av3.is_eligible is False
    assert "Missing timestamp" in av3.reason

    # 4. Record with invalid timestamp
    record_bad_ts = {
        "agent_id": "worker-bad",
        "updated_at": "invalid-timestamp",
        "state": "idle",
    }
    av4 = evaluate_agent_record(record_bad_ts, as_of=as_of)
    assert av4.state == AgentAvailabilityState.OFFLINE
    assert av4.agent_id == "worker-bad"
    assert av4.is_eligible is False
    assert "Invalid timestamp" in av4.reason


def test_orchestrator_domain_reexports() -> None:
    assert DomainAgentAvailability is AgentAvailability
    assert DomainAgentAvailabilityState is AgentAvailabilityState
    assert DomainAgentExecutionState is AgentExecutionState
    assert DomainAgentHeartbeat is AgentHeartbeat
    assert DomainLivenessPolicy is LivenessPolicy
    assert domain_evaluate_agent_record is evaluate_agent_record
    assert domain_evaluate_availability is evaluate_availability
    assert domain_parse_timestamp is parse_timestamp
