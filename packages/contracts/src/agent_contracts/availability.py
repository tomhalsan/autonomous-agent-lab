import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


def now_iso() -> str:
    """Return the current UTC timestamp formatted as ISO 8601."""
    return datetime.now(UTC).isoformat()


def parse_timestamp(value: datetime | str) -> datetime:
    """Parse an ISO 8601 string or datetime into a UTC-aware datetime.

    Raises:
        ValueError: If value is an empty or invalid ISO timestamp.
        TypeError: If value is not a datetime or string.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    if not isinstance(value, str):
        raise TypeError(f"Expected datetime or str, got {type(value).__name__}")

    stripped = value.strip()
    if not stripped:
        raise ValueError("Timestamp string must not be empty")

    try:
        dt = datetime.fromisoformat(stripped)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid ISO timestamp format: {value!r}") from exc

    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class AgentAvailabilityState(str, Enum):
    """Platform-level availability states determining orchestration eligibility."""

    READY = "ready"
    BUSY = "busy"
    STALE = "stale"
    OFFLINE = "offline"

    @property
    def is_eligible(self) -> bool:
        """True if the agent is ready to be dispatched new work."""
        return self is AgentAvailabilityState.READY

    @property
    def is_reachable(self) -> bool:
        """True if the agent process is actively responsive (ready or busy)."""
        return self in (AgentAvailabilityState.READY, AgentAvailabilityState.BUSY)


class AgentExecutionState(str, Enum):
    """Operational state describing what an agent is currently doing."""

    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, value: str) -> "AgentExecutionState":
        """Normalize a string to an AgentExecutionState."""
        if not isinstance(value, str):
            raise TypeError(f"Expected str, got {type(value).__name__}")
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True)
class LivenessPolicy:
    """Configurable thresholds for deterministic agent liveness evaluation."""

    stale_threshold_seconds: float = 30.0
    offline_threshold_seconds: float = 90.0
    clock_skew_tolerance_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name, val in [
            ("stale_threshold_seconds", self.stale_threshold_seconds),
            ("offline_threshold_seconds", self.offline_threshold_seconds),
            ("clock_skew_tolerance_seconds", self.clock_skew_tolerance_seconds),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"{name} must be a finite number")

        if self.stale_threshold_seconds <= 0:
            raise ValueError(
                f"stale_threshold_seconds must be positive and non-zero (got {self.stale_threshold_seconds})"
            )
        if self.offline_threshold_seconds <= self.stale_threshold_seconds:
            raise ValueError(
                f"offline_threshold_seconds ({self.offline_threshold_seconds}) must be strictly greater "
                f"than stale_threshold_seconds ({self.stale_threshold_seconds})"
            )
        if self.clock_skew_tolerance_seconds < 0:
            raise ValueError(
                f"clock_skew_tolerance_seconds must not be negative (got {self.clock_skew_tolerance_seconds})"
            )

    def calculate_heartbeat_age(
        self,
        last_seen_at: datetime | str,
        as_of: datetime | str,
    ) -> float:
        """Calculate elapsed seconds since last_seen_at relative to as_of.

        Raises:
            ValueError: If last_seen_at is in the future beyond clock skew tolerance.
        """
        last_seen_dt = parse_timestamp(last_seen_at)
        as_of_dt = parse_timestamp(as_of)

        elapsed = (as_of_dt - last_seen_dt).total_seconds()
        if elapsed < -self.clock_skew_tolerance_seconds:
            raise ValueError(
                f"last_seen timestamp ({last_seen_dt.isoformat()}) is in the future relative to "
                f"as_of ({as_of_dt.isoformat()}) beyond tolerance ({self.clock_skew_tolerance_seconds}s)"
            )
        if elapsed < 0:
            return 0.0
        return elapsed

    def evaluate(
        self,
        last_seen_at: datetime | str,
        execution_state: AgentExecutionState | str = AgentExecutionState.IDLE,
        as_of: datetime | str | None = None,
        *,
        agent_id: str | None = None,
        explicit_offline: bool = False,
    ) -> "AgentAvailability":
        """Deterministically evaluate agent availability against reference time as_of."""
        as_of_dt = parse_timestamp(as_of) if as_of is not None else datetime.now(UTC)
        last_seen_dt = parse_timestamp(last_seen_at)
        heartbeat_age = self.calculate_heartbeat_age(last_seen_dt, as_of_dt)

        exec_state = (
            execution_state
            if isinstance(execution_state, AgentExecutionState)
            else AgentExecutionState.from_str(str(execution_state))
        )

        last_seen_iso = last_seen_dt.isoformat()
        as_of_iso = as_of_dt.isoformat()

        if explicit_offline:
            return AgentAvailability(
                state=AgentAvailabilityState.OFFLINE,
                execution_state=exec_state,
                heartbeat_age_seconds=heartbeat_age,
                last_seen_at=last_seen_iso,
                as_of=as_of_iso,
                reason="Agent explicitly reported offline",
                agent_id=agent_id,
            )

        if heartbeat_age > self.offline_threshold_seconds:
            return AgentAvailability(
                state=AgentAvailabilityState.OFFLINE,
                execution_state=exec_state,
                heartbeat_age_seconds=heartbeat_age,
                last_seen_at=last_seen_iso,
                as_of=as_of_iso,
                reason=(
                    f"Heartbeat age {heartbeat_age:.1f}s exceeds offline threshold "
                    f"{self.offline_threshold_seconds:.1f}s"
                ),
                agent_id=agent_id,
            )

        if heartbeat_age > self.stale_threshold_seconds:
            return AgentAvailability(
                state=AgentAvailabilityState.STALE,
                execution_state=exec_state,
                heartbeat_age_seconds=heartbeat_age,
                last_seen_at=last_seen_iso,
                as_of=as_of_iso,
                reason=(
                    f"Heartbeat age {heartbeat_age:.1f}s exceeds stale threshold "
                    f"{self.stale_threshold_seconds:.1f}s"
                ),
                agent_id=agent_id,
            )

        # Heartbeat is fresh (elapsed <= stale_threshold_seconds)
        if exec_state == AgentExecutionState.ERROR:
            return AgentAvailability(
                state=AgentAvailabilityState.BUSY,
                execution_state=exec_state,
                heartbeat_age_seconds=heartbeat_age,
                last_seen_at=last_seen_iso,
                as_of=as_of_iso,
                reason=(
                    "Agent is reachable with fresh heartbeat, but in error execution state "
                    "(not eligible for dispatch)"
                ),
                agent_id=agent_id,
            )

        if exec_state == AgentExecutionState.WORKING:
            return AgentAvailability(
                state=AgentAvailabilityState.BUSY,
                execution_state=exec_state,
                heartbeat_age_seconds=heartbeat_age,
                last_seen_at=last_seen_iso,
                as_of=as_of_iso,
                reason="Agent is actively working on a task",
                agent_id=agent_id,
            )

        if exec_state == AgentExecutionState.UNKNOWN:
            return AgentAvailability(
                state=AgentAvailabilityState.BUSY,
                execution_state=exec_state,
                heartbeat_age_seconds=heartbeat_age,
                last_seen_at=last_seen_iso,
                as_of=as_of_iso,
                reason=(
                    f"Agent is reachable with fresh heartbeat, but in unknown execution state "
                    f"{execution_state!r} (not eligible for dispatch)"
                ),
                agent_id=agent_id,
            )

        return AgentAvailability(
            state=AgentAvailabilityState.READY,
            execution_state=exec_state,
            heartbeat_age_seconds=heartbeat_age,
            last_seen_at=last_seen_iso,
            as_of=as_of_iso,
            reason="Agent is idle with fresh heartbeat",
            agent_id=agent_id,
        )


@dataclass(frozen=True)
class AgentAvailability:
    """Immutable evaluation of an agent's runtime availability."""

    state: AgentAvailabilityState
    execution_state: AgentExecutionState
    heartbeat_age_seconds: float
    last_seen_at: str
    as_of: str
    reason: str
    agent_id: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.state, str) and not isinstance(self.state, AgentAvailabilityState):
            object.__setattr__(self, "state", AgentAvailabilityState(self.state.lower()))
        elif not isinstance(self.state, AgentAvailabilityState):
            raise TypeError(f"state must be an AgentAvailabilityState, got {type(self.state).__name__}")

        if isinstance(self.execution_state, str) and not isinstance(self.execution_state, AgentExecutionState):
            object.__setattr__(self, "execution_state", AgentExecutionState.from_str(self.execution_state))
        elif not isinstance(self.execution_state, AgentExecutionState):
            raise TypeError(
                f"execution_state must be an AgentExecutionState, got {type(self.execution_state).__name__}"
            )

        if self.heartbeat_age_seconds < 0:
            raise ValueError(f"heartbeat_age_seconds must not be negative: {self.heartbeat_age_seconds}")

        if not self.reason.strip():
            raise ValueError("reason must not be empty")

    @property
    def is_eligible(self) -> bool:
        """True if the agent is ready to be dispatched new work."""
        return self.state == AgentAvailabilityState.READY

    @property
    def is_reachable(self) -> bool:
        """True if the agent is actively responsive (ready or busy)."""
        return self.state in (AgentAvailabilityState.READY, AgentAvailabilityState.BUSY)

    @property
    def is_stale(self) -> bool:
        """True if the agent's heartbeat is in the stale decay window."""
        return self.state == AgentAvailabilityState.STALE

    @property
    def is_offline(self) -> bool:
        """True if the agent is determined offline."""
        return self.state == AgentAvailabilityState.OFFLINE

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "state": self.state.value,
            "execution_state": self.execution_state.value,
            "heartbeat_age_seconds": round(self.heartbeat_age_seconds, 3),
            "last_seen_at": self.last_seen_at,
            "as_of": self.as_of,
            "reason": self.reason,
            "is_eligible": self.is_eligible,
            "is_reachable": self.is_reachable,
            "version": self.version,
        }
        if self.agent_id is not None:
            data["agent_id"] = self.agent_id
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentAvailability":
        return cls(
            state=AgentAvailabilityState(str(data["state"])),
            execution_state=AgentExecutionState.from_str(str(data["execution_state"])),
            heartbeat_age_seconds=float(data["heartbeat_age_seconds"]),
            last_seen_at=str(data["last_seen_at"]),
            as_of=str(data["as_of"]),
            reason=str(data["reason"]),
            agent_id=str(data["agent_id"]) if data.get("agent_id") is not None else None,
            version=int(data.get("version", 1)),
        )

    @classmethod
    def from_json(cls, value: str) -> "AgentAvailability":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class AgentHeartbeat:
    """Heartbeat signal published by an agent to maintain runtime liveness."""

    agent_id: str
    occurred_at: str
    execution_state: str = "idle"
    version: int = 1

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be empty")
        if not self.occurred_at.strip():
            raise ValueError("occurred_at must not be empty")

    @classmethod
    def create(
        cls,
        agent_id: str,
        execution_state: str = "idle",
        occurred_at: str | None = None,
    ) -> "AgentHeartbeat":
        return cls(
            agent_id=agent_id.strip(),
            occurred_at=occurred_at or now_iso(),
            execution_state=execution_state,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentHeartbeat":
        return cls(
            agent_id=str(data["agent_id"]),
            occurred_at=str(data["occurred_at"]),
            execution_state=str(data.get("execution_state", "idle")),
            version=int(data.get("version", 1)),
        )

    @classmethod
    def from_json(cls, value: str) -> "AgentHeartbeat":
        return cls.from_dict(json.loads(value))


def evaluate_availability(
    last_seen_at: datetime | str,
    execution_state: AgentExecutionState | str = AgentExecutionState.IDLE,
    as_of: datetime | str | None = None,
    *,
    policy: LivenessPolicy | None = None,
    agent_id: str | None = None,
    explicit_offline: bool = False,
) -> AgentAvailability:
    """Evaluate an agent's availability using the specified or default policy."""
    active_policy = policy or LivenessPolicy()
    return active_policy.evaluate(
        last_seen_at=last_seen_at,
        execution_state=execution_state,
        as_of=as_of,
        agent_id=agent_id,
        explicit_offline=explicit_offline,
    )


def evaluate_agent_record(
    record: dict[str, Any],
    as_of: datetime | str | None = None,
    *,
    policy: LivenessPolicy | None = None,
) -> AgentAvailability:
    """Evaluate an agent's availability from a stored registration/event dictionary."""
    agent_id = str(record.get("agent_id", "")).strip() or None
    raw_timestamp = record.get("updated_at") or record.get("occurred_at")
    raw_state = record.get("state", "idle")

    if not raw_timestamp:
        as_of_dt = parse_timestamp(as_of) if as_of is not None else datetime.now(UTC)
        return AgentAvailability(
            state=AgentAvailabilityState.OFFLINE,
            execution_state=AgentExecutionState.from_str(str(raw_state)),
            heartbeat_age_seconds=0.0,
            last_seen_at="",
            as_of=as_of_dt.isoformat(),
            reason="Missing timestamp in agent record",
            agent_id=agent_id,
        )

    try:
        return evaluate_availability(
            last_seen_at=str(raw_timestamp),
            execution_state=str(raw_state),
            as_of=as_of,
            policy=policy,
            agent_id=agent_id,
        )
    except ValueError as exc:
        as_of_dt = parse_timestamp(as_of) if as_of is not None else datetime.now(UTC)
        return AgentAvailability(
            state=AgentAvailabilityState.OFFLINE,
            execution_state=AgentExecutionState.from_str(str(raw_state)),
            heartbeat_age_seconds=0.0,
            last_seen_at=str(raw_timestamp),
            as_of=as_of_dt.isoformat(),
            reason=f"Invalid timestamp in agent record: {exc}",
            agent_id=agent_id,
        )

