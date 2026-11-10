import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .availability import now_iso, parse_timestamp

DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS: float = 3.0


def parse_health_probe_timeout(
    value: str | float | None,
    default: float = DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS,
) -> float:
    """Validate and parse an inference health probe timeout in seconds.

    Must be a positive, finite number.
    Rejects malformed, zero, negative, NaN, infinity, and boolean values cleanly.
    """
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
    if type(value) is bool:
        raise ValueError(
            f"probe_timeout_seconds must be a positive number, got {value!r}"
        )
    try:
        val = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"probe_timeout_seconds must be a positive number, got {value!r}"
        ) from exc
    if not math.isfinite(val) or val <= 0:
        raise ValueError(
            f"probe_timeout_seconds must be a positive number, got {val!r}"
        )
    return val


class InferenceHealthStatus(str, Enum):
    """Provider-neutral inference provider availability and health states.

    Represents operational reachability, service quality, and capacity constraints
    independently of any specific vendor, model, or orchestration policy.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    UNAVAILABLE = "unavailable"

    @property
    def is_available(self) -> bool:
        """Whether the provider is currently eligible to accept inference requests."""
        return self in (InferenceHealthStatus.HEALTHY, InferenceHealthStatus.DEGRADED)

    @property
    def is_connected(self) -> bool:
        """Whether the provider is reachable over the network and communicating."""
        return self in (
            InferenceHealthStatus.HEALTHY,
            InferenceHealthStatus.DEGRADED,
            InferenceHealthStatus.RATE_LIMITED,
            InferenceHealthStatus.QUOTA_EXHAUSTED,
        )

    @property
    def is_constrained(self) -> bool:
        """Whether the provider is temporarily or quota blocked despite being reachable."""
        return self in (
            InferenceHealthStatus.RATE_LIMITED,
            InferenceHealthStatus.QUOTA_EXHAUSTED,
        )


@dataclass(frozen=True)
class InferenceProviderHealth:
    """Immutable snapshot of inference provider health and availability.

    Captures point-in-time reachability, constraint status, latency, and sanitized reason.
    Does not encode provider-selection policies, fallback decisions, or historical metrics.
    """

    provider_name: str
    status: InferenceHealthStatus
    model: str | None = None
    checked_at: str = ""
    latency_ms: float | None = None
    retry_after_seconds: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")

        if isinstance(self.status, str) and not isinstance(self.status, InferenceHealthStatus):
            try:
                object.__setattr__(self, "status", InferenceHealthStatus(self.status))
            except ValueError as exc:
                raise ValueError(f"Invalid health status: {self.status!r}") from exc
        elif not isinstance(self.status, InferenceHealthStatus):
            raise TypeError(
                f"status must be an InferenceHealthStatus enum, got {type(self.status).__name__}"
            )

        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("model must be a non-empty string if specified")

        if not self.checked_at:
            object.__setattr__(self, "checked_at", now_iso())
        else:
            if not isinstance(self.checked_at, str):
                raise TypeError(
                    f"checked_at must be an ISO format string, got {type(self.checked_at).__name__}"
                )
            parse_timestamp(self.checked_at)

        if self.latency_ms is not None:
            if type(self.latency_ms) is bool or not isinstance(self.latency_ms, (int, float)):
                raise TypeError(
                    f"latency_ms must be a non-negative number, got {type(self.latency_ms).__name__}"
                )
            if self.latency_ms < 0:
                raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")
            object.__setattr__(self, "latency_ms", float(self.latency_ms))

        if self.retry_after_seconds is not None:
            if type(self.retry_after_seconds) is bool or not isinstance(
                self.retry_after_seconds, (int, float)
            ):
                raise TypeError(
                    f"retry_after_seconds must be a non-negative number, got {type(self.retry_after_seconds).__name__}"
                )
            if self.retry_after_seconds < 0:
                raise ValueError(
                    f"retry_after_seconds must be non-negative, got {self.retry_after_seconds}"
                )
            object.__setattr__(self, "retry_after_seconds", float(self.retry_after_seconds))

        if not isinstance(self.reason, str):
            raise TypeError(f"reason must be a string, got {type(self.reason).__name__}")

        # Invariant: Constrained or unavailable providers must have an explanatory reason.
        if (
            self.status
            in (
                InferenceHealthStatus.RATE_LIMITED,
                InferenceHealthStatus.QUOTA_EXHAUSTED,
                InferenceHealthStatus.UNAVAILABLE,
            )
            and not self.reason.strip()
        ):
            raise ValueError(
                f"reason must be provided when health status is {self.status.value}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "status": self.status.value,
            "model": self.model,
            "checked_at": self.checked_at,
            "latency_ms": self.latency_ms,
            "retry_after_seconds": self.retry_after_seconds,
            "reason": self.reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceProviderHealth":
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, got {type(data).__name__}")

        if "provider_name" not in data:
            raise ValueError("Missing required field 'provider_name'")
        if not isinstance(data["provider_name"], str):
            raise TypeError(
                f"provider_name must be a string, got {type(data['provider_name']).__name__}"
            )

        if "status" not in data:
            raise ValueError("Missing required field 'status'")

        raw_status = data["status"]
        if isinstance(raw_status, InferenceHealthStatus):
            status = raw_status
        elif isinstance(raw_status, str):
            try:
                status = InferenceHealthStatus(raw_status)
            except ValueError as exc:
                raise ValueError(f"Invalid health status: {raw_status!r}") from exc
        else:
            raise TypeError(
                f"status must be a str or InferenceHealthStatus, got {type(raw_status).__name__}"
            )

        model = data.get("model")
        if model is not None and not isinstance(model, str):
            raise TypeError(f"model must be a string or None, got {type(model).__name__}")

        checked_at = data.get("checked_at", "")
        if checked_at is not None and not isinstance(checked_at, str):
            raise TypeError(
                f"checked_at must be an ISO format string, got {type(checked_at).__name__}"
            )

        latency_ms = data.get("latency_ms")
        retry_after_seconds = data.get("retry_after_seconds")

        reason = data.get("reason", "")
        if reason is not None and not isinstance(reason, str):
            raise TypeError(f"reason must be a string, got {type(reason).__name__}")

        return cls(
            provider_name=data["provider_name"],
            status=status,
            model=model,
            checked_at=checked_at if checked_at else "",
            latency_ms=latency_ms,
            retry_after_seconds=retry_after_seconds,
            reason=reason if reason is not None else "",
        )

    @classmethod
    def from_json(cls, json_str: str) -> "InferenceProviderHealth":
        if not isinstance(json_str, str):
            raise TypeError(f"json_str must be a string, got {type(json_str).__name__}")
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        return cls.from_dict(parsed)
