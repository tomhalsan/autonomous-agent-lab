import json
from dataclasses import asdict, dataclass
from typing import Any

from .capabilities import DISALLOWED_METADATA_KEYS
from .messages import now_iso


def _validate_no_sensitive_data(obj: Any, path: str = "") -> None:
    """Recursively verify that no sensitive keys or substrings appear in decision records."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            k_str = str(k).lower()
            for disallowed in DISALLOWED_METADATA_KEYS:
                if disallowed in k_str:
                    raise ValueError(
                        f"Sensitive key {k!r} must not be exposed in routing decision at {path or 'root'}"
                    )
            _validate_no_sensitive_data(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _validate_no_sensitive_data(item, f"{path}[{i}]")


ALLOWED_CLARITIES = {None, "clear", "ambiguous"}
ALLOWED_PROVISIONAL_CAPABILITIES = {None, "general", "research"}
ALLOWED_ADVISOR_VERDICTS = {
    None,
    "agreed",
    "changed",
    "unavailable",
    "timeout",
    "failed",
}


@dataclass(frozen=True)
class RoutingProvenance:
    """Detailed provenance tracking the decision pipeline for a task."""

    is_explicit_override: bool = False
    assessment_clarity: str | None = None
    provisional_capability: str | None = None
    advisor_consulted: bool = False
    advisor_verdict: str | None = None
    fallback_used: bool = False
    final_selection_reason: str = ""
    deterministic_assessment_reason: str = ""
    deterministic_signal_types: tuple[str, ...] = ()
    advisor_reason: str = ""

    def __post_init__(self) -> None:
        for field_name in ("is_explicit_override", "advisor_consulted", "fallback_used"):
            val = getattr(self, field_name)
            if type(val) is not bool:
                raise TypeError(
                    f"{field_name} must be a boolean, got {type(val).__name__} ({val!r})"
                )

        if not isinstance(self.final_selection_reason, str) or not self.final_selection_reason.strip():
            raise ValueError("final_selection_reason must be a non-empty string")

        if not isinstance(self.deterministic_assessment_reason, str):
            raise TypeError("deterministic_assessment_reason must be a string")

        if not isinstance(self.advisor_reason, str):
            raise TypeError("advisor_reason must be a string")

        if not isinstance(self.deterministic_signal_types, tuple):
            if isinstance(self.deterministic_signal_types, list):
                object.__setattr__(
                    self,
                    "deterministic_signal_types",
                    tuple(self.deterministic_signal_types),
                )
            else:
                raise TypeError(
                    f"deterministic_signal_types must be a tuple or list of strings, got {type(self.deterministic_signal_types).__name__}"
                )

        for sig in self.deterministic_signal_types:
            if not isinstance(sig, str):
                raise TypeError("all elements in deterministic_signal_types must be strings")

        if self.assessment_clarity not in ALLOWED_CLARITIES:
            raise ValueError(
                f"Invalid assessment_clarity: {self.assessment_clarity!r}. Allowed: {sorted(c for c in ALLOWED_CLARITIES if c is not None)}"
            )

        if self.provisional_capability not in ALLOWED_PROVISIONAL_CAPABILITIES:
            raise ValueError(
                f"Invalid provisional_capability: {self.provisional_capability!r}. Allowed: {sorted(c for c in ALLOWED_PROVISIONAL_CAPABILITIES if c is not None)}"
            )

        if self.advisor_verdict not in ALLOWED_ADVISOR_VERDICTS:
            raise ValueError(
                f"Invalid advisor_verdict: {self.advisor_verdict!r}. Allowed: {sorted(c for c in ALLOWED_ADVISOR_VERDICTS if c is not None)}"
            )

        if self.is_explicit_override:
            if self.advisor_verdict is not None:
                raise ValueError("explicit override requires advisor_verdict=None")
            if self.fallback_used:
                raise ValueError("explicit override requires fallback_used=False")
            if self.advisor_consulted:
                raise ValueError("explicit override requires advisor_consulted=False")

        # Invariant validations
        if self.advisor_verdict is not None and not self.advisor_consulted:
            raise ValueError("advisor_verdict requires advisor_consulted=True")

        if not self.advisor_consulted and self.advisor_verdict is not None:
            raise ValueError("advisor_consulted=False requires advisor_verdict=None")

        if self.advisor_verdict in ("agreed", "changed") and self.fallback_used:
            raise ValueError(f"advisor_verdict '{self.advisor_verdict}' requires fallback_used=False")

        if self.advisor_verdict in ("unavailable", "timeout", "failed") and not self.fallback_used:
            raise ValueError(f"advisor_verdict '{self.advisor_verdict}' requires fallback_used=True")

        if self.fallback_used and not self.advisor_consulted:
            raise ValueError("fallback_used=True requires advisor_consulted=True")

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_explicit_override": self.is_explicit_override,
            "assessment_clarity": self.assessment_clarity,
            "provisional_capability": self.provisional_capability,
            "advisor_consulted": self.advisor_consulted,
            "advisor_verdict": self.advisor_verdict,
            "fallback_used": self.fallback_used,
            "final_selection_reason": self.final_selection_reason,
            "deterministic_assessment_reason": self.deterministic_assessment_reason,
            "deterministic_signal_types": list(self.deterministic_signal_types),
            "advisor_reason": self.advisor_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoutingProvenance":
        if not isinstance(data, dict):
            raise TypeError(f"RoutingProvenance.from_dict requires dict, got {type(data).__name__}")

        def _get_bool(key: str, default: bool = False) -> bool:
            if key not in data:
                return default
            val = data[key]
            if type(val) is not bool:
                raise TypeError(
                    f"Field '{key}' must be a boolean, got {type(val).__name__} ({val!r})"
                )
            return val

        def _get_optional_str(key: str) -> str | None:
            val = data.get(key)
            if val is None:
                return None
            if not isinstance(val, str):
                raise TypeError(f"Field '{key}' must be a string or None, got {type(val).__name__}")
            return val

        def _get_str(key: str, default: str = "") -> str:
            if key not in data:
                return default
            val = data[key]
            if not isinstance(val, str):
                raise TypeError(f"Field '{key}' must be a string, got {type(val).__name__}")
            return val

        signals_raw = data.get("deterministic_signal_types", ())
        if not isinstance(signals_raw, (list, tuple)):
            raise TypeError(
                f"Field 'deterministic_signal_types' must be a list or tuple, got {type(signals_raw).__name__}"
            )
        signals: list[str] = []
        for s in signals_raw:
            if not isinstance(s, str):
                raise TypeError(f"Signal type must be string, got {type(s).__name__}")
            signals.append(s)

        return cls(
            is_explicit_override=_get_bool("is_explicit_override", False),
            assessment_clarity=_get_optional_str("assessment_clarity"),
            provisional_capability=_get_optional_str("provisional_capability"),
            advisor_consulted=_get_bool("advisor_consulted", False),
            advisor_verdict=_get_optional_str("advisor_verdict"),
            fallback_used=_get_bool("fallback_used", False),
            final_selection_reason=_get_str("final_selection_reason", ""),
            deterministic_assessment_reason=_get_str("deterministic_assessment_reason", ""),
            deterministic_signal_types=tuple(signals),
            advisor_reason=_get_str("advisor_reason", ""),
        )


@dataclass(frozen=True)
class RoutingDecision:
    """Structured record of an orchestration routing decision."""

    task_id: str
    prompt_classification: str
    required_capabilities: list[str]
    candidates_evaluated: list[dict[str, Any]]
    selected_agent_id: str | None
    dispatch_target: str | None
    rationale: str
    decision_timestamp: str = ""
    version: int = 1
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not str(self.task_id).strip():
            raise ValueError("task_id must not be empty")

        if not self.prompt_classification or not str(self.prompt_classification).strip():
            raise ValueError("prompt_classification must not be empty")

        if not isinstance(self.required_capabilities, list) or not self.required_capabilities:
            raise ValueError("required_capabilities must be a non-empty list of strings")

        for cap in self.required_capabilities:
            if not isinstance(cap, str) or not cap.strip():
                raise ValueError("Capability entries must be non-empty strings")

        if not isinstance(self.candidates_evaluated, list):
            raise TypeError("candidates_evaluated must be a list of candidate dictionaries")

        if not self.rationale or not str(self.rationale).strip():
            raise ValueError("rationale must not be empty")

        if not self.decision_timestamp:
            object.__setattr__(self, "decision_timestamp", now_iso())

        if self.provenance is not None:
            if not isinstance(self.provenance, dict):
                raise TypeError("provenance must be a dictionary or None")
            _validate_no_sensitive_data(self.provenance, "provenance")

        # Validate that no sensitive credentials or provider tokens are present
        _validate_no_sensitive_data(self.candidates_evaluated, "candidates_evaluated")
        _validate_no_sensitive_data(self.rationale, "rationale")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoutingDecision":
        return cls(
            task_id=str(data["task_id"]),
            prompt_classification=str(data["prompt_classification"]),
            required_capabilities=list(data.get("required_capabilities", [])),
            candidates_evaluated=list(data.get("candidates_evaluated", [])),
            selected_agent_id=(
                str(data["selected_agent_id"]) if data.get("selected_agent_id") is not None else None
            ),
            dispatch_target=(
                str(data["dispatch_target"]) if data.get("dispatch_target") is not None else None
            ),
            rationale=str(data.get("rationale", "")),
            decision_timestamp=str(data.get("decision_timestamp", "")),
            version=int(data.get("version", 1)),
            provenance=(
                dict(data["provenance"]) if data.get("provenance") is not None else None
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "RoutingDecision":
        return cls.from_dict(json.loads(value))

