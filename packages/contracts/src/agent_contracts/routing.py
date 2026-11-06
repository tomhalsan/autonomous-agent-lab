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
        )

    @classmethod
    def from_json(cls, value: str) -> "RoutingDecision":
        return cls.from_dict(json.loads(value))

