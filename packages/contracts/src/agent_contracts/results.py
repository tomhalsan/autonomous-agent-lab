import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceResponse:
    """Normalized provider-neutral response envelope for worker inference."""

    text: str
    provider_name: str
    model_name: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a str, got {type(self.text).__name__}")

        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        object.__setattr__(self, "provider_name", self.provider_name.strip().lower())

        if self.model_name is not None:
            if not isinstance(self.model_name, str) or not self.model_name.strip():
                raise ValueError("model_name must be a non-empty string if specified")
            object.__setattr__(self, "model_name", self.model_name.strip())

        if self.finish_reason is not None:
            if not isinstance(self.finish_reason, str) or not self.finish_reason.strip():
                raise ValueError("finish_reason must be a non-empty string if specified")
            object.__setattr__(self, "finish_reason", self.finish_reason.strip().lower())

        for token_field, val in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("total_tokens", self.total_tokens),
        ):
            if val is not None:
                if type(val) is bool or not isinstance(val, int):
                    raise TypeError(f"{token_field} must be an int or None, got {type(val).__name__}")
                if val < 0:
                    raise ValueError(f"{token_field} must be a non-negative integer, got {val}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceResponse":
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "InferenceResponse":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    output: str
    status: str = "completed"
    acknowledged: bool = True
    sources: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.status.strip():
            raise ValueError("status must not be empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TaskResult":
        raw_sources = data.get("sources")
        sources_list: list[dict[str, object]] | None = None
        if isinstance(raw_sources, list):
            sources_list = [s for s in raw_sources if isinstance(s, dict)]

        return cls(
            task_id=str(data["task_id"]),
            output=str(data["output"]),
            status=str(data.get("status", "completed")),
            acknowledged=bool(data.get("acknowledged", True)),
            sources=sources_list,
        )

    @classmethod
    def from_json(cls, value: str) -> "TaskResult":
        return cls.from_dict(json.loads(value))
