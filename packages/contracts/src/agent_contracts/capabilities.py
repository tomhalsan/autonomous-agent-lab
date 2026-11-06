import json
from dataclasses import dataclass, field

DISALLOWED_METADATA_KEYS = {
    "account_id",
    "api_key",
    "credential",
    "credentials",
    "key",
    "password",
    "secret",
    "token",
}


@dataclass(frozen=True)
class AgentCapabilityMetadata:
    agent_id: str
    capabilities: list[str]
    agent_type: str = "agent"
    providers: dict[str, str] = field(default_factory=dict)
    dispatch_target: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be empty")

        if self.dispatch_target is not None:
            if not isinstance(self.dispatch_target, str) or not self.dispatch_target.strip():
                raise ValueError("dispatch_target must be a non-empty string if provided")
            for disallowed in DISALLOWED_METADATA_KEYS:
                if disallowed in self.dispatch_target.lower():
                    raise ValueError(
                        f"Sensitive key {disallowed!r} must not be exposed in dispatch_target"
                    )

        for key in self.providers:
            normalized_key = key.lower()
            for disallowed in DISALLOWED_METADATA_KEYS:
                if disallowed in normalized_key:
                    raise ValueError(
                        f"Sensitive key {key!r} must not be exposed in provider metadata"
                    )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": list(self.capabilities),
            "providers": dict(self.providers),
            "version": self.version,
        }
        if self.dispatch_target is not None:
            data["dispatch_target"] = self.dispatch_target
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AgentCapabilityMetadata":
        return cls(
            agent_id=str(data["agent_id"]),
            capabilities=list(data.get("capabilities", [])),  # type: ignore[arg-type]
            agent_type=str(data.get("agent_type", "agent")),
            providers=dict(data.get("providers", {})),  # type: ignore[arg-type]
            dispatch_target=(
                str(data["dispatch_target"]).strip()
                if data.get("dispatch_target") is not None
                else None
            ),
            version=int(data.get("version", 1)),  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, value: str) -> "AgentCapabilityMetadata":
        return cls.from_dict(json.loads(value))
