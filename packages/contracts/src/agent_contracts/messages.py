import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class TaskRequested:
    task_id: str
    prompt: str
    created_at: str
    version: int = 1

    @classmethod
    def create(cls, prompt: str) -> "TaskRequested":
        return cls(task_id=str(uuid4()), prompt=prompt.strip(), created_at=now_iso())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TaskRequested":
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "TaskRequested":
        return cls(**json.loads(value))


@dataclass(frozen=True)
class AgentEvent:
    agent_id: str
    state: str
    detail: str
    occurred_at: str
    task_id: str | None = None
    version: int = 1

    @classmethod
    def create(
        cls, agent_id: str, state: str, detail: str, task_id: str | None = None
    ) -> "AgentEvent":
        return cls(
            agent_id=agent_id,
            state=state,
            detail=detail,
            occurred_at=now_iso(),
            task_id=task_id,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AgentEvent":
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "AgentEvent":
        return cls(**json.loads(value))
