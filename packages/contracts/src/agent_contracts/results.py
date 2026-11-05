import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    output: str
    status: str = "completed"
    acknowledged: bool = True

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
        return cls(
            task_id=str(data["task_id"]),
            output=str(data["output"]),
            status=str(data.get("status", "completed")),
            acknowledged=bool(data.get("acknowledged", True)),
        )

    @classmethod
    def from_json(cls, value: str) -> "TaskResult":
        return cls.from_dict(json.loads(value))
