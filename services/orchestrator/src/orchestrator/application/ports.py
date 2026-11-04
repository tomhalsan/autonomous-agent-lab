from typing import Any, Protocol

from agent_contracts import TaskRequested

from orchestrator.domain.capability import AgentCapability


class TaskQueue(Protocol):
    async def enqueue(
        self,
        task: TaskRequested,
        capability: AgentCapability = AgentCapability.GENERAL,
    ) -> None: ...


class StateReader(Protocol):
    async def agents(self) -> list[dict[str, str]]: ...
    async def events(self, limit: int = 50) -> list[dict[str, str]]: ...


class CapabilityRouter(Protocol):
    def route(self, prompt: str) -> AgentCapability: ...


class ResultReader(Protocol):
    async def get_result(self, task_id: str) -> dict[str, Any] | None: ...
