from typing import Any, Protocol, runtime_checkable

from agent_contracts import AgentCandidate, AgentEvent, RoutingDecision, TaskRequested

from orchestrator.domain.capability import AgentCapability


@runtime_checkable
class TaskQueue(Protocol):
    async def enqueue(
        self,
        task: TaskRequested,
        capability: AgentCapability = AgentCapability.GENERAL,
        target: str | None = None,
    ) -> None: ...


@runtime_checkable
class TaskDispatcher(Protocol):
    async def dispatch(
        self,
        task: TaskRequested,
        target: str,
    ) -> None: ...


class StateReader(Protocol):
    async def agents(self) -> list[dict[str, Any]]: ...
    async def events(self, limit: int = 50) -> list[dict[str, str]]: ...
    async def routing_decision(self, task_id: str) -> dict[str, Any] | None: ...


class EventPublisher(Protocol):
    async def publish(self, event: AgentEvent) -> None: ...


@runtime_checkable
class RoutingRecorder(Protocol):
    async def record_routing_decision(self, decision: RoutingDecision) -> None: ...


class CandidateFinder(Protocol):
    async def execute(
        self,
        capability: AgentCapability | str,
        as_of: Any = None,
    ) -> list[AgentCandidate]: ...


class CapabilityRouter(Protocol):
    def route(self, prompt: str) -> AgentCapability: ...


class ResultReader(Protocol):
    async def get_result(self, task_id: str) -> dict[str, Any] | None: ...
