from typing import Protocol

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    TaskRequested,
    TaskResult,
)


class MessageBus(Protocol):
    async def next_task(self) -> TaskRequested: ...
    async def publish(self, event: AgentEvent) -> None: ...
    async def register(self, metadata: AgentCapabilityMetadata) -> None: ...


class ResultStore(Protocol):
    async def save(self, result: TaskResult) -> None: ...
