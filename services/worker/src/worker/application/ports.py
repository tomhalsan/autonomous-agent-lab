from typing import Protocol, runtime_checkable

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    AgentHeartbeat,
    InferenceProviderHealth,
    TaskRequested,
    TaskResult,
)


class MessageBus(Protocol):
    async def next_task(self) -> TaskRequested: ...
    async def publish(self, event: AgentEvent) -> None: ...
    async def register(self, metadata: AgentCapabilityMetadata) -> None: ...
    async def heartbeat(self, heartbeat: AgentHeartbeat) -> None: ...


class ResultStore(Protocol):
    async def save(self, result: TaskResult) -> None: ...


class InferenceProvider(Protocol):
    async def generate(self, prompt: str) -> str: ...


@runtime_checkable
class InferenceHealthProbe(Protocol):
    """Application port for active or passive inference provider health probing."""

    async def check_health(self) -> InferenceProviderHealth: ...


@runtime_checkable
class RuntimeInferenceProvider(InferenceProvider, InferenceHealthProbe, Protocol):
    """Application port combining prompt generation and health probing capabilities."""

