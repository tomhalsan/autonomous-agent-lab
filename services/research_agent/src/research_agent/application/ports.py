from typing import Protocol, runtime_checkable

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    AgentHeartbeat,
    InferenceProviderHealth,
    TaskRequested,
)

from research_agent.domain.models import ResearchResult, SynthesisResult


class MessageBus(Protocol):
    async def next_task(self) -> TaskRequested: ...
    async def publish(self, event: AgentEvent) -> None: ...
    async def register(self, metadata: AgentCapabilityMetadata) -> None: ...
    async def heartbeat(self, heartbeat: AgentHeartbeat) -> None: ...


class ResearchExecutor(Protocol):
    async def execute(self, task: TaskRequested) -> ResearchResult: ...


class ResultStore(Protocol):
    async def save(self, result: ResearchResult) -> None: ...


class InferenceProvider(Protocol):
    async def synthesize(self, question: str, evidence: str) -> SynthesisResult: ...


ResearchSynthesizer = InferenceProvider


@runtime_checkable
class InferenceHealthProbe(Protocol):
    """Application port for active or passive inference provider health probing."""

    async def check_health(self) -> InferenceProviderHealth: ...


@runtime_checkable
class RuntimeInferenceProvider(InferenceProvider, InferenceHealthProbe, Protocol):
    """Application port combining research synthesis and health probing capabilities."""
