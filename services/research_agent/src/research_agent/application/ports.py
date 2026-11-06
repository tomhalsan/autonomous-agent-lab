from typing import Protocol

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    AgentHeartbeat,
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
