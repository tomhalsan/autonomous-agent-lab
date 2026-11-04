from typing import Protocol

from agent_contracts import AgentEvent, TaskRequested

from research_agent.domain.models import ResearchResult


class MessageBus(Protocol):
    async def next_task(self) -> TaskRequested: ...
    async def publish(self, event: AgentEvent) -> None: ...


class ResearchExecutor(Protocol):
    async def execute(self, task: TaskRequested) -> ResearchResult: ...

