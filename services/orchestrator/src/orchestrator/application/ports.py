from typing import Protocol

from agent_contracts import TaskRequested


class TaskQueue(Protocol):
    async def enqueue(self, task: TaskRequested) -> None: ...

class StateReader(Protocol):
    async def agents(self) -> list[dict[str, str]]: ...
    async def events(self, limit: int = 50) -> list[dict[str, str]]: ...
