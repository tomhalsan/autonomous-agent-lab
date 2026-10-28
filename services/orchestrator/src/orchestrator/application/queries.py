from .ports import StateReader


class GetAgentStates:
    def __init__(self, reader: StateReader) -> None:
        self._reader = reader

    async def execute(self) -> list[dict[str, str]]:
        return await self._reader.agents()


class GetRecentEvents:
    def __init__(self, reader: StateReader) -> None:
        self._reader = reader

    async def execute(self, limit: int = 50) -> list[dict[str, str]]:
        bounded_limit = min(max(limit, 1), 100)
        return await self._reader.events(limit=bounded_limit)

