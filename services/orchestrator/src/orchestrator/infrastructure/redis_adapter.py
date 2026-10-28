import redis.asyncio as redis
from agent_contracts import TaskRequested

class RedisGateway:
    def __init__(self, url: str) -> None:
        self._client = redis.from_url(url, decode_responses=True)

    async def enqueue(self, task: TaskRequested) -> None:
        await self._client.rpush("agent.tasks", task.to_json())

    async def agents(self) -> list[dict[str, str]]:
        ids = await self._client.smembers("agent.ids")
        return [await self._client.hgetall(f"agent.state:{agent_id}") for agent_id in sorted(ids)]

    async def events(self, limit: int = 50) -> list[dict[str, str]]:
        rows = await self._client.xrevrange("agent.events", count=limit)
        return [dict(fields, event_id=event_id) for event_id, fields in rows]
