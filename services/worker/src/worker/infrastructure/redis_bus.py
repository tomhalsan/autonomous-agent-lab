import redis.asyncio as redis
from agent_contracts import AgentEvent, TaskRequested

class RedisBus:
    def __init__(self, url: str) -> None:
        self.client = redis.from_url(url, decode_responses=True)

    async def next_task(self) -> TaskRequested:
        _, value = await self.client.blpop("agent.tasks", timeout=0)
        return TaskRequested.from_json(value)

    async def publish(self, event: AgentEvent) -> None:
        mapping = event.to_mapping()
        await self.client.sadd("agent.ids", event.agent_id)
        await self.client.hset(f"agent.state:{event.agent_id}", mapping=mapping)
        await self.client.xadd("agent.events", mapping, maxlen=500, approximate=True)
