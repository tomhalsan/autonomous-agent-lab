import json
from typing import Any

import redis.asyncio as redis
from agent_contracts import TaskRequested

from orchestrator.domain.capability import AgentCapability

CAPABILITY_QUEUE_MAP: dict[AgentCapability, str] = {
    AgentCapability.GENERAL: "agent.tasks.general",
    AgentCapability.RESEARCH: "agent.tasks.research",
}


class RedisGateway:
    def __init__(self, url: str) -> None:
        self._client = redis.from_url(url, decode_responses=True)

    async def enqueue(
        self,
        task: TaskRequested,
        capability: AgentCapability = AgentCapability.GENERAL,
    ) -> None:
        queue_name = CAPABILITY_QUEUE_MAP.get(capability)
        if not queue_name:
            raise ValueError(f"Unknown capability: {capability}")
        await self._client.set(
            f"task.status:{task.task_id}",
            json.dumps({"task_id": task.task_id, "status": "pending"}),
        )
        await self._client.rpush(queue_name, task.to_json())

    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        raw_result = await self._client.get(f"task.result:{task_id}")
        if raw_result is not None:
            return json.loads(raw_result)

        raw_status = await self._client.get(f"task.status:{task_id}")
        if raw_status is not None:
            status_data = json.loads(raw_status)
            return {
                "task_id": task_id,
                "status": status_data.get("status", "pending"),
                "output": None,
            }

        events = await self.events(limit=100)
        matching = [e for e in events if e.get("task_id") == task_id]
        if matching:
            latest = matching[0]
            state = latest.get("state", "pending")
            return {
                "task_id": task_id,
                "status": "pending" if state == "working" else state,
                "output": None,
            }

        return None

    async def agents(self) -> list[dict[str, str]]:
        ids = await self._client.smembers("agent.ids")
        return [await self._client.hgetall(f"agent.state:{agent_id}") for agent_id in sorted(ids)]

    async def events(self, limit: int = 50) -> list[dict[str, str]]:
        rows = await self._client.xrevrange("agent.events", count=limit)
        return [dict(fields, event_id=event_id) for event_id, fields in rows]

    async def aclose(self) -> None:
        await self._client.aclose()

