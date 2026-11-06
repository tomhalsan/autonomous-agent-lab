import asyncio
import json
import logging
from collections.abc import Callable, Coroutine

import redis.asyncio as redis
from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    AgentHeartbeat,
    TaskRequested,
)

from research_agent.domain.models import ResearchResult

logger = logging.getLogger(__name__)


async def retry_with_backoff[T](
    action: Callable[[], Coroutine[object, object, T]],
    max_retries: int = 5,
    initial_delay: float = 0.5,
    max_delay: float = 5.0,
    backoff_factor: float = 2.0,
) -> T:
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return await action()
        except (redis.ConnectionError, redis.TimeoutError, OSError) as exc:
            if attempt == max_retries:
                logger.error("Redis operation failed after %d attempts: %s", max_retries, exc)
                raise
            logger.warning(
                "Redis connection failed (attempt %d/%d). Retrying in %.1fs: %s",
                attempt,
                max_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
    raise RuntimeError("Retry loop exited unexpectedly")


class RedisBus:
    def __init__(self, url: str, queue_name: str = "agent.tasks.research") -> None:
        self.client = redis.from_url(url, decode_responses=True)
        self.queue_name = queue_name
        self.current_state: str = "idle"

    async def next_task(self) -> TaskRequested:
        async def _fetch():
            return await self.client.blpop(self.queue_name, timeout=2)

        while True:
            res = await retry_with_backoff(_fetch)
            if res is not None:
                _, value = res
                return TaskRequested.from_json(value)

    async def register(self, metadata: AgentCapabilityMetadata) -> None:
        mapping = {
            "agent_id": metadata.agent_id,
            "agent_type": metadata.agent_type,
            "capabilities": json.dumps(metadata.capabilities),
            "providers": json.dumps(metadata.providers),
            "version": str(metadata.version),
        }
        if metadata.dispatch_target:
            mapping["dispatch_target"] = metadata.dispatch_target
        for k, v in metadata.providers.items():
            mapping[k] = str(v)

        async def _write():
            await self.client.sadd("agent.ids", metadata.agent_id)
            await self.client.hset(f"agent.state:{metadata.agent_id}", mapping=mapping)

        await retry_with_backoff(_write)

    async def publish(self, event: AgentEvent) -> None:
        self.current_state = event.state
        mapping = {
            key: "" if value is None else str(value)
            for key, value in event.to_dict().items()
        }
        mapping["updated_at"] = event.occurred_at

        async def _write():
            await self.client.sadd("agent.ids", event.agent_id)
            await self.client.hset(f"agent.state:{event.agent_id}", mapping=mapping)
            await self.client.xadd("agent.events", mapping, maxlen=500, approximate=True)

        await retry_with_backoff(_write)

    async def heartbeat(self, heartbeat: AgentHeartbeat) -> None:
        mapping = {
            "updated_at": heartbeat.occurred_at,
        }

        async def _write():
            await self.client.sadd("agent.ids", heartbeat.agent_id)
            await self.client.hset(f"agent.state:{heartbeat.agent_id}", mapping=mapping)

        await retry_with_backoff(_write)

    async def save(self, result: ResearchResult) -> None:
        data = {
            "task_id": result.task_id,
            "status": "completed",
            "output": result.output,
            "acknowledged": result.acknowledged,
            "truncated": result.truncated,
        }

        async def _write():
            await self.client.set(f"task.result:{result.task_id}", json.dumps(data))

        await retry_with_backoff(_write)

    async def aclose(self) -> None:
        await self.client.aclose()

