import asyncio
import os
from agent_contracts import AgentEvent
from worker.application.processor import TaskProcessor
from worker.infrastructure.redis_bus import RedisBus

async def main() -> None:
    agent_id = os.getenv("AGENT_ID", "worker-1")
    bus = RedisBus(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    processor = TaskProcessor(agent_id, bus)
    await bus.publish(AgentEvent.create(agent_id, "idle", "Waiting for work"))
    while True:
        task = await bus.next_task()
        try:
            await processor.run_once(task)
        except Exception as exc:
            await bus.publish(AgentEvent.create(agent_id, "error", str(exc), task.task_id))

if __name__ == "__main__":
    asyncio.run(main())
