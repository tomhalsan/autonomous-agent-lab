import asyncio
import logging
import os
import sys

from agent_contracts import AgentEvent

from worker.application.processor import TaskExecutionError, TaskProcessor
from worker.infrastructure.redis_bus import RedisBus

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("worker")


async def run_worker() -> None:
    agent_id = os.getenv("AGENT_ID", "worker-1")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    bus = RedisBus(redis_url)
    processor = TaskProcessor(agent_id, bus)

    logger.info("Initializing worker %s...", agent_id)

    try:
        await bus.publish(AgentEvent.create(agent_id, "idle", "Waiting for work"))
        logger.info("Worker %s registered and waiting for tasks", agent_id)

        while True:
            task = await bus.next_task()
            logger.info("Dispatching task %s to processor", task.task_id)
            try:
                await processor.run_once(task)
            except TaskExecutionError as exc:
                logger.warning(
                    "Task %s processing failed with error: %s; ready for next task",
                    exc.task_id,
                    exc.original_error,
                )
    finally:
        logger.info("Shutting down worker %s, closing bus...", agent_id)
        await bus.aclose()
        logger.info("Worker %s shutdown complete", agent_id)


def main() -> None:
    try:
        asyncio.run(run_worker())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Worker process interrupted. Exiting.")


if __name__ == "__main__":
    main()
