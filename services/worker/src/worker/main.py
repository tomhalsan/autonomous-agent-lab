import asyncio
import logging
import os
import sys

from agent_contracts import AgentCapabilityMetadata, AgentEvent

from worker.application.processor import TaskExecutionError, TaskProcessor
from worker.infrastructure.inference_factory import create_inference_provider
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
    task_queue = os.getenv("TASK_QUEUE", "agent.tasks.general")
    inference_provider_name = os.getenv("INFERENCE_PROVIDER", "stub")

    inference_provider = create_inference_provider(inference_provider_name)
    bus = RedisBus(redis_url, queue_name=task_queue)
    processor = TaskProcessor(
        agent_id,
        bus,
        result_store=bus,
        inference_provider=inference_provider,
    )

    metadata = AgentCapabilityMetadata(
        agent_id=agent_id,
        agent_type="worker",
        capabilities=["general"],
        providers={
            "inference_provider": inference_provider_name,
        },
        dispatch_target=task_queue,
    )

    logger.info(
        "Initializing worker %s on queue %s with inference provider %s...",
        agent_id,
        task_queue,
        inference_provider_name,
    )

    try:
        await bus.register(metadata)
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
