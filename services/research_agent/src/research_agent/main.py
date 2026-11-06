import asyncio
import logging
import os
import sys

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    HeartbeatRunner,
    parse_heartbeat_interval,
)

from research_agent.application.processor import ResearchExecutionError, ResearchProcessor
from research_agent.infrastructure.executor_factory import create_research_executor
from research_agent.infrastructure.inference_factory import create_inference_provider
from research_agent.infrastructure.redis_bus import RedisBus

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("research_agent")


async def run_research_agent(heartbeat_interval: float | None = None) -> None:
    agent_id = os.getenv("AGENT_ID", "research-agent-1")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    task_queue = os.getenv("TASK_QUEUE", "agent.tasks.research")
    executor_name = os.getenv("RESEARCH_EXECUTOR", "stub")
    inference_provider_name = os.getenv("INFERENCE_PROVIDER", "stub")
    interval = (
        heartbeat_interval
        if heartbeat_interval is not None
        else parse_heartbeat_interval(os.getenv("HEARTBEAT_INTERVAL_SECONDS"))
    )

    executor = create_research_executor(executor_name)
    inference_provider = create_inference_provider(inference_provider_name)
    bus = RedisBus(redis_url, queue_name=task_queue)
    processor = ResearchProcessor(
        agent_id,
        bus,
        executor,
        result_store=bus,
        inference_provider=inference_provider,
    )

    metadata = AgentCapabilityMetadata(
        agent_id=agent_id,
        agent_type="research-agent",
        capabilities=["research"],
        providers={
            "research_executor": executor_name,
            "inference_provider": inference_provider_name,
        },
        dispatch_target=task_queue,
    )

    heartbeat_runner = HeartbeatRunner(
        agent_id=agent_id,
        publish_heartbeat=bus.heartbeat,
        get_state=lambda: bus.current_state,
        interval_seconds=interval,
    )

    logger.info(
        "Initializing research agent %s on queue %s with executor %s and inference provider %s (heartbeat: %.1fs)...",
        agent_id,
        task_queue,
        executor_name,
        inference_provider_name,
        interval,
    )

    try:
        await bus.register(metadata)
        await bus.publish(AgentEvent.create(agent_id, "idle", "Waiting for work"))
        heartbeat_runner.start()
        logger.info("Research agent %s registered and waiting for tasks", agent_id)

        while True:
            task = await bus.next_task()
            logger.info("Dispatching task %s to research processor", task.task_id)
            try:
                await processor.run_once(task)
            except ResearchExecutionError as exc:
                logger.warning(
                    "Research task %s processing failed with error: %s; ready for next task",
                    exc.task_id,
                    exc.original_error,
                )
    finally:
        logger.info("Shutting down research agent %s, closing bus...", agent_id)
        await heartbeat_runner.stop()
        await bus.aclose()
        logger.info("Research agent %s shutdown complete", agent_id)


def main() -> None:
    try:
        asyncio.run(run_research_agent())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Research agent process interrupted. Exiting.")


if __name__ == "__main__":
    main()
