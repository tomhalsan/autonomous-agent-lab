import asyncio
import logging
import os
import sys

from agent_contracts import AgentEvent

from research_agent.application.processor import ResearchExecutionError, ResearchProcessor
from research_agent.infrastructure.redis_bus import RedisBus
from research_agent.infrastructure.stub_executor import StubResearchExecutor

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("research_agent")


async def run_research_agent() -> None:
    agent_id = os.getenv("AGENT_ID", "research-agent-1")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    task_queue = os.getenv("TASK_QUEUE", "agent.tasks.research")

    bus = RedisBus(redis_url, queue_name=task_queue)
    executor = StubResearchExecutor()
    processor = ResearchProcessor(agent_id, bus, executor)

    logger.info("Initializing research agent %s on queue %s...", agent_id, task_queue)

    try:
        await bus.publish(AgentEvent.create(agent_id, "idle", "Waiting for work"))
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
        await bus.aclose()
        logger.info("Research agent %s shutdown complete", agent_id)


def main() -> None:
    try:
        asyncio.run(run_research_agent())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Research agent process interrupted. Exiting.")


if __name__ == "__main__":
    main()

