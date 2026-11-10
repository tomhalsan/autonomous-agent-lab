import asyncio
import logging
import os
import sys

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    HeartbeatRunner,
    parse_health_probe_timeout,
    parse_heartbeat_interval,
)
from agent_policy import parse_provider_priority

from worker.application.fallback import WorkerFallbackCoordinator
from worker.application.ports import RuntimeInferenceProvider
from worker.application.processor import TaskExecutionError, TaskProcessor
from worker.infrastructure.inference_factory import create_inference_provider
from worker.infrastructure.redis_bus import RedisBus

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("worker")


async def run_worker(
    heartbeat_interval: float | None = None,
    health_probe_timeout: float | None = None,
) -> None:
    agent_id = os.getenv("AGENT_ID", "worker-1")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    task_queue = os.getenv("TASK_QUEUE", "agent.tasks.general")
    inference_provider_name = os.getenv("INFERENCE_PROVIDER", "stub")
    interval = (
        heartbeat_interval
        if heartbeat_interval is not None
        else parse_heartbeat_interval(os.getenv("HEARTBEAT_INTERVAL_SECONDS"))
    )
    probe_timeout = (
        health_probe_timeout
        if health_probe_timeout is not None
        else parse_health_probe_timeout(os.getenv("HEALTH_PROBE_TIMEOUT_SECONDS"))
    )

    preferences = parse_provider_priority(inference_provider_name, default=("stub",))
    providers: dict[str, RuntimeInferenceProvider] = {
        name: create_inference_provider(name) for name in preferences
    }

    bus = RedisBus(redis_url, queue_name=task_queue)
    fallback_coordinator = WorkerFallbackCoordinator(
        agent_id=agent_id,
        bus=bus,
        preferences=preferences,
        providers=providers,
        probe_timeout_seconds=probe_timeout,
    )
    processor = TaskProcessor(
        agent_id,
        bus,
        result_store=bus,
        inference_provider=providers[preferences[0]],
        fallback_coordinator=fallback_coordinator,
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

    heartbeat_runner = HeartbeatRunner(
        agent_id=agent_id,
        publish_heartbeat=bus.heartbeat,
        get_state=lambda: bus.current_state,
        interval_seconds=interval,
    )

    logger.info(
        "Initializing worker %s on queue %s with inference provider %s (heartbeat: %.1fs, probe_timeout: %.1fs)...",
        agent_id,
        task_queue,
        inference_provider_name,
        interval,
        probe_timeout,
    )

    try:
        await bus.register(metadata)
        await bus.publish(AgentEvent.create(agent_id, "idle", "Waiting for work"))
        heartbeat_runner.start()
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
        await heartbeat_runner.stop()
        await bus.aclose()
        logger.info("Worker %s shutdown complete", agent_id)


def main() -> None:
    try:
        asyncio.run(run_worker())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Worker process interrupted. Exiting.")


if __name__ == "__main__":
    main()
