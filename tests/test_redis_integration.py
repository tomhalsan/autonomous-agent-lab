import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator
from urllib.parse import urlparse

import pytest
import redis.asyncio as redis
from agent_contracts import (
    AgentEvent,
    InferenceExecutionTelemetry,
    TaskRequested,
    TaskResult,
)
from orchestrator.infrastructure.redis_adapter import RedisGateway
from research_agent.domain.evidence import AuthorityTier, EvaluatedEvidence, SourceType
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.redis_bus import RedisBus as ResearchRedisBus
from worker.infrastructure.redis_bus import RedisBus as WorkerRedisBus

pytestmark = [pytest.mark.anyio, pytest.mark.redis]

REQUIRED_TEST_DB = "15"


def _sanitize_redis_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        if parsed.password:
            user = parsed.username or ""
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            userinfo = f"{user}:***@" if user else ":***@"
            masked_netloc = f"{userinfo}{host}{port}"
            return parsed._replace(netloc=masked_netloc).geturl()
        return url
    except ValueError:
        return "<invalid-url>"


def validate_test_redis_url(url: str) -> None:
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValueError(f"Invalid REDIS_URL format: {exc}") from exc

    if parsed.scheme not in ("redis", "rediss"):
        sanitized = _sanitize_redis_url(url)
        raise ValueError(
            f"Safety check failed: REDIS_URL scheme must be 'redis' or 'rediss', got endpoint: {sanitized}"
        )

    db_segment = parsed.path.strip("/")
    if db_segment != REQUIRED_TEST_DB:
        sanitized = _sanitize_redis_url(url)
        raise ValueError(
            f"Safety check failed: Redis integration tests require database {REQUIRED_TEST_DB} "
            f"to prevent accidental data loss, got endpoint: {sanitized}"
        )


@pytest.fixture
def redis_url() -> str:
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        pytest.skip("Redis integration tests disabled: REDIS_URL environment variable is not set")
    try:
        validate_test_redis_url(url)
    except ValueError as exc:
        pytest.fail(str(exc))
    return url


@pytest.fixture
async def clean_redis(redis_url: str) -> AsyncGenerator[str, None]:
    validate_test_redis_url(redis_url)
    client = redis.from_url(redis_url, decode_responses=True)
    sanitized_url = _sanitize_redis_url(redis_url)
    try:
        await asyncio.wait_for(client.ping(), timeout=1.0)
    except (redis.ConnectionError, OSError, TimeoutError) as exc:
        await client.aclose()
        pytest.fail(
            f"Configured Redis endpoint is unreachable at {sanitized_url} within 1.0s timeout: {exc}"
        )

    # Clean before test
    await client.flushdb()
    try:
        yield redis_url
    finally:
        try:
            await client.flushdb()
        finally:
            await client.aclose()


async def test_redis_connection_and_client_lifecycle(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    bus = WorkerRedisBus(clean_redis, queue_name=f"agent.tasks.test_{uuid.uuid4().hex[:8]}")

    try:
        gateway_pong = await gateway._client.ping()
        assert gateway_pong is True

        bus_pong = await bus.client.ping()
        assert bus_pong is True
    finally:
        await gateway.aclose()
        await bus.aclose()


async def test_redis_task_dispatch_and_consumption_round_trip(clean_redis: str) -> None:
    queue_name = f"agent.tasks.test_{uuid.uuid4().hex[:8]}"
    gateway = RedisGateway(clean_redis)
    bus = WorkerRedisBus(clean_redis, queue_name=queue_name)

    try:
        task = TaskRequested.create("Perform integration analysis on real Redis wire protocol")
        task_id = task.task_id

        # 1. Dispatch through RedisGateway (sets task.status:{task_id} and rpush to queue)
        await gateway.dispatch(task, target=queue_name)

        # 2. Verify persisted pending status directly in Redis
        raw_status = await gateway._client.get(f"task.status:{task_id}")
        assert raw_status is not None
        status_data = json.loads(raw_status)
        assert status_data["task_id"] == task_id
        assert status_data["status"] == "pending"

        # 3. Consume through WorkerRedisBus.next_task()
        consumed_task = await asyncio.wait_for(bus.next_task(), timeout=2.0)

        assert consumed_task.task_id == task_id
        assert consumed_task.prompt == "Perform integration analysis on real Redis wire protocol"
        assert consumed_task.created_at == task.created_at
        assert consumed_task.version == task.version

        # 4. Verify queue is now empty
        remaining = await gateway._client.llen(queue_name)
        assert remaining == 0
    finally:
        await gateway.aclose()
        await bus.aclose()


async def test_redis_result_persistence_and_retrieval_round_trip(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    worker_bus = WorkerRedisBus(clean_redis)
    research_bus = ResearchRedisBus(clean_redis)

    try:
        # 1. Worker TaskResult round trip
        worker_task_id = f"task_worker_{uuid.uuid4().hex[:8]}"
        worker_result = TaskResult(
            task_id=worker_task_id,
            status="completed",
            output="General worker finished successfully",
            acknowledged=True,
        )
        await worker_bus.save(worker_result)

        retrieved_worker = await gateway.get_result(worker_task_id)
        assert retrieved_worker is not None
        assert retrieved_worker["task_id"] == worker_task_id
        assert retrieved_worker["status"] == "completed"
        assert retrieved_worker["output"] == "General worker finished successfully"
        assert retrieved_worker["acknowledged"] is True

        # 2. ResearchResult with sources round trip
        evidence = EvaluatedEvidence(
            url="https://example.org/spec",
            title="Specification Doc",
            content="Details on wire protocol",
            domain="example.org",
            authority_tier=AuthorityTier.HIGH,
            source_type=SourceType.OFFICIAL_DOCS,
            quality_signals=("verified", "official"),
            relevance_signal="high",
            recency_indicator="recent",
            rank_score=0.95,
            retained=True,
        )
        research_task_id_with_sources = f"task_research_src_{uuid.uuid4().hex[:8]}"
        research_result_with_sources = ResearchResult(
            task_id=research_task_id_with_sources,
            output="Detailed synthesis output",
            acknowledged=True,
            truncated=False,
            sources=(evidence,),
        )
        await research_bus.save(research_result_with_sources)

        retrieved_research_1 = await gateway.get_result(research_task_id_with_sources)
        assert retrieved_research_1 is not None
        assert retrieved_research_1["task_id"] == research_task_id_with_sources
        assert retrieved_research_1["status"] == "completed"
        assert retrieved_research_1["output"] == "Detailed synthesis output"
        assert retrieved_research_1["acknowledged"] is True
        assert retrieved_research_1["truncated"] is False
        assert "sources" in retrieved_research_1
        assert len(retrieved_research_1["sources"]) == 1
        assert retrieved_research_1["sources"][0]["url"] == "https://example.org/spec"
        assert retrieved_research_1["sources"][0]["domain"] == "example.org"

        # 3. ResearchResult without sources (conditional source omission)
        research_task_id_no_sources = f"task_research_nosrc_{uuid.uuid4().hex[:8]}"
        research_result_no_sources = ResearchResult(
            task_id=research_task_id_no_sources,
            output="Synthesis without sources",
            acknowledged=True,
            truncated=False,
            sources=(),
        )
        await research_bus.save(research_result_no_sources)

        retrieved_research_2 = await gateway.get_result(research_task_id_no_sources)
        assert retrieved_research_2 is not None
        assert retrieved_research_2["task_id"] == research_task_id_no_sources
        assert retrieved_research_2["status"] == "completed"
        assert retrieved_research_2["output"] == "Synthesis without sources"
        assert retrieved_research_2["acknowledged"] is True
        assert "sources" not in retrieved_research_2
    finally:
        await gateway.aclose()
        await worker_bus.aclose()
        await research_bus.aclose()


async def test_redis_event_and_telemetry_round_trip(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    bus = WorkerRedisBus(clean_redis)

    try:
        agent_id = f"worker_{uuid.uuid4().hex[:8]}"
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        telemetry = InferenceExecutionTelemetry(
            provider_name="cloudflare",
            attempt_index=0,
            latency_ms=88.5,
            status="succeeded",
            model_name="@cf/meta/llama-3.1-8b-instruct",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )
        event = AgentEvent.create(
            agent_id=agent_id,
            state="working",
            detail="Inference execution succeeded",
            task_id=task_id,
            inference_telemetry=telemetry,
        )

        # 1. Publish event to Redis stream and agent state
        await bus.publish(event)

        # 2. Query event stream
        events = await gateway.events(limit=10)
        matching_events = [e for e in events if e.get("task_id") == task_id]
        assert len(matching_events) == 1
        published_event = matching_events[0]
        assert published_event["agent_id"] == agent_id
        assert published_event["state"] == "working"
        assert "-" in published_event["event_id"]

        # 3. Query telemetry parsed from Redis stream
        telemetry_result = await gateway.task_inference_telemetry(task_id)
        assert telemetry_result is not None
        assert telemetry_result["task_id"] == task_id
        telem_data = telemetry_result["inference_telemetry"]
        assert telem_data["provider_name"] == "cloudflare"
        assert telem_data["latency_ms"] == 88.5
        assert telem_data["status"] == "succeeded"
        assert telem_data["model_name"] == "@cf/meta/llama-3.1-8b-instruct"

        # 4. Verify agent state discovery
        agents = await gateway.agents()
        matching_agents = [a for a in agents if a.get("agent_id") == agent_id]
        assert len(matching_agents) == 1
        assert matching_agents[0]["state"] == "working"
    finally:
        await gateway.aclose()
        await bus.aclose()
