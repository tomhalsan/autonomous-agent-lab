"""Dedicated sentinel security tests for AgentEvent.detail sanitization (Issue #191).

Verifies that lifecycle AgentEvent.detail values across Worker and Research Agent
processors are sanitized to fixed, non-sensitive strings and do not leak:
- raw task prompts,
- generated task or research synthesis outputs,
- raw exception strings, database messages, or credentials.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from agent_contracts import AgentEvent, TaskRequested, TaskResult
from research_agent.application.processor import (
    ResearchExecutionError,
    ResearchProcessor,
)
from research_agent.domain.models import ResearchResult
from worker.application.processor import (
    TaskExecutionError,
    TaskProcessor,
)


class MemoryBus:
    def __init__(self) -> None:
        self.published: list[AgentEvent] = []

    async def publish(self, event: AgentEvent) -> None:
        self.published.append(event)


class MemoryResultStore:
    def __init__(self) -> None:
        self.saved: dict[str, Any] = {}

    async def save(self, result: Any) -> None:
        self.saved[result.task_id] = result


class FakeFailingWorkerProcessor(TaskProcessor):
    def __init__(self, agent_id: str, bus: MemoryBus, exc: Exception) -> None:
        super().__init__(agent_id=agent_id, bus=bus)  # type: ignore[arg-type]
        self._exc = exc

    async def process(self, task: TaskRequested) -> TaskResult:
        raise self._exc


class CustomOutputResearchExecutor:
    def __init__(self, output: str) -> None:
        self._output = output

    async def execute(self, task: TaskRequested) -> ResearchResult:
        return ResearchResult(task_id=task.task_id, output=self._output)


class FailingResearchExecutor:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def execute(self, task: TaskRequested) -> ResearchResult:
        raise self._exc


@pytest.mark.anyio
async def test_worker_task_processor_sanitizes_lifecycle_events_on_success() -> None:
    bus = MemoryBus()
    store = MemoryResultStore()
    processor = TaskProcessor("worker-security-1", bus)  # type: ignore[arg-type]
    processor.result_store = store  # type: ignore[assignment]

    prompt_secret = "RAW-PROMPT-SECRET-WORKER-99-API-KEY-ABCDEF123456"
    task = TaskRequested.create(prompt_secret)

    await processor.run_once(task)

    # 1. Result store preserves genuine execution result
    assert task.task_id in store.saved
    assert prompt_secret in store.saved[task.task_id].output

    # 2. Exactly 2 lifecycle events published: working -> idle
    assert len(bus.published) == 2
    ev_working, ev_idle = bus.published

    assert ev_working.state == "working"
    assert ev_working.task_id == task.task_id
    assert ev_working.detail == "Processing task"

    assert ev_idle.state == "idle"
    assert ev_idle.task_id == task.task_id
    assert ev_idle.detail == "Task completed"

    # 3. Sentinel security check: prompt secret MUST NOT appear in event details
    for ev in bus.published:
        assert prompt_secret not in ev.detail
        assert "RAW-PROMPT-SECRET" not in ev.detail

        # Serialization checks
        payload_dict = ev.to_dict()
        assert prompt_secret not in str(payload_dict["detail"])
        payload_json = ev.to_json()
        deserialized = json.loads(payload_json)
        assert prompt_secret not in str(deserialized.get("detail", ""))


@pytest.mark.anyio
async def test_worker_task_processor_sanitizes_lifecycle_events_on_failure() -> None:
    bus = MemoryBus()
    store = MemoryResultStore()
    raw_error_message = (
        "FATAL-DATABASE-EXCEPTION-WORKER-77: Connection to postgres://user:SECRET_PASS_99@internal:5432 failed"
    )
    processor = FakeFailingWorkerProcessor("worker-security-1", bus, RuntimeError(raw_error_message))
    processor.result_store = store  # type: ignore[assignment]

    prompt_secret = "WORKER-FAILING-TASK-PROMPT-CONFIDENTIAL-01"
    task = TaskRequested.create(prompt_secret)

    with pytest.raises(TaskExecutionError) as exc_info:
        await processor.run_once(task)

    # 1. Original exception is preserved internally in TaskExecutionError
    assert exc_info.value.task_id == task.task_id
    assert raw_error_message in str(exc_info.value.original_error)

    # 2. Result store is empty on failure
    assert len(store.saved) == 0

    # 3. Exactly 3 lifecycle events published: working -> error -> idle (availability restored)
    assert len(bus.published) == 3
    ev_working, ev_error, ev_idle = bus.published

    assert ev_working.state == "working"
    assert ev_working.task_id == task.task_id
    assert ev_working.detail == "Processing task"

    assert ev_error.state == "error"
    assert ev_error.task_id == task.task_id
    assert ev_error.detail == "Task failed"

    assert ev_idle.state == "idle"
    assert ev_idle.detail == "Waiting for work"

    # 4. Sentinel security check: neither prompt nor exception message leaks to event details
    for ev in bus.published:
        assert prompt_secret not in ev.detail
        assert "FATAL-DATABASE-EXCEPTION" not in ev.detail
        assert "SECRET_PASS_99" not in ev.detail

        # Serialization checks
        payload_dict = ev.to_dict()
        assert "FATAL-DATABASE-EXCEPTION" not in str(payload_dict["detail"])
        assert "SECRET_PASS_99" not in str(payload_dict["detail"])
        assert prompt_secret not in str(payload_dict["detail"])


@pytest.mark.anyio
async def test_research_processor_sanitizes_lifecycle_events_on_success() -> None:
    bus = MemoryBus()
    store = MemoryResultStore()
    synthesis_secret = "SYNTHESIS-SECRET-RESEARCH-44: Proprietary algorithm analysis"
    evidence_secret = "EVIDENCE-RAW-EXCERPT-55: Private document snippet text"
    executor = CustomOutputResearchExecutor(
        output=f"{synthesis_secret}\n{evidence_secret}"
    )
    processor = ResearchProcessor("research-security-1", bus, executor)  # type: ignore[arg-type]
    processor.result_store = store  # type: ignore[assignment]

    prompt_secret = "CONFIDENTIAL-RESEARCH-PROMPT-TOPIC-77"
    task = TaskRequested.create(prompt_secret)

    await processor.run_once(task)

    # 1. Result store preserves genuine execution result
    assert task.task_id in store.saved
    assert synthesis_secret in store.saved[task.task_id].output
    assert evidence_secret in store.saved[task.task_id].output

    # 2. Exactly 2 lifecycle events published: working -> idle
    assert len(bus.published) == 2
    ev_working, ev_idle = bus.published

    assert ev_working.state == "working"
    assert ev_working.task_id == task.task_id
    assert ev_working.detail == "Processing task"

    assert ev_idle.state == "idle"
    assert ev_idle.task_id == task.task_id
    assert ev_idle.detail == "Task completed"

    # 3. Sentinel security check: neither prompt nor synthesis/evidence leaks to event details
    for ev in bus.published:
        assert prompt_secret not in ev.detail
        assert "CONFIDENTIAL-RESEARCH-PROMPT" not in ev.detail
        assert synthesis_secret not in ev.detail
        assert "SYNTHESIS-SECRET" not in ev.detail
        assert evidence_secret not in ev.detail
        assert "EVIDENCE-RAW" not in ev.detail

        # Serialization checks
        payload_dict = ev.to_dict()
        assert synthesis_secret not in str(payload_dict["detail"])
        assert prompt_secret not in str(payload_dict["detail"])
        payload_json = ev.to_json()
        deserialized = json.loads(payload_json)
        assert synthesis_secret not in str(deserialized.get("detail", ""))


@pytest.mark.anyio
async def test_research_processor_sanitizes_lifecycle_events_on_failure() -> None:
    bus = MemoryBus()
    store = MemoryResultStore()
    raw_error_message = (
        "RESEARCH-API-FAILURE-33: HTTP 403 Forbidden with Bearer token eyJhbGciOi...SECRET_BEARER_TOKEN"
    )
    executor = FailingResearchExecutor(RuntimeError(raw_error_message))
    processor = ResearchProcessor("research-security-1", bus, executor)  # type: ignore[arg-type]
    processor.result_store = store  # type: ignore[assignment]

    prompt_secret = "CONFIDENTIAL-FAILING-RESEARCH-TASK-PROMPT-88"
    task = TaskRequested.create(prompt_secret)

    with pytest.raises(ResearchExecutionError) as exc_info:
        await processor.run_once(task)

    # 1. Original exception is preserved internally in ResearchExecutionError
    assert exc_info.value.task_id == task.task_id
    assert raw_error_message in str(exc_info.value.original_error)

    # 2. Result store is empty on failure
    assert len(store.saved) == 0

    # 3. Exactly 2 lifecycle events published: working -> error (ResearchAgent maintains error state)
    assert len(bus.published) == 2
    ev_working, ev_error = bus.published

    assert ev_working.state == "working"
    assert ev_working.task_id == task.task_id
    assert ev_working.detail == "Processing task"

    assert ev_error.state == "error"
    assert ev_error.task_id == task.task_id
    assert ev_error.detail == "Task failed"

    # 4. Sentinel security check: neither prompt nor exception message leaks to event details
    for ev in bus.published:
        assert prompt_secret not in ev.detail
        assert "RESEARCH-API-FAILURE" not in ev.detail
        assert "SECRET_BEARER_TOKEN" not in ev.detail

        # Serialization checks
        payload_dict = ev.to_dict()
        assert "RESEARCH-API-FAILURE" not in str(payload_dict["detail"])
        assert "SECRET_BEARER_TOKEN" not in str(payload_dict["detail"])
        assert prompt_secret not in str(payload_dict["detail"])
