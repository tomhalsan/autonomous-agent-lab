"""Tests for task-scoped inference fallback attempt history projection (Issue #189, ADR 0005)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from agent_contracts import (
    FallbackTransition,
    InferenceExecutionTelemetry,
)
from orchestrator.application.inference_history import (
    TaskEventObservation,
    project_task_inference_history,
)
from orchestrator.application.ports import TelemetryStorageError
from orchestrator.config import Settings
from orchestrator.presentation.api import create_app, task_telemetry
from starlette.exceptions import HTTPException
from starlette.requests import Request

from tests.test_application import FakeStateReader

# ==============================================================================
# Helper Factories
# ==============================================================================


def _make_observation(
    task_id: str,
    stream_position: tuple[int, int] = (0, 0),
    stream_order: int = 0,
    event_id: str = "1000-0",
    state: str | None = "working",
    telemetry: InferenceExecutionTelemetry | None = None,
    transition: FallbackTransition | None = None,
    is_terminal_lifecycle: bool = False,
    is_terminal_error: bool = False,
    is_malformed: bool = False,
) -> TaskEventObservation:
    return TaskEventObservation(
        event_id=event_id,
        task_id=task_id,
        stream_position=stream_position,
        stream_order=stream_order,
        state=state,
        inference_telemetry=telemetry,
        fallback_transition=transition,
        is_terminal_lifecycle=is_terminal_lifecycle or (state in ("idle", "completed")),
        is_terminal_error=is_terminal_error or (state == "error"),
        is_malformed=is_malformed,
    )


def _make_attempt_obs(
    task_id: str,
    attempt_index: int,
    provider_name: str,
    status: str = "succeeded",
    latency_ms: float = 100.0,
    model_name: str | None = None,
    failure_reason: str | None = None,
    status_code: int | None = None,
    stream_position: tuple[int, int] = (0, 0),
    stream_order: int = 0,
    event_id: str = "1000-0",
    state: str | None = "working",
) -> TaskEventObservation:
    telem = InferenceExecutionTelemetry(
        provider_name=provider_name,
        attempt_index=attempt_index,
        latency_ms=latency_ms,
        status=status,
        model_name=model_name,
        failure_reason=failure_reason,
        status_code=status_code,
    )
    return _make_observation(
        task_id=task_id,
        stream_position=stream_position,
        stream_order=stream_order,
        event_id=event_id,
        state=state,
        telemetry=telem,
    )


def _make_transition_obs(
    task_id: str,
    attempt_index: int,
    failed_provider: str,
    target_provider: str,
    reason: str = "invalid_response",
    stream_position: tuple[int, int] = (0, 0),
    stream_order: int = 0,
    event_id: str = "1000-0",
    state: str | None = "fallback",
) -> TaskEventObservation:
    trans = FallbackTransition(
        failed_provider=failed_provider,
        target_provider=target_provider,
        reason=reason,
        attempt_index=attempt_index,
    )
    return _make_observation(
        task_id=task_id,
        stream_position=stream_position,
        stream_order=stream_order,
        event_id=event_id,
        state=state,
        transition=trans,
    )


# ==============================================================================
# Core Projection Tests
# ==============================================================================


def test_projection_first_attempt_success_with_terminal_completion() -> None:
    task_id = "task-01"
    obs = [
        _make_attempt_obs(
            task_id=task_id,
            attempt_index=0,
            provider_name="cloudflare",
            status="succeeded",
            latency_ms=120.5,
            model_name="@cf/meta/llama-3.1-8b-instruct",
            stream_order=0,
            event_id="1-0",
        ),
        _make_observation(task_id=task_id, stream_order=1, event_id="1-1", state="idle"),
    ]

    res = project_task_inference_history(
        task_id, obs, task_exists_durable=True, task_completed=True
    )
    assert res is not None
    assert res["task_id"] == task_id
    assert res["history_status"] == "complete"
    assert res["final_status"] == "succeeded"
    assert res["inference_telemetry"] is not None
    assert res["inference_telemetry"]["provider_name"] == "cloudflare"
    assert len(res["attempts"]) == 1
    assert len(res["fallback_transitions"]) == 0


def test_projection_succeeded_attempt_no_terminal_evidence_is_partial() -> None:
    task_id = "task-no-terminal-success"
    obs = [
        _make_attempt_obs(
            task_id=task_id,
            attempt_index=0,
            provider_name="cloudflare",
            status="succeeded",
            latency_ms=120.5,
            model_name="@cf/meta/llama-3.1-8b-instruct",
            stream_order=0,
            event_id="1-0",
            state="working",
        )
    ]

    # Without durable task_completed, history must be partial
    res = project_task_inference_history(
        task_id, obs, task_exists_durable=True, task_completed=False
    )
    assert res is not None
    assert res["history_status"] == "partial"
    assert res["final_status"] == "succeeded"
    assert res["inference_telemetry"] is not None


def test_projection_failed_attempt_no_terminal_evidence_is_partial() -> None:
    task_id = "task-no-terminal-failed"
    obs = [
        _make_attempt_obs(
            task_id=task_id,
            attempt_index=0,
            provider_name="cloudflare",
            status="fallback_eligible_failure",
            failure_reason="timeout",
            stream_order=0,
            event_id="1-0",
            state="working",
        )
    ]

    # Without terminal failure evidence, do not infer all_providers_failed; final_status is None, history partial
    res = project_task_inference_history(task_id, obs, task_exists_durable=True, task_failed=False)
    assert res is not None
    assert res["history_status"] == "partial"
    assert res["final_status"] is None
    assert res["inference_telemetry"] is None
    assert len(res["attempts"]) == 1


def test_projection_all_providers_failed_with_terminal_error() -> None:
    task_id = "task-all-failed"
    obs = [
        _make_attempt_obs(
            task_id,
            0,
            "cloudflare",
            "fallback_eligible_failure",
            failure_reason="timeout",
            stream_order=0,
            event_id="1-0",
        ),
        _make_transition_obs(
            task_id,
            0,
            "cloudflare",
            "gemini",
            "timeout",
            stream_order=1,
            event_id="1-1",
        ),
        _make_attempt_obs(
            task_id,
            1,
            "gemini",
            "fallback_eligible_failure",
            failure_reason="rate_limited",
            stream_order=2,
            event_id="1-2",
        ),
        _make_observation(task_id, stream_order=3, event_id="1-3", state="error"),
    ]

    res = project_task_inference_history(task_id, obs, task_exists_durable=True)
    assert res is not None
    assert res["task_id"] == task_id
    assert res["history_status"] == "complete"
    assert res["final_status"] == "all_providers_failed"
    assert res["inference_telemetry"] is None
    assert len(res["attempts"]) == 2
    assert len(res["fallback_transitions"]) == 1


def test_projection_completed_non_inference_task_with_retained_lifecycle() -> None:
    task_id = "task-non-inference"
    obs = [
        _make_observation(task_id=task_id, stream_order=0, event_id="1-0", state="working"),
        _make_observation(task_id=task_id, stream_order=1, event_id="1-1", state="idle"),
    ]

    # Retained lifecycle events in stream + completed task -> no_inference_recorded (NOT aged_out)
    res = project_task_inference_history(
        task_id, obs, task_exists_durable=True, task_completed=True
    )
    assert res is not None
    assert res["task_id"] == task_id
    assert res["history_status"] == "no_inference_recorded"
    assert res["final_status"] == "succeeded"
    assert res["inference_telemetry"] is None
    assert res["attempts"] == []
    assert res["fallback_transitions"] == []


def test_projection_aged_out_classification_semantics() -> None:
    # 1. Routing-only known task, no events => no_inference_recorded, final_status=None
    res_routing = project_task_inference_history(
        "task-routing-only",
        [],
        task_exists_durable=True,
        task_completed=False,
        task_failed=False,
    )
    assert res_routing is not None
    assert res_routing["history_status"] == "no_inference_recorded"
    assert res_routing["final_status"] is None

    # 2. Pending-status known task, no events => no_inference_recorded, final_status=None
    res_pending = project_task_inference_history(
        "task-pending",
        [],
        task_exists_durable=True,
        task_completed=False,
        task_failed=False,
    )
    assert res_pending is not None
    assert res_pending["history_status"] == "no_inference_recorded"
    assert res_pending["final_status"] is None

    # 3. Completed result, no events => aged_out, final_status=succeeded
    res_completed = project_task_inference_history(
        "task-completed",
        [],
        task_exists_durable=True,
        task_completed=True,
        task_failed=False,
    )
    assert res_completed is not None
    assert res_completed["history_status"] == "aged_out"
    assert res_completed["final_status"] == "succeeded"

    # 4. Durable failed status, no events => aged_out, final_status=failed
    res_failed = project_task_inference_history(
        "task-failed",
        [],
        task_exists_durable=True,
        task_completed=False,
        task_failed=True,
    )
    assert res_failed is not None
    assert res_failed["history_status"] == "aged_out"
    assert res_failed["final_status"] == "failed"


def test_projection_zero_attempts_error_then_idle_precedence() -> None:
    task_id = "task-err-idle"
    # Worker publishes error, then transitions agent back to idle
    obs = [
        _make_observation(task_id, stream_order=0, event_id="1-0", state="error"),
        _make_observation(task_id, stream_order=1, event_id="1-1", state="idle"),
    ]

    res = project_task_inference_history(task_id, obs, task_exists_durable=True)
    assert res is not None
    assert res["final_status"] == "failed"
    assert res["history_status"] == "no_inference_recorded"


def test_projection_zero_attempts_idle_without_durable_result_does_not_prove_completion() -> None:
    task_id = "task-idle-no-result"
    # Idle state observed, but durable task.result does not exist
    obs = [
        _make_observation(task_id, stream_order=0, event_id="1-0", state="idle"),
    ]

    res = project_task_inference_history(
        task_id, obs, task_exists_durable=True, task_completed=False
    )
    assert res is not None
    assert res["final_status"] is None
    assert res["history_status"] == "no_inference_recorded"


def test_projection_unknown_task_returns_none() -> None:
    task_id = "task-unknown"
    # Neither stream events nor durable records exist
    res = project_task_inference_history(task_id, [], task_exists_durable=False)
    assert res is None


def test_projection_fallback_success_complete_lifecycle() -> None:
    task_id = "task-fallback-success"
    obs = [
        _make_attempt_obs(
            task_id,
            0,
            "cloudflare",
            "fallback_eligible_failure",
            failure_reason="invalid_response",
            stream_order=0,
            event_id="1-0",
        ),
        _make_transition_obs(
            task_id,
            0,
            "cloudflare",
            "gemini",
            "invalid_response",
            stream_order=1,
            event_id="1-1",
        ),
        _make_attempt_obs(
            task_id,
            1,
            "gemini",
            "succeeded",
            model_name="gemini-2.5-flash",
            stream_order=2,
            event_id="1-2",
        ),
        _make_observation(task_id=task_id, stream_order=3, event_id="1-3", state="idle"),
    ]

    res = project_task_inference_history(
        task_id, obs, task_exists_durable=True, task_completed=True
    )
    assert res is not None
    assert res["history_status"] == "complete"
    assert res["final_status"] == "succeeded"
    assert res["inference_telemetry"]["provider_name"] == "gemini"
    assert len(res["attempts"]) == 2
    assert len(res["fallback_transitions"]) == 1


def test_projection_duplicate_events_deduplicated() -> None:
    task_id = "task-dedup"
    att0_a = _make_attempt_obs(
        task_id,
        0,
        "cloudflare",
        "succeeded",
        latency_ms=100.0,
        stream_order=0,
        event_id="1-0",
    )
    att0_b = _make_attempt_obs(
        task_id,
        0,
        "cloudflare",
        "succeeded",
        latency_ms=100.0,
        stream_order=1,
        event_id="1-1",
    )
    lifecycle = _make_observation(task_id=task_id, stream_order=2, event_id="1-2", state="idle")

    res = project_task_inference_history(
        task_id, [att0_a, att0_b, lifecycle], task_exists_durable=True, task_completed=True
    )
    assert res is not None
    assert res["history_status"] == "complete"
    assert len(res["attempts"]) == 1


def test_projection_conflicting_same_index_attempt_marks_partial() -> None:
    task_id = "task-conflict"
    att0_a = _make_attempt_obs(
        task_id,
        0,
        "cloudflare",
        "succeeded",
        latency_ms=100.0,
        stream_order=0,
        event_id="1-0",
    )
    att0_b = _make_attempt_obs(
        task_id,
        0,
        "gemini",
        "succeeded",
        latency_ms=200.0,
        stream_order=1,
        event_id="1-1",
    )
    lifecycle = _make_observation(task_id=task_id, stream_order=2, event_id="1-2", state="idle")

    res = project_task_inference_history(
        task_id, [att0_a, att0_b, lifecycle], task_exists_durable=True, task_completed=True
    )
    assert res is not None
    assert res["history_status"] == "partial"
    assert len(res["attempts"]) == 2


def test_projection_attempt_index_gap_marks_partial() -> None:
    task_id = "task-gap"
    att1 = _make_attempt_obs(task_id, 1, "gemini", "succeeded", stream_order=0, event_id="1-0")
    lifecycle = _make_observation(task_id=task_id, stream_order=1, event_id="1-1", state="idle")

    res = project_task_inference_history(
        task_id, [att1, lifecycle], task_exists_durable=True, task_completed=True
    )
    assert res is not None
    assert res["history_status"] == "partial"


def test_projection_unanchored_transition_marks_partial() -> None:
    task_id = "task-unanchored"
    att0 = _make_attempt_obs(
        task_id,
        0,
        "cloudflare",
        "fallback_eligible_failure",
        failure_reason="timeout",
        stream_order=0,
        event_id="1-0",
    )
    tr0 = _make_transition_obs(
        task_id, 0, "cloudflare", "gemini", "timeout", stream_order=1, event_id="1-1"
    )
    err = _make_observation(task_id=task_id, stream_order=2, event_id="1-2", state="error")

    res = project_task_inference_history(task_id, [att0, tr0, err], task_exists_durable=True)
    assert res is not None
    assert res["history_status"] == "partial"


def test_projection_malformed_event_marks_partial() -> None:
    task_id = "task-malformed"
    att0 = _make_attempt_obs(task_id, 0, "gemini", "succeeded", stream_order=0, event_id="1-0")
    malformed = _make_observation(task_id, stream_order=1, event_id="1-1", is_malformed=True)
    lifecycle = _make_observation(task_id, stream_order=2, event_id="1-2", state="idle")

    res = project_task_inference_history(
        task_id, [att0, malformed, lifecycle], task_exists_durable=True, task_completed=True
    )
    assert res is not None
    assert res["history_status"] == "partial"


def test_projection_preserves_none_model_name() -> None:
    task_id = "task-none-model"
    att0 = _make_attempt_obs(
        task_id,
        0,
        "cloudflare",
        "fallback_eligible_failure",
        failure_reason="invalid_response",
        model_name=None,
        stream_order=0,
        event_id="1-0",
    )
    err = _make_observation(task_id, stream_order=1, event_id="1-1", state="error")

    res = project_task_inference_history(task_id, [att0, err], task_exists_durable=True)
    assert res is not None
    assert res["attempts"][0]["model_name"] is None


def test_projection_chronological_tie_break_using_normalized_stream_position() -> None:
    task_id = "task-chrono"
    # Observations delivered out of order: position (1000, 2), position (1000, 1), position (1000, 0)
    att1 = _make_attempt_obs(
        task_id,
        1,
        "gemini",
        "succeeded",
        model_name="gemini-2.5-flash",
        stream_position=(1000, 2),
        stream_order=2,
        event_id="1000-2",
    )
    tr0 = _make_transition_obs(
        task_id,
        0,
        "cloudflare",
        "gemini",
        "timeout",
        stream_position=(1000, 1),
        stream_order=1,
        event_id="1000-1",
    )
    att0 = _make_attempt_obs(
        task_id,
        0,
        "cloudflare",
        "fallback_eligible_failure",
        failure_reason="timeout",
        stream_position=(1000, 0),
        stream_order=0,
        event_id="1000-0",
    )
    lifecycle = _make_observation(
        task_id,
        stream_position=(1000, 3),
        stream_order=3,
        event_id="1000-3",
        state="idle",
    )

    res = project_task_inference_history(
        task_id,
        [att1, tr0, att0, lifecycle],
        task_exists_durable=True,
        task_completed=True,
    )
    assert res is not None
    assert res["history_status"] == "complete"
    assert res["attempts"][0]["attempt_index"] == 0
    assert res["attempts"][1]["attempt_index"] == 1
    assert res["attempts"][0]["provider_name"] == "cloudflare"
    assert res["attempts"][1]["provider_name"] == "gemini"


def test_projection_security_boundary_no_detail_accepted() -> None:
    # TaskEventObservation does not accept or have detail field
    task_id = "task-sec"
    obs = _make_attempt_obs(task_id, 0, "gemini", "succeeded")
    assert not hasattr(obs, "detail")


# ==============================================================================
# API Endpoint Integration Tests
# ==============================================================================


@pytest.mark.anyio
async def test_api_telemetry_returns_200_with_attempts_on_terminal_failure() -> None:
    app = create_app(Settings())
    task_id = "task-api-fail"
    telem_cf = InferenceExecutionTelemetry(
        provider_name="cloudflare",
        attempt_index=0,
        latency_ms=150.0,
        status="fallback_eligible_failure",
        failure_reason="invalid_response",
    )
    trans = FallbackTransition(
        failed_provider="cloudflare",
        target_provider="gemini",
        reason="invalid_response",
        attempt_index=0,
    )
    telem_gemini = InferenceExecutionTelemetry(
        provider_name="gemini",
        attempt_index=1,
        latency_ms=210.0,
        status="fallback_eligible_failure",
        failure_reason="rate_limited",
    )
    events_data = [
        {
            "event_id": "1-0",
            "task_id": task_id,
            "inference_telemetry": json.dumps(telem_cf.to_dict()),
        },
        {
            "event_id": "1-1",
            "task_id": task_id,
            "fallback_transition": json.dumps(trans.to_dict()),
        },
        {
            "event_id": "1-2",
            "task_id": task_id,
            "inference_telemetry": json.dumps(telem_gemini.to_dict()),
        },
        {
            "event_id": "1-3",
            "task_id": task_id,
            "state": "error",
        },
    ]

    reader = FakeStateReader(events_data=events_data, task_statuses={task_id: "failed"})
    scope = {"type": "http", "app": app}
    request = Request(scope)
    request.app.state.get_task_inference_telemetry = AsyncMock()
    request.app.state.get_task_inference_telemetry.execute = AsyncMock(
        side_effect=reader.task_inference_telemetry
    )

    result = await task_telemetry(task_id, request)
    assert result["task_id"] == task_id
    assert result["inference_telemetry"] is None
    assert result["final_status"] == "all_providers_failed"
    assert result["history_status"] == "complete"
    assert len(result["attempts"]) == 2
    assert len(result["fallback_transitions"]) == 1


@pytest.mark.anyio
async def test_api_telemetry_returns_404_for_unknown_task() -> None:
    app = create_app(Settings())
    reader = FakeStateReader(events_data=[])
    scope = {"type": "http", "app": app}
    request = Request(scope)
    request.app.state.get_task_inference_telemetry = AsyncMock()
    request.app.state.get_task_inference_telemetry.execute = AsyncMock(
        side_effect=reader.task_inference_telemetry
    )

    with pytest.raises(HTTPException) as exc_info:
        await task_telemetry("task-unknown", request)
    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_api_telemetry_event_stream_read_failure_raises_503() -> None:
    app = create_app(Settings())
    scope = {"type": "http", "app": app}
    request = Request(scope)
    request.app.state.get_task_inference_telemetry = AsyncMock()
    request.app.state.get_task_inference_telemetry.execute = AsyncMock(
        side_effect=TelemetryStorageError("Event stream storage unavailable: ConnectionError")
    )

    with pytest.raises(HTTPException) as exc_info:
        await task_telemetry("task-stream-fail", request)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Task telemetry storage unavailable"


@pytest.mark.anyio
async def test_api_telemetry_durable_key_read_failure_raises_503() -> None:
    app = create_app(Settings())
    scope = {"type": "http", "app": app}
    request = Request(scope)
    request.app.state.get_task_inference_telemetry = AsyncMock()
    request.app.state.get_task_inference_telemetry.execute = AsyncMock(
        side_effect=TelemetryStorageError("Task durable state storage unavailable: RedisError")
    )

    with pytest.raises(HTTPException) as exc_info:
        await task_telemetry("task-durable-fail", request)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Task telemetry storage unavailable"


@pytest.mark.anyio
async def test_api_telemetry_backward_compatible_winning_telemetry() -> None:
    app = create_app(Settings())
    task_id = "task-compat"
    telem = InferenceExecutionTelemetry(
        provider_name="gemini",
        attempt_index=0,
        latency_ms=95.0,
        status="succeeded",
        model_name="gemini-2.5-flash",
    )
    events_data = [
        {
            "event_id": "1-0",
            "task_id": task_id,
            "inference_telemetry": json.dumps(telem.to_dict()),
        },
        {
            "event_id": "1-1",
            "task_id": task_id,
            "state": "idle",
        },
    ]

    reader = FakeStateReader(events_data=events_data, task_results={task_id: {"output": "ok"}})
    scope = {"type": "http", "app": app}
    request = Request(scope)
    request.app.state.get_task_inference_telemetry = AsyncMock()
    request.app.state.get_task_inference_telemetry.execute = AsyncMock(
        side_effect=reader.task_inference_telemetry
    )

    result = await task_telemetry(task_id, request)
    assert result["task_id"] == task_id
    assert result["inference_telemetry"]["provider_name"] == "gemini"
    assert result["inference_telemetry"]["status"] == "succeeded"
    assert result["final_status"] == "succeeded"
    assert result["history_status"] == "complete"
