import json

import pytest
from fastapi import HTTPException
from orchestrator.config import Settings
from orchestrator.domain.capability import AgentCapability
from orchestrator.presentation.api import (
    TaskInput,
    agents,
    create_app,
    create_task,
    dashboard,
    events,
    health,
    task_result,
    task_telemetry,
)
from pydantic import ValidationError
from starlette.requests import Request

from tests.test_application import (
    FakeResultReader,
    FakeStateReader,
    FakeTaskQueue,
)


def test_task_input_rejects_empty_prompt() -> None:
    with pytest.raises(ValidationError):
        TaskInput(prompt="")


def test_task_input_accepts_valid_prompt() -> None:
    item = TaskInput(prompt="Hello Agent")
    assert item.prompt == "Hello Agent"


@pytest.mark.anyio
async def test_dashboard_html_binds_prompt_element_explicitly() -> None:
    html = await dashboard()
    assert '<form id="task">' in html
    assert "document.getElementById('task')" in html or 'document.getElementById("task")' in html
    assert "document.getElementById('prompt')" in html or 'document.getElementById("prompt")' in html
    assert "taskForm.onsubmit" in html
    assert "promptInput.value" in html
    # Ensure window.prompt and global element IDs are not accidentally referenced
    assert "JSON.stringify({prompt:prompt.value})" not in html
    assert "JSON.stringify({prompt: prompt.value})" not in html
    assert "task.onsubmit" not in html


@pytest.mark.anyio
async def test_dashboard_displays_authoritative_version() -> None:
    html = await dashboard()
    assert "v0.3.0" in html
    assert '<span class="version-badge">v0.3.0</span>' in html


@pytest.mark.anyio
async def test_dashboard_displays_primary_logo() -> None:
    html = await dashboard()
    assert '<img src="/static/logo.png"' in html
    assert 'class="brand-logo"' in html
    assert 'alt="Autonomous Agent Lab logo"' in html
    assert 'class="brand-group"' in html


@pytest.mark.anyio
async def test_static_logo_endpoint_serves_image() -> None:
    import httpx

    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        res = await client.get("/static/logo.png")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        assert len(res.content) > 0


@pytest.mark.anyio
async def test_dashboard_references_favicon() -> None:
    html = await dashboard()
    assert 'rel="icon" type="image/png" sizes="32x32" href="/static/favicon-32x32.png"' in html
    assert 'rel="icon" type="image/png" sizes="16x16" href="/static/favicon-16x16.png"' in html
    assert 'rel="shortcut icon" href="/static/favicon.ico"' in html
    assert 'rel="apple-touch-icon" sizes="180x180" href="/static/apple-touch-icon.png"' in html


@pytest.mark.anyio
async def test_favicon_endpoints_serve_assets() -> None:
    import httpx

    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Static mount serving
        res_png = await client.get("/static/favicon-32x32.png")
        assert res_png.status_code == 200
        assert res_png.headers["content-type"] == "image/png"
        assert len(res_png.content) > 0

        res_ico_static = await client.get("/static/favicon.ico")
        assert res_ico_static.status_code == 200
        assert len(res_ico_static.content) > 0

        # Dedicated root favicon.ico route serving
        res_ico_root = await client.get("/favicon.ico")
        assert res_ico_root.status_code == 200
        assert len(res_ico_root.content) > 0


@pytest.mark.anyio
async def test_health_endpoint() -> None:
    res = await health()
    assert res == {"status": "ok"}


@pytest.mark.anyio
async def test_create_app_with_fakes_and_endpoints() -> None:
    fake_queue = FakeTaskQueue()
    fake_reader = FakeStateReader(
        agents_data=[{"agent_id": "worker-1", "state": "idle"}],
        events_data=[{"event_id": "1", "state": "idle"}],
    )
    custom_settings = Settings(redis_url="redis://fake:6379/9", app_env="test")

    app = create_app(
        settings=custom_settings,
        task_queue=fake_queue,
        state_reader=fake_reader,
    )

    assert app.version == "0.3.0"
    assert app.state.settings.app_env == "test"
    assert app.state.gateway is None

    req = Request({"type": "http", "app": app})

    # Test task submission endpoint
    task_res = await create_task(TaskInput(prompt="build agent"), request=req)
    assert task_res["status"] == "accepted"
    assert len(fake_queue.enqueued) == 1
    enqueued_task, capability = fake_queue.enqueued[0]
    assert enqueued_task.prompt == "build agent"
    assert capability == AgentCapability.GENERAL

    # Test agents query endpoint
    agents_res = await agents(request=req)
    assert agents_res == [{"agent_id": "worker-1", "state": "idle"}]

    # Test events query endpoint
    events_res = await events(request=req, limit=10)
    assert events_res == [{"event_id": "1", "state": "idle"}]
    assert fake_reader.last_events_limit == 10


@pytest.mark.anyio
async def test_events_endpoint_preserves_inference_telemetry() -> None:
    telemetry_payload = {
        "provider_name": "cloudflare",
        "attempt_index": 0,
        "latency_ms": 12.5,
        "status": "succeeded",
    }
    fake_reader = FakeStateReader(
        events_data=[
            {
                "event_id": "1",
                "state": "working",
                "task_id": "task-test-1",
                "inference_telemetry": json.dumps(telemetry_payload),
            }
        ]
    )
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        state_reader=fake_reader,
    )
    req = Request({"type": "http", "app": app})
    events_res = await events(request=req, limit=5)
    assert len(events_res) == 1
    assert events_res[0]["task_id"] == "task-test-1"
    assert "inference_telemetry" in events_res[0]
    parsed_telem = json.loads(events_res[0]["inference_telemetry"])
    assert parsed_telem["provider_name"] == "cloudflare"
    assert parsed_telem["status"] == "succeeded"


@pytest.mark.anyio
async def test_create_task_automatic_research_routing() -> None:
    fake_queue = FakeTaskQueue()
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        task_queue=fake_queue,
        state_reader=FakeStateReader(),
    )
    req = Request({"type": "http", "app": app})

    # Phase B requires external retrieval/freshness/source evidence rather than weak verbs alone
    task_res = await create_task(
        TaskInput(prompt="Research the latest consensus algorithms"),
        request=req,
    )
    assert task_res["status"] == "accepted"
    assert len(fake_queue.enqueued) == 1
    enqueued_task, capability = fake_queue.enqueued[0]
    assert enqueued_task.prompt == "Research the latest consensus algorithms"
    assert capability == AgentCapability.RESEARCH


@pytest.mark.anyio
async def test_create_task_explicit_capability_override() -> None:
    fake_queue = FakeTaskQueue()
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        task_queue=fake_queue,
        state_reader=FakeStateReader(),
    )
    req = Request({"type": "http", "app": app})

    task_res = await create_task(
        TaskInput(
            prompt="Research consensus algorithms",
            capability=AgentCapability.GENERAL,
        ),
        request=req,
    )
    assert task_res["status"] == "accepted"
    assert len(fake_queue.enqueued) == 1
    enqueued_task, capability = fake_queue.enqueued[0]
    assert enqueued_task.prompt == "Research consensus algorithms"
    assert capability == AgentCapability.GENERAL


@pytest.mark.anyio
async def test_get_task_result_endpoint_completed() -> None:
    fake_results = FakeResultReader(
        results={
            "task-done-1": {
                "task_id": "task-done-1",
                "status": "completed",
                "output": "Completed research report",
            }
        }
    )
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        result_reader=fake_results,
    )
    req = Request({"type": "http", "app": app})

    res = await task_result("task-done-1", request=req)
    assert res == {
        "task_id": "task-done-1",
        "status": "completed",
        "output": "Completed research report",
    }


@pytest.mark.anyio
async def test_get_task_result_endpoint_pending() -> None:
    fake_results = FakeResultReader(pending={"task-waiting-1"})
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        result_reader=fake_results,
    )
    req = Request({"type": "http", "app": app})

    res = await task_result("task-waiting-1", request=req)
    assert res == {
        "task_id": "task-waiting-1",
        "status": "pending",
        "output": None,
    }


@pytest.mark.anyio
async def test_get_task_result_endpoint_not_found() -> None:
    fake_results = FakeResultReader()
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        result_reader=fake_results,
    )
    req = Request({"type": "http", "app": app})

    with pytest.raises(HTTPException) as exc_info:
        await task_result("unknown-id", request=req)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Task 'unknown-id' not found"


@pytest.mark.anyio
async def test_agents_endpoint_returns_capability_metadata() -> None:
    fake_reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "research-agent-1",
                "agent_type": "research-agent",
                "capabilities": ["research"],
                "providers": {
                    "research_executor": "tavily",
                    "inference_provider": "cloudflare",
                },
                "research_executor": "tavily",
                "inference_provider": "cloudflare",
                "state": "idle",
                "detail": "Waiting for work",
            },
            {
                "agent_id": "worker-1",
                "agent_type": "worker",
                "capabilities": ["general"],
                "providers": {},
                "state": "idle",
                "detail": "Waiting for work",
            },
        ]
    )
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        state_reader=fake_reader,
    )
    req = Request({"type": "http", "app": app})

    res = await agents(request=req)
    assert len(res) == 2
    assert res[0]["agent_id"] == "research-agent-1"
    assert res[0]["capabilities"] == ["research"]
    assert res[0]["providers"]["research_executor"] == "tavily"
    assert res[0]["providers"]["inference_provider"] == "cloudflare"
    assert res[0]["state"] == "idle"

    assert res[1]["agent_id"] == "worker-1"
    assert res[1]["capabilities"] == ["general"]
    assert res[1]["state"] == "idle"


@pytest.mark.anyio
async def test_dashboard_endpoint_serves_console() -> None:
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        task_queue=FakeTaskQueue(),
        state_reader=FakeStateReader(),
    )
    assert app is not None
    html_content = await dashboard()
    assert "<!doctype html>" in html_content
    assert "Autonomous Agent Lab" in html_content
    assert "A generic platform for experimenting with autonomous-agent patterns." in html_content


@pytest.mark.anyio
async def test_dashboard_html_contains_required_console_elements() -> None:
    html = await dashboard()

    # Form and inputs
    assert '<form id="task">' in html
    assert '<textarea' in html
    assert 'id="prompt"' in html
    assert 'id="submit-btn"' in html
    assert 'id="clear-prompt-btn"' in html
    assert 'id="submit-error"' in html

    # Tracking and results
    assert 'id="active-task-id"' in html
    assert 'id="task-routing-status"' in html
    assert 'id="task-progress-bar"' in html
    assert 'id="result-box"' in html
    assert 'id="result-content"' in html
    assert 'id="copy-result-btn"' in html
    assert 'id="clear-task-btn"' not in html

    # Observatory panels
    assert 'id="agents"' in html
    assert 'id="events"' in html
    assert 'Available Agents' in html
    assert 'Activity Timeline' in html


@pytest.mark.anyio
async def test_dashboard_html_accessibility_and_security_attributes() -> None:
    html = await dashboard()

    # Accessibility: labels and aria-live regions
    assert '<label for="prompt">' in html
    assert 'aria-live="polite"' in html
    assert 'role="alert"' in html
    assert 'prefers-reduced-motion' in html

    # Security: rel attributes on external links and protocol validation
    assert 'target="_blank"' in html
    assert 'rel="noopener noreferrer"' in html
    # Protocol restriction strictly to http or https in safe link formatting
    assert 'https?:\\/\\/' in html or 'https?://' in html


@pytest.mark.anyio
async def test_dashboard_html_does_not_contain_credential_field_names() -> None:
    html = await dashboard()
    sensitive_keys = [
        "api_key",
        "tavily_api_key",
        "auth_token",
        "account_id",
        "user_password",
        "cf_account_id",
        "cf_api_token",
    ]
    for key in sensitive_keys:
        assert key not in html, f"Found sensitive key '{key}' in dashboard HTML/JS"


# ==============================================================================
# Issue #8: Dynamic Routing Decision Display in Prototype Console Tests
# ==============================================================================


@pytest.mark.anyio
async def test_dashboard_html_contains_routing_progression_elements() -> None:
    html = await dashboard()

    # Core routing pipeline containers
    assert 'id="task-routing-pipeline"' in html
    assert 'class="routing-pipeline"' in html
    assert 'id="pipeline-stepper"' in html
    assert 'class="pipeline-stepper"' in html
    assert 'id="routing-rationale"' in html
    assert 'class="routing-rationale"' in html
    assert 'id="routing-diagnostics"' in html
    assert 'class="routing-diagnostics"' in html
    assert 'id="routing-diag-summary"' in html
    assert 'id="routing-diag-content"' in html

    # Conceptual progression representation in script
    assert "fetchRoutingDecision" in html
    assert "/api/tasks/${encodeURIComponent(taskId)}/routing" in html or "/api/tasks/" in html
    assert "renderRoutingPipeline" in html
    assert "renderRoutingUnavailable" in html

    # Progression steps
    assert "Submitted" in html
    assert "Capability:" in html
    assert "Candidates:" in html
    assert "Selected:" in html
    assert "Result Synthesized" in html

    # Candidate evaluation table structure
    assert "diag-table" in html
    assert "Candidate Agent" in html or "Agent ID" in html


@pytest.mark.anyio
async def test_unavailable_capability_503_response_includes_task_id() -> None:
    fake_queue = FakeTaskQueue()
    app = create_app(
        task_queue=fake_queue,
        state_reader=FakeStateReader(agents_data=[]),
        enforce_availability=True,
    )
    req = Request({"type": "http", "app": app})

    with pytest.raises(HTTPException) as exc_info:
        await create_task(
            TaskInput(
                prompt="perform specialized research",
                capability=AgentCapability.RESEARCH,
            ),
            request=req,
        )

    assert exc_info.value.status_code == 503
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert detail["error"] == "capability_unavailable"
    assert detail["capability"] == "research"
    assert "No eligible agent" in detail["message"]
    assert "task_id" in detail
    assert detail["task_id"] is not None
    assert len(detail["task_id"]) > 0


@pytest.mark.anyio
async def test_presentation_task_routing_endpoint_serves_routing_decision() -> None:
    from orchestrator.presentation.api import task_routing

    decision_payload = {
        "task_id": "task-route-1",
        "prompt_classification": "research",
        "required_capabilities": ["research"],
        "candidates_evaluated": [
            {
                "agent_id": "research-agent-1",
                "agent_type": "research-agent",
                "capabilities": ["research"],
                "execution_state": "idle",
                "availability_state": "ready",
                "dispatch_target": "agent.tasks.research",
                "is_eligible": True,
            }
        ],
        "selected_agent_id": "research-agent-1",
        "dispatch_target": "agent.tasks.research",
        "rationale": "Selected agent 'research-agent-1' for capability 'research'",
        "decision_timestamp": "2026-11-06T05:30:00Z",
    }
    fake_reader = FakeStateReader(
        routing_decisions={"task-route-1": decision_payload}
    )

    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        state_reader=fake_reader,
    )
    req = Request({"type": "http", "app": app})

    result = await task_routing("task-route-1", request=req)
    assert result["task_id"] == "task-route-1"
    assert result["prompt_classification"] == "research"
    assert result["selected_agent_id"] == "research-agent-1"
    assert result["dispatch_target"] == "agent.tasks.research"
    assert len(result["candidates_evaluated"]) == 1
    assert result["candidates_evaluated"][0]["agent_id"] == "research-agent-1"
    assert result["candidates_evaluated"][0]["is_eligible"] is True
    assert "Selected agent 'research-agent-1'" in result["rationale"]


@pytest.mark.anyio
async def test_presentation_task_routing_endpoint_not_found() -> None:
    from orchestrator.presentation.api import task_routing

    fake_reader = FakeStateReader()
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        state_reader=fake_reader,
    )
    req = Request({"type": "http", "app": app})

    with pytest.raises(HTTPException) as exc_info:
        await task_routing("task-nonexistent", request=req)

    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_dashboard_html_prompt_clear_button_spec() -> None:
    html = await dashboard()

    # Clear button must exist beside submit button with explicit non-submitting type
    assert 'id="clear-prompt-btn"' in html
    assert '<button type="button" id="clear-prompt-btn" class="btn-secondary">Clear</button>' in html

    # Verify script handles clearing prompt input and resetting counter without submitting
    assert "clearPromptBtn = document.getElementById('clear-prompt-btn')" in html
    assert "clearPromptBtn.addEventListener('click'" in html
    assert "promptInput.value = ''" in html
    assert "promptCount.textContent = '0 / 2000'" in html
    assert "promptInput.focus()" in html


@pytest.mark.anyio
async def test_dashboard_html_clear_button_does_not_clear_task_state_or_observability() -> None:
    html = await dashboard()

    # The prompt clear handler must be scoped only to prompt input, counter, and focus.
    # It must NOT reset or clear activeTaskId, result-content, routing pipeline, agents, or events.
    clear_btn_block = html.split("clearPromptBtn.addEventListener('click'")[1].split("});")[0]

    assert "promptInput.value = ''" in clear_btn_block
    assert "promptCount.textContent = '0 / 2000'" in clear_btn_block
    assert "promptInput.focus()" in clear_btn_block

    # Assert it does NOT touch task results or routing or timeline
    assert "activeTaskId" not in clear_btn_block
    assert "resultContentEl" not in clear_btn_block
    assert "resetRoutingPipeline" not in clear_btn_block
    assert "stopResultPolling" not in clear_btn_block
    assert "agentsContainer" not in clear_btn_block
    assert "eventsContainer" not in clear_btn_block


# ==============================================================================
# Issue #18 Increment 1: Task-Result Inference Provider Presentation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_dashboard_html_contains_inference_metadata_structure() -> None:
    html = await dashboard()

    # Core metadata container in Current Task / Results card
    assert 'id="task-inference-meta"' in html
    assert 'class="inference-meta-bar"' in html
    assert 'style="display: none;"' in html
    assert 'aria-live="polite"' in html

    # Styling and subcomponent classes
    assert "inference-meta-header" in html
    assert "inference-meta-group" in html
    assert "inference-meta-item" in html
    assert "meta-label" in html
    assert "meta-val" in html

    # Script DOM binding
    assert "taskInferenceMetaEl = document.getElementById('task-inference-meta')" in html

    # Function declarations
    assert "clearInferenceMeta" in html
    assert "renderInferenceMeta" in html


@pytest.mark.anyio
async def test_inference_metadata_telemetry_endpoint_called_by_result_polling() -> None:
    html = await dashboard()

    poll_fn = html.split("async function pollTaskResult(")[1].split("// --- Task Submission ---")[0]

    # Directly queries task-scoped telemetry endpoint
    assert "fetch(`/api/tasks/${encodeURIComponent(taskId)}/telemetry`)" in poll_fn
    # Guards execution against stale async responses
    assert "if (telemRes.ok && activeTaskId === taskId)" in poll_fn
    assert "renderInferenceMeta(telem)" in poll_fn


@pytest.mark.anyio
async def test_inference_metadata_result_polling_does_not_call_events_stream() -> None:
    html = await dashboard()

    poll_fn = html.split("async function pollTaskResult(")[1].split("// --- Task Submission ---")[0]

    # Task polling must not query /api/events or stream events
    assert "/api/events" not in poll_fn
    assert "syncEventsForTask" not in html
    assert "findSuccessfulTelemetry" not in html


@pytest.mark.anyio
async def test_inference_metadata_activity_timeline_retains_events_stream() -> None:
    html = await dashboard()

    refresh_fn = html.split("async function refreshSystem() {")[1].split("function timeAgo(")[0]

    # Activity Timeline retains polling /api/events?limit=25
    assert "fetch('/api/events?limit=25')" in refresh_fn


@pytest.mark.anyio
async def test_inference_metadata_provider_from_telemetry_not_configuration() -> None:
    html = await dashboard()

    render_fn = html.split("function renderInferenceMeta(")[1].split("function updateRoutingObservation(")[0]

    # Provider comes directly from telem.provider_name
    assert "telem.provider_name" in render_fn
    assert '<span class="meta-label">Provider:</span>' in render_fn
    assert '<strong class="meta-val">${esc(providerName)}</strong>' in render_fn

    # Must not reference configured preferences or agent card providers
    assert "a.providers" not in render_fn
    assert "preferences" not in render_fn


@pytest.mark.anyio
async def test_inference_metadata_model_displayed_when_available() -> None:
    html = await dashboard()

    render_fn = html.split("function renderInferenceMeta(")[1].split("function updateRoutingObservation(")[0]

    # Displays model when available
    assert "typeof telem.model_name === 'string' && telem.model_name.trim()" in render_fn
    assert '<span class="meta-label">Model:</span>' in render_fn
    assert '<span class="meta-val">${esc(telem.model_name.trim())}</span>' in render_fn


@pytest.mark.anyio
async def test_inference_metadata_missing_model_omitted_not_default() -> None:
    html = await dashboard()

    render_fn = html.split("function renderInferenceMeta(")[1].split("function updateRoutingObservation(")[0]

    # When model is missing, it must not fabricate "default"
    assert '"default"' not in render_fn
    assert "'default'" not in render_fn
    assert "|| 'default'" not in render_fn
    assert '|| "default"' not in render_fn


@pytest.mark.anyio
async def test_inference_metadata_latency_displayed_when_available() -> None:
    html = await dashboard()

    render_fn = html.split("function renderInferenceMeta(")[1].split("function updateRoutingObservation(")[0]

    # Validates finite non-negative latency
    assert "typeof telem.latency_ms === 'number' && Number.isFinite(telem.latency_ms) && telem.latency_ms >= 0" in render_fn
    assert '<span class="meta-label">Latency:</span>' in render_fn
    assert '<span class="meta-val">${esc(formattedLatency)} ms</span>' in render_fn


@pytest.mark.anyio
async def test_inference_metadata_attempt_index_displays_one_based() -> None:
    html = await dashboard()

    render_fn = html.split("function renderInferenceMeta(")[1].split("function updateRoutingObservation(")[0]

    # Validates integer attempt index and maps zero-based attempt to human-friendly 1-based attempt
    assert "typeof telem.attempt_index === 'number' && Number.isInteger(telem.attempt_index) && telem.attempt_index >= 0" in render_fn
    assert '<span class="meta-label">Attempt:</span>' in render_fn
    assert "${telem.attempt_index + 1}" in render_fn


@pytest.mark.anyio
async def test_inference_metadata_telemetry_from_another_task_isolated() -> None:
    html = await dashboard()

    poll_fn = html.split("async function pollTaskResult(")[1].split("// --- Task Submission ---")[0]

    # Result polling ensures activeTaskId matches before applying telemetry
    assert "if (telemRes.ok && activeTaskId === taskId)" in poll_fn
    assert "renderInferenceMeta(telem)" in poll_fn


@pytest.mark.anyio
async def test_inference_metadata_client_handles_malformed_telemetry_payload_safely() -> None:
    html = await dashboard()

    render_fn = html.split("function renderInferenceMeta(")[1].split("function updateRoutingObservation(")[0]
    poll_fn = html.split("async function pollTaskResult(")[1].split("// --- Task Submission ---")[0]

    # Validates structure before rendering
    assert "if (!telem || typeof telem !== 'object' || Array.isArray(telem)) {" in render_fn
    assert "clearInferenceMeta();" in render_fn

    # Polling validates envelope object shape
    assert "envelope && typeof envelope === 'object' && !Array.isArray(envelope)" in poll_fn


@pytest.mark.anyio
async def test_inference_metadata_missing_telemetry_preserves_result_rendering() -> None:
    html = await dashboard()

    poll_fn = html.split("async function pollTaskResult(")[1].split("// --- Task Submission ---")[0]
    render_fn = html.split("function renderInferenceMeta(")[1].split("function updateRoutingObservation(")[0]

    # Output formatting occurs before and independently of telemetry rendering
    output_idx = poll_fn.index("safeFormatResult(data.output)")
    fetch_idx = poll_fn.index("fetch(`/api/tasks/${encodeURIComponent(taskId)}/telemetry`)")
    assert output_idx < fetch_idx

    # If telemetry is null/missing, renderInferenceMeta safely clears/hides the container
    assert "if (!telem || typeof telem !== 'object' || Array.isArray(telem)) {" in render_fn
    assert "clearInferenceMeta();" in render_fn


@pytest.mark.anyio
async def test_inference_metadata_reset_behavior_clears_stale_metadata() -> None:
    html = await dashboard()

    # Verify clearInferenceMeta() resets display and content
    clear_fn = html.split("function clearInferenceMeta(")[1].split("function renderInferenceMeta(")[0]
    assert "taskInferenceMetaEl.style.display = 'none'" in clear_fn
    assert "taskInferenceMetaEl.innerHTML = ''" in clear_fn

    # Form submission clears metadata
    submit_block = html.split("taskForm.onsubmit = async (ev) => {")[1].split("const response = await fetch('/api/tasks'")[0]
    assert "clearInferenceMeta();" in submit_block

    # Routing unavailable clears metadata
    unavail_block = html.split("function renderRoutingUnavailable(")[1].split("pipelineStepperEl.innerHTML")[0]
    assert "clearInferenceMeta();" in unavail_block

    # Failed task observation clears metadata
    failed_block = html.split("if (status === 'failed') {")[1].split("routingStatusEl.innerHTML")[0]
    assert "clearInferenceMeta();" in failed_block


@pytest.mark.anyio
async def test_inference_metadata_dynamic_content_safely_escaped() -> None:
    html = await dashboard()

    render_fn = html.split("function renderInferenceMeta(")[1].split("function updateRoutingObservation(")[0]
    pipeline_fn = html.split("function renderRoutingPipeline(")[1].split("routingRationaleEl.className")[0]

    # Dynamic fields are passed through esc()
    assert "${esc(providerName)}" in render_fn
    assert "${esc(telem.model_name.trim())}" in render_fn
    assert "${esc(formattedLatency)}" in render_fn
    assert "${esc(providerName)}" in pipeline_fn


@pytest.mark.anyio
async def test_inference_metadata_narrow_screen_safeguards() -> None:
    html = await dashboard()

    meta_bar_css = html.split(".inference-meta-bar {")[1].split("}")[0]
    meta_group_css = html.split(".inference-meta-group {")[1].split("}")[0]
    meta_val_css = html.split(".inference-meta-item .meta-val {")[1].split("}")[0]

    # Flex wrapping to accommodate narrow screens
    assert "flex-wrap: wrap;" in meta_bar_css
    assert "flex-wrap: wrap;" in meta_group_css

    # Word break to prevent long model names from overflowing
    assert "word-break: break-all;" in meta_val_css
    assert "overflow-wrap: anywhere;" in meta_val_css


@pytest.mark.anyio
async def test_api_tasks_telemetry_success() -> None:
    telem = {
        "provider_name": "cloudflare",
        "attempt_index": 0,
        "latency_ms": 115.2,
        "status": "succeeded",
        "model_name": "@cf/meta/llama-3.1-8b-instruct",
        "finish_reason": "stop",
    }
    fake_reader = FakeStateReader(
        events_data=[
            {
                "task_id": "task-telemetry-endpoint-1",
                "inference_telemetry": telem,
            }
        ]
    )
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        state_reader=fake_reader,
    )
    req = Request({"type": "http", "app": app})
    res = await task_telemetry("task-telemetry-endpoint-1", request=req)

    assert res["task_id"] == "task-telemetry-endpoint-1"
    assert res["inference_telemetry"]["provider_name"] == "cloudflare"
    assert res["inference_telemetry"]["model_name"] == "@cf/meta/llama-3.1-8b-instruct"
    assert res["inference_telemetry"]["latency_ms"] == 115.2
    assert res["inference_telemetry"]["attempt_index"] == 0
    assert res["inference_telemetry"]["status"] == "succeeded"


@pytest.mark.anyio
async def test_api_tasks_telemetry_not_found() -> None:
    fake_reader = FakeStateReader(events_data=[])
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        state_reader=fake_reader,
    )
    req = Request({"type": "http", "app": app})
    with pytest.raises(HTTPException) as exc_info:
        await task_telemetry("task-not-existing", request=req)
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail


@pytest.mark.anyio
async def test_api_tasks_telemetry_empty_task_id_rejected() -> None:
    fake_reader = FakeStateReader(events_data=[])
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        state_reader=fake_reader,
    )
    req = Request({"type": "http", "app": app})
    with pytest.raises(HTTPException) as exc_info:
        await task_telemetry("", request=req)
    assert exc_info.value.status_code == 400
    assert "task_id must not be empty" in exc_info.value.detail

    with pytest.raises(HTTPException) as exc_info2:
        await task_telemetry("   ", request=req)
    assert exc_info2.value.status_code == 400


@pytest.mark.anyio
async def test_api_tasks_telemetry_unconfigured_reader_returns_503() -> None:
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        state_reader=None,
        task_queue=FakeTaskQueue(),
        result_reader=FakeResultReader(),
    )
    # Ensure get_task_inference_telemetry is None
    app.state.get_task_inference_telemetry = None
    req = Request({"type": "http", "app": app})
    with pytest.raises(HTTPException) as exc_info:
        await task_telemetry("task-any", request=req)
    assert exc_info.value.status_code == 503
    assert "not configured" in exc_info.value.detail


@pytest.mark.anyio
async def test_api_events_serves_task_correlated_inference_telemetry() -> None:
    from agent_contracts import AgentEvent, InferenceExecutionTelemetry

    telem = InferenceExecutionTelemetry(
        provider_name="cloudflare",
        attempt_index=0,
        latency_ms=115.2,
        status="succeeded",
        model_name="@cf/meta/llama-3.1-8b-instruct",
        finish_reason="stop",
    )
    event = AgentEvent.create(
        agent_id="worker-1",
        state="working",
        detail="Inference attempt 0 succeeded (cloudflare)",
        task_id="task-increment-1-test",
        inference_telemetry=telem,
    )
    fake_reader = FakeStateReader(events_data=[event.to_dict()])
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        state_reader=fake_reader,
    )
    req = Request({"type": "http", "app": app})
    res = await events(request=req, limit=5)

    assert len(res) == 1
    ev = res[0]
    assert ev["task_id"] == "task-increment-1-test"
    telem_dict = (
        json.loads(ev["inference_telemetry"])
        if isinstance(ev["inference_telemetry"], str)
        else ev["inference_telemetry"]
    )
    assert telem_dict["provider_name"] == "cloudflare"
    assert telem_dict["model_name"] == "@cf/meta/llama-3.1-8b-instruct"
    assert telem_dict["latency_ms"] == 115.2
    assert telem_dict["attempt_index"] == 0
    assert telem_dict["status"] == "succeeded"

