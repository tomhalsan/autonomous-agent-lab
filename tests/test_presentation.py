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
    assert "v0.2.0" in html
    assert '<span class="version-badge">v0.2.0</span>' in html


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

    assert app.version == "0.2.0"
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
async def test_create_task_automatic_research_routing() -> None:
    fake_queue = FakeTaskQueue()
    app = create_app(
        settings=Settings(redis_url="redis://fake:6379/9", app_env="test"),
        task_queue=fake_queue,
        state_reader=FakeStateReader(),
    )
    req = Request({"type": "http", "app": app})

    task_res = await create_task(
        TaskInput(prompt="Research consensus algorithms"),
        request=req,
    )
    assert task_res["status"] == "accepted"
    assert len(fake_queue.enqueued) == 1
    enqueued_task, capability = fake_queue.enqueued[0]
    assert enqueued_task.prompt == "Research consensus algorithms"
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
    assert 'id="clear-task-btn"' in html

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
    assert "resetRoutingPipeline" in html

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
        await create_task(TaskInput(prompt="perform specialized research"), request=req)

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

