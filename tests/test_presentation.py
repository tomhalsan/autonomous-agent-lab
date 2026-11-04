import pytest
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
)
from pydantic import ValidationError
from starlette.requests import Request

from tests.test_application import FakeStateReader, FakeTaskQueue


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
