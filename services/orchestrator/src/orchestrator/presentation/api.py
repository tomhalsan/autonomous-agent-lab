from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from orchestrator.application.ports import (
    CapabilityRouter,
    ResultReader,
    StateReader,
    TaskQueue,
)
from orchestrator.application.queries import (
    GetAgentStates,
    GetRecentEvents,
    GetTaskResult,
)
from orchestrator.application.use_cases import SubmitTask
from orchestrator.config import Settings
from orchestrator.domain.capability import AgentCapability
from orchestrator.infrastructure.redis_adapter import RedisGateway


class TaskInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    capability: AgentCapability | None = None


async def health() -> dict[str, str]:
    return {"status": "ok"}


async def dashboard() -> str:
    return Path(__file__).with_name("index.html").read_text()


async def create_task(value: TaskInput, request: Request) -> dict[str, str]:
    submit_task: SubmitTask = request.app.state.submit_task
    try:
        task = await submit_task.execute(value.prompt, capability=value.capability)
        return {"task_id": task.task_id, "status": "accepted"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def agents(request: Request) -> list[dict[str, str]]:
    get_agent_states: GetAgentStates = request.app.state.get_agent_states
    return await get_agent_states.execute()


async def events(request: Request, limit: int = 25) -> list[dict[str, str]]:
    get_recent_events: GetRecentEvents = request.app.state.get_recent_events
    return await get_recent_events.execute(limit=limit)


async def task_result(task_id: str, request: Request) -> dict[str, Any]:
    get_task_result: GetTaskResult = request.app.state.get_task_result
    result = await get_task_result.execute(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return result


class NoOpResultReader:
    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        return None


def create_app(
    settings: Settings | None = None,
    gateway: RedisGateway | None = None,
    task_queue: TaskQueue | None = None,
    state_reader: StateReader | None = None,
    result_reader: ResultReader | None = None,
    router: CapabilityRouter | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()

    if (
        gateway is None
        and task_queue is None
        and state_reader is None
        and result_reader is None
    ):
        gateway = RedisGateway(resolved_settings.redis_url)

    queue = task_queue or gateway
    reader = state_reader or gateway
    results = result_reader or gateway or NoOpResultReader()

    submit_task = SubmitTask(queue, router=router)
    get_agent_states = GetAgentStates(reader)
    get_recent_events = GetRecentEvents(reader)
    get_task_result = GetTaskResult(results)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if gateway is not None and hasattr(gateway, "aclose"):
            await gateway.aclose()

    app = FastAPI(title="Autonomous Agent Lab", version="0.1.0", lifespan=lifespan)

    app.state.settings = resolved_settings
    app.state.gateway = gateway
    app.state.submit_task = submit_task
    app.state.get_agent_states = get_agent_states
    app.state.get_recent_events = get_recent_events
    app.state.get_task_result = get_task_result

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/", dashboard, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/api/tasks", create_task, methods=["POST"], status_code=202)
    app.add_api_route("/api/agents", agents, methods=["GET"])
    app.add_api_route("/api/events", events, methods=["GET"])
    app.add_api_route("/api/tasks/{task_id}/result", task_result, methods=["GET"])

    return app


app = create_app()
