from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from orchestrator.application.ports import StateReader, TaskQueue
from orchestrator.application.queries import GetAgentStates, GetRecentEvents
from orchestrator.application.use_cases import SubmitTask
from orchestrator.config import Settings
from orchestrator.infrastructure.redis_adapter import RedisGateway


class TaskInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)


async def health() -> dict[str, str]:
    return {"status": "ok"}


async def dashboard() -> str:
    return Path(__file__).with_name("index.html").read_text()


async def create_task(value: TaskInput, request: Request) -> dict[str, str]:
    submit_task: SubmitTask = request.app.state.submit_task
    try:
        task = await submit_task.execute(value.prompt)
        return {"task_id": task.task_id, "status": "accepted"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def agents(request: Request) -> list[dict[str, str]]:
    get_agent_states: GetAgentStates = request.app.state.get_agent_states
    return await get_agent_states.execute()


async def events(request: Request, limit: int = 25) -> list[dict[str, str]]:
    get_recent_events: GetRecentEvents = request.app.state.get_recent_events
    return await get_recent_events.execute(limit=limit)


def create_app(
    settings: Settings | None = None,
    gateway: RedisGateway | None = None,
    task_queue: TaskQueue | None = None,
    state_reader: StateReader | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()

    if gateway is None and task_queue is None and state_reader is None:
        gateway = RedisGateway(resolved_settings.redis_url)

    queue = task_queue or gateway
    reader = state_reader or gateway

    submit_task = SubmitTask(queue)
    get_agent_states = GetAgentStates(reader)
    get_recent_events = GetRecentEvents(reader)

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

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/", dashboard, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/api/tasks", create_task, methods=["POST"], status_code=202)
    app.add_api_route("/api/agents", agents, methods=["GET"])
    app.add_api_route("/api/events", events, methods=["GET"])

    return app


app = create_app()
