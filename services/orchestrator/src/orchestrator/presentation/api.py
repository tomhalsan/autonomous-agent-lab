from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from agent_contracts import __version__
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from orchestrator.application.ports import (
    CandidateFinder,
    CapabilityRouter,
    EventPublisher,
    ResultReader,
    StateReader,
    TaskDispatcher,
    TaskQueue,
)
from orchestrator.application.queries import (
    FindCandidatesByCapability,
    GetAgentStates,
    GetRecentEvents,
    GetTaskResult,
)
from orchestrator.application.use_cases import SubmitTask
from orchestrator.config import Settings
from orchestrator.domain.capability import (
    AgentCapability,
    CapabilityUnavailableError,
)
from orchestrator.domain.selection import CandidateSelectionPolicy
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
    except CapabilityUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "capability_unavailable",
                "capability": str(exc.capability),
                "message": str(exc.message),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def agents(request: Request) -> list[dict[str, Any]]:
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


async def candidates(capability: str, request: Request) -> list[dict[str, Any]]:
    find_candidates: FindCandidatesByCapability = request.app.state.find_candidates
    try:
        candidate_list = await find_candidates.execute(capability)
        return [c.to_dict() for c in candidate_list]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class NoOpResultReader:
    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        return None


def create_app(
    settings: Settings | None = None,
    gateway: RedisGateway | None = None,
    task_queue: TaskQueue | None = None,
    dispatcher: TaskDispatcher | None = None,
    state_reader: StateReader | None = None,
    result_reader: ResultReader | None = None,
    router: CapabilityRouter | None = None,
    candidate_finder: CandidateFinder | None = None,
    event_publisher: EventPublisher | None = None,
    enforce_availability: bool | None = None,
    selection_policy: CandidateSelectionPolicy | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()

    if (
        gateway is None
        and task_queue is None
        and dispatcher is None
        and state_reader is None
        and result_reader is None
    ):
        gateway = RedisGateway(resolved_settings.redis_url)

    queue = task_queue or gateway
    dispatch_client = dispatcher or (gateway if isinstance(gateway, TaskDispatcher) else None)
    reader = state_reader or gateway
    results = result_reader or gateway or NoOpResultReader()
    events_out = event_publisher or gateway

    should_enforce = enforce_availability if enforce_availability is not None else (gateway is not None)
    resolved_finder = candidate_finder
    if resolved_finder is None and should_enforce and reader is not None:
        resolved_finder = FindCandidatesByCapability(reader)

    find_candidates_query = resolved_finder or (FindCandidatesByCapability(reader) if reader else None)

    submit_task = SubmitTask(
        queue=queue,
        dispatcher=dispatch_client,
        router=router,
        candidate_finder=resolved_finder,
        events=events_out,
        selection_policy=selection_policy,
    )
    get_agent_states = GetAgentStates(reader) if reader else None
    get_recent_events = GetRecentEvents(reader) if reader else None
    get_task_result = GetTaskResult(results)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if gateway is not None and hasattr(gateway, "aclose"):
            await gateway.aclose()

    app = FastAPI(title="Autonomous Agent Lab", version=__version__, lifespan=lifespan)

    app.state.settings = resolved_settings
    app.state.gateway = gateway
    app.state.submit_task = submit_task
    app.state.get_agent_states = get_agent_states
    app.state.get_recent_events = get_recent_events
    app.state.get_task_result = get_task_result
    app.state.find_candidates = find_candidates_query

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        favicon_file = static_dir / "favicon.ico"
        if favicon_file.exists():

            async def favicon() -> FileResponse:
                return FileResponse(favicon_file)

            app.add_api_route(
                "/favicon.ico", favicon, methods=["GET"], include_in_schema=False
            )

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/", dashboard, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/api/tasks", create_task, methods=["POST"], status_code=202)
    app.add_api_route("/api/agents", agents, methods=["GET"])
    app.add_api_route("/api/events", events, methods=["GET"])
    app.add_api_route("/api/tasks/{task_id}/result", task_result, methods=["GET"])
    app.add_api_route("/api/candidates", candidates, methods=["GET"])

    return app


app = create_app()
