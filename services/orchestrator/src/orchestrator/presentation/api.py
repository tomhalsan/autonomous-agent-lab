import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from orchestrator.application.use_cases import SubmitTask
from orchestrator.infrastructure.redis_adapter import RedisGateway

app = FastAPI(title="Autonomous Agent Lab", version="0.1.0")
gateway = RedisGateway(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
submit_task = SubmitTask(gateway)

class TaskInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return Path(__file__).with_name("index.html").read_text()

@app.post("/api/tasks", status_code=202)
async def create_task(value: TaskInput) -> dict[str, str]:
    try:
        task = await submit_task.execute(value.prompt)
        return {"task_id": task.task_id, "status": "accepted"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/api/agents")
async def agents() -> list[dict[str, str]]:
    return await gateway.agents()

@app.get("/api/events")
async def events(limit: int = 25) -> list[dict[str, str]]:
    return await gateway.events(min(max(limit, 1), 100))
