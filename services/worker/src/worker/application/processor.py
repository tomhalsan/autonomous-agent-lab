import asyncio
from agent_contracts import AgentEvent, TaskRequested
from .ports import MessageBus

class TaskProcessor:
    def __init__(self, agent_id: str, bus: MessageBus) -> None:
        self.agent_id = agent_id
        self.bus = bus

    async def run_once(self, task: TaskRequested) -> None:
        await self.bus.publish(AgentEvent.create(self.agent_id, "working", f"Processing: {task.prompt}", task.task_id))
        await asyncio.sleep(1)
        await self.bus.publish(AgentEvent.create(self.agent_id, "idle", f"Completed: {task.prompt}", task.task_id))
