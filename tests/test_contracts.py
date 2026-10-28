from agent_contracts import AgentEvent, TaskRequested


def test_task_message_round_trip() -> None:
    original = TaskRequested.create("hello")
    restored = TaskRequested.from_json(original.to_json())
    assert restored == original


def test_agent_event_round_trip() -> None:
    original = AgentEvent.create("worker-1", "working", "Processing prompt", task_id="task-123")
    restored = AgentEvent.from_json(original.to_json())
    assert restored == original
    assert restored.task_id == "task-123"


def test_agent_event_to_dict() -> None:
    event = AgentEvent.create("worker-1", "idle", "Waiting")
    data = event.to_dict()
    assert data["agent_id"] == "worker-1"
    assert data["state"] == "idle"
    assert data["task_id"] is None
