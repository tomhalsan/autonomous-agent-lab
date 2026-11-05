import pytest
from agent_contracts import AgentCapabilityMetadata, AgentEvent, TaskRequested


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


def test_agent_capability_metadata_round_trip() -> None:
    original = AgentCapabilityMetadata(
        agent_id="research-agent-1",
        capabilities=["research"],
        agent_type="research-agent",
        providers={"research_executor": "tavily", "inference_provider": "cloudflare"},
    )
    restored = AgentCapabilityMetadata.from_json(original.to_json())
    assert restored == original
    assert restored.agent_id == "research-agent-1"
    assert restored.capabilities == ["research"]
    assert restored.agent_type == "research-agent"
    assert restored.providers["research_executor"] == "tavily"
    assert restored.providers["inference_provider"] == "cloudflare"


def test_agent_capability_metadata_rejects_empty_agent_id() -> None:
    with pytest.raises(ValueError, match="agent_id must not be empty"):
        AgentCapabilityMetadata(agent_id="   ", capabilities=["general"])


@pytest.mark.parametrize(
    "bad_key",
    ["api_key", "tavily_api_key", "token", "auth_token", "account_id", "secret", "user_password", "credential"],
)
def test_agent_capability_metadata_rejects_sensitive_keys(bad_key: str) -> None:
    with pytest.raises(ValueError, match="must not be exposed in provider metadata"):
        AgentCapabilityMetadata(
            agent_id="agent-1",
            capabilities=["research"],
            providers={bad_key: "confidential_value"},
        )
