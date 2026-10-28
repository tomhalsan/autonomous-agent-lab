import pytest
from orchestrator.domain.task import Task


def test_task_rejects_blank_prompt() -> None:
    with pytest.raises(ValueError):
        Task("   ")

def test_task_accepts_prompt() -> None:
    assert Task("inspect system").prompt == "inspect system"
