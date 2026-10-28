import pytest
from orchestrator.presentation.api import TaskInput, dashboard
from pydantic import ValidationError


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
