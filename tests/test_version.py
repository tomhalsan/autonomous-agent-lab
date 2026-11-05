import tomllib
from pathlib import Path

import orchestrator
import pytest
from agent_contracts import __version__ as contracts_version
from agent_contracts.version import __version__ as root_version
from orchestrator.config import Settings
from orchestrator.presentation.api import create_app, dashboard

from tests.test_application import FakeStateReader, FakeTaskQueue


def test_authoritative_version_is_0_2_0() -> None:
    """Verify that the authoritative version in code is 0.2.0."""
    assert root_version == "0.2.0"
    assert contracts_version == "0.2.0"
    assert orchestrator.__version__ == "0.2.0"


def test_pyproject_matches_authoritative_version() -> None:
    """Ensure pyproject.toml package version remains strictly synchronized."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    assert data["project"]["version"] == "0.2.0"
    assert data["project"]["version"] == root_version


def test_fastapi_reports_version() -> None:
    """Verify FastAPI instance and OpenAPI document report 0.2.0."""
    fake_queue = FakeTaskQueue()
    fake_reader = FakeStateReader()
    app = create_app(
        settings=Settings(redis_url="redis://localhost:6379/0"),
        gateway=None,
        task_queue=fake_queue,
        state_reader=fake_reader,
    )

    assert app.version == "0.2.0"
    assert app.version == root_version

    openapi_doc = app.openapi()
    assert openapi_doc["info"]["version"] == "0.2.0"
    assert openapi_doc["info"]["version"] == root_version


@pytest.mark.anyio
async def test_prototype_console_displays_version() -> None:
    """Verify Prototype Console header and footer display v0.2.0."""
    html = await dashboard()
    assert "v0.2.0" in html
    assert f"v{root_version}" in html
    assert '<span class="version-badge">v0.2.0</span>' in html
    assert '<footer class="app-footer">' in html
