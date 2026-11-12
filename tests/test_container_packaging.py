"""Tests for container packaging and dependency boundaries."""

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_worker_dockerfile_packaging() -> None:
    dockerfile = (REPO_ROOT / "services" / "worker" / "Dockerfile").read_text()
    lines = dockerfile.splitlines()

    copy_lines = [line.strip() for line in lines if line.strip().startswith("COPY packages/")]
    expected_copies = [
        "COPY packages/contracts/src /app/contracts",
        "COPY packages/policy/src /app/policy",
        "COPY packages/infrastructure/src /app/infrastructure",
    ]
    assert copy_lines == expected_copies

    env_line = next(line for line in lines if "PYTHONPATH=" in line)
    assert "PYTHONPATH=/app/src:/app/contracts:/app/policy:/app/infrastructure" in env_line


def test_research_agent_dockerfile_packaging() -> None:
    dockerfile = (REPO_ROOT / "services" / "research_agent" / "Dockerfile").read_text()
    lines = dockerfile.splitlines()

    copy_lines = [line.strip() for line in lines if line.strip().startswith("COPY packages/")]
    expected_copies = [
        "COPY packages/contracts/src /app/contracts",
        "COPY packages/policy/src /app/policy",
        "COPY packages/infrastructure/src /app/infrastructure",
    ]
    assert copy_lines == expected_copies

    env_line = next(line for line in lines if "PYTHONPATH=" in line)
    assert "PYTHONPATH=/app/src:/app/contracts:/app/policy:/app/infrastructure" in env_line


def test_orchestrator_dockerfile_preserves_boundaries() -> None:
    dockerfile = (REPO_ROOT / "services" / "orchestrator" / "Dockerfile").read_text()
    lines = dockerfile.splitlines()

    copy_lines = [line.strip() for line in lines if line.strip().startswith("COPY packages/")]
    assert copy_lines == ["COPY packages/contracts/src /app/contracts"]

    assert not any("packages/policy" in line for line in lines)
    assert not any("packages/infrastructure" in line for line in lines)
    env_line = next(line for line in lines if "PYTHONPATH=" in line)
    assert "PYTHONPATH=/app/src:/app/contracts" in env_line
    assert "/app/policy" not in env_line
    assert "/app/infrastructure" not in env_line


def test_agent_infrastructure_does_not_require_mcp_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing agent_infrastructure must not require the optional MCP SDK."""
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.types", None)
    for mod in list(sys.modules):
        if mod.startswith("agent_infrastructure.mcp"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    import agent_infrastructure

    reloaded = importlib.reload(agent_infrastructure)
    assert hasattr(reloaded, "InMemoryToolRegistry")
    assert not hasattr(reloaded, "McpClient")
    assert not hasattr(reloaded, "McpTool")
    assert not hasattr(reloaded, "McpServerConfig")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_infrastructure.mcp")


def test_mcp_dependency_isolation_in_service_requirements() -> None:
    """MCP SDK is only packaged in services that actively run MCP adapters."""
    research_reqs = (REPO_ROOT / "services" / "research_agent" / "requirements.txt").read_text()
    orchestrator_reqs = (REPO_ROOT / "services" / "orchestrator" / "requirements.txt").read_text()
    worker_reqs = (REPO_ROOT / "services" / "worker" / "requirements.txt").read_text()

    assert "mcp" not in research_reqs
    assert "mcp" not in orchestrator_reqs
    assert "mcp>=2.3.0,<3.0.0" in worker_reqs
