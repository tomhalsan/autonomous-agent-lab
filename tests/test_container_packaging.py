"""Tests for container packaging and dependency boundaries."""

from pathlib import Path

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
