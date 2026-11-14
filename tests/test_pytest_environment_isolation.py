from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_LIVE_TEST_NAME = "test_live_tavily_mcp_server_contract"


def test_provider_environment_is_cleared_and_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
    provider_environment_variables: tuple[str, ...],
) -> None:
    assert all(variable not in os.environ for variable in provider_environment_variables)

    monkeypatch.setenv("INFERENCE_PROVIDER", "cloudflare")
    assert os.environ["INFERENCE_PROVIDER"] == "cloudflare"


@pytest.mark.parametrize(
    ("pytest_args", "expected_marker_output"),
    [
        ([], None),
        (["-m", "not redis"], None),
        (["-m", "redis"], "test_redis_integration.py"),
        (["-m", "external_provider"], None),
        (["--run-external-providers", "-m", "external_provider"], _LIVE_TEST_NAME),
    ],
    ids=[
        "normal",
        "not-redis",
        "redis",
        "external-marker-without-opt-in",
        "explicit-external-opt-in",
    ],
)
def test_external_provider_collection_requires_explicit_opt_in(
    pytest_args: list[str],
    expected_marker_output: str | None,
) -> None:
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "tests/test_pytest_environment_isolation.py::test_provider_environment_is_cleared_and_can_be_overridden",
        "tests/test_redis_integration.py",
        "tests/test_tavily_mcp_live.py",
        *pytest_args,
    ]

    result = subprocess.run(
        command,
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode in (0, 5), output
    assert (_LIVE_TEST_NAME in output) is (
        expected_marker_output == _LIVE_TEST_NAME
    )
    if expected_marker_output is not None:
        assert expected_marker_output in output
