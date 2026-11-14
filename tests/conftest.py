from __future__ import annotations

import pytest

_PROVIDER_ENVIRONMENT_VARIABLES = (
    "RESEARCH_EXECUTOR",
    "TAVILY_API_KEY",
    "TAVILY_MCP_URL",
    "INFERENCE_PROVIDER",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_AI_API_TOKEN",
    "CLOUDFLARE_AI_MODEL",
    "CLOUDFLARE_AI_MAX_TOKENS",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_BASE_URL",
    "GEMINI_MAX_TOKENS",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("external providers")
    group.addoption(
        "--run-external-providers",
        action="store_true",
        default=False,
        help="allow tests marked external_provider to execute",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("run_external_providers"):
        return

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        if item.get_closest_marker("external_provider") is None:
            selected.append(item)
        else:
            deselected.append(item)

    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


@pytest.fixture(autouse=True)
def isolate_provider_environment(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if request.node.get_closest_marker("external_provider") is not None:
        return

    for variable in _PROVIDER_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def provider_environment_variables() -> tuple[str, ...]:
    return _PROVIDER_ENVIRONMENT_VARIABLES
