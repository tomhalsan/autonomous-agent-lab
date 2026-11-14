"""Unit tests for McpTavilyResearchExecutor and executor factory integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_contracts import (
    TaskRequested,
    ToolDefinition,
    ToolInvocationRequest,
    ToolInvocationResult,
)
from agent_infrastructure.mcp import McpClient, McpTool
from research_agent.application.processor import ResearchProcessor
from research_agent.domain.evaluator import EvidenceEvaluator
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.executor_factory import create_research_executor
from research_agent.infrastructure.mcp_tavily_executor import McpTavilyResearchExecutor
from research_agent.infrastructure.tavily_executor import TavilyResearchExecutor


def _assert_secrets_absent(text: str, secrets: list[str]) -> None:
    if any(secret and secret in text for secret in secrets):
        pytest.fail("Provider output unexpectedly contained a credential")


def _mock_tavily_tool() -> AsyncMock:
    tool = AsyncMock(spec=McpTool)
    tool.definition = ToolDefinition(
        name="tavily_search",
        description="Search tool",
        parameters={"type": "object"},
    )
    return tool


def _mock_mcp_client(tool: AsyncMock) -> AsyncMock:
    client = AsyncMock(spec=McpClient)
    client.__aenter__.return_value = client
    client.list_tools.return_value = [tool]
    return client


def test_executor_factory_creates_tavily_mcp_canonical_only() -> None:
    # 1. Canonical 'tavily_mcp' works
    executor = create_research_executor("tavily_mcp", api_key="tvly-mock-key")
    assert isinstance(executor, McpTavilyResearchExecutor)
    assert executor.url == "https://mcp.tavily.com/mcp/"
    assert executor._headers == {"Authorization": "Bearer tvly-mock-key"}

    # 2. Keyless / without key preserves None headers
    executor_keyless = create_research_executor("tavily_mcp", api_key=None)
    assert isinstance(executor_keyless, McpTavilyResearchExecutor)
    assert executor_keyless._headers is None

    # 3. Direct 'tavily' remains supported
    direct_executor = create_research_executor("tavily", api_key="tvly-direct-key")
    assert isinstance(direct_executor, TavilyResearchExecutor)

    # 4. Aliases are not supported
    with pytest.raises(ValueError, match="Unsupported research executor: 'mcp_tavily'"):
        create_research_executor("mcp_tavily")
    with pytest.raises(ValueError, match="Unsupported research executor: 'tavily-mcp'"):
        create_research_executor("tavily-mcp")


def test_mcp_tavily_executor_repr_redacts_url_and_authorization() -> None:
    secret = "mcp-repr-canary"
    executor = McpTavilyResearchExecutor(
        url=f"https://user:{secret}@mock.mcp/mcp/?api_key={secret}",
        headers={"Authorization": f"Custom {secret}"},
    )

    representation = repr(executor)

    _assert_secrets_absent(representation, [secret])
    assert "[REDACTED]" in representation


def test_executor_factory_does_not_open_network_connections() -> None:
    # Factory constructs object synchronously without opening sockets or connections
    executor = create_research_executor("tavily_mcp")
    assert isinstance(executor, McpTavilyResearchExecutor)


@pytest.mark.anyio
async def test_mcp_tavily_executor_tool_discovery_zero_match_fails() -> None:
    executor = McpTavilyResearchExecutor(url="https://mock.mcp/mcp/")

    mock_client = AsyncMock(spec=McpClient)
    mock_client.list_tools.return_value = []  # No tools found

    with patch.object(McpClient, "from_streamable_http", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        task = TaskRequested.create("test search")
        with pytest.raises(RuntimeError, match="Expected Tavily search tool 'tavily_search' not found"):
            await executor.execute(task)


@pytest.mark.anyio
async def test_mcp_tavily_executor_tool_discovery_multiple_matches_fails() -> None:
    executor = McpTavilyResearchExecutor(url="https://mock.mcp/mcp/")

    def1 = ToolDefinition(name="tavily_search", description="d1", parameters={"type": "object"})
    def2 = ToolDefinition(name="tavily_search", description="d2", parameters={"type": "object"})

    mock_tool1 = MagicMock(spec=McpTool)
    mock_tool1.definition = def1
    mock_tool2 = MagicMock(spec=McpTool)
    mock_tool2.definition = def2

    mock_client = AsyncMock(spec=McpClient)
    mock_client.list_tools.return_value = [mock_tool1, mock_tool2]

    with patch.object(McpClient, "from_streamable_http", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        task = TaskRequested.create("test search")
        with pytest.raises(RuntimeError, match="Multiple tools found matching 'tavily_search'"):
            await executor.execute(task)


@pytest.mark.anyio
async def test_mcp_tavily_executor_rejects_speculative_aliases() -> None:
    executor = McpTavilyResearchExecutor(url="https://mock.mcp/mcp/")

    # Server publishes speculative aliases rather than exact 'tavily_search'
    tool_alias1 = MagicMock(spec=McpTool)
    tool_alias1.definition = ToolDefinition(name="tavily-search", description="d1", parameters={"type": "object"})
    tool_alias2 = MagicMock(spec=McpTool)
    tool_alias2.definition = ToolDefinition(name="search", description="d2", parameters={"type": "object"})

    mock_client = AsyncMock(spec=McpClient)
    mock_client.list_tools.return_value = [tool_alias1, tool_alias2]

    with patch.object(McpClient, "from_streamable_http", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        task = TaskRequested.create("test search")
        with pytest.raises(RuntimeError, match="Expected Tavily search tool 'tavily_search' not found"):
            await executor.execute(task)


@pytest.mark.anyio
async def test_mcp_tavily_executor_success_and_result_mapping() -> None:
    evaluator = EvidenceEvaluator()
    executor = McpTavilyResearchExecutor(
        url="https://mock.mcp/mcp/",
        headers={"Authorization": "Bearer secret"},
        evaluator=evaluator,
    )

    tavily_tool_def = ToolDefinition(
        name="tavily_search",
        description="Search tool",
        parameters={"type": "object"},
    )
    mock_tool = AsyncMock(spec=McpTool)
    mock_tool.definition = tavily_tool_def

    expected_payload = {
        "results": [
            {
                "url": "https://docs.python.org/3/",
                "title": "Python Documentation",
                "content": "Official Python 3 documentation and tutorials.",
                "score": 0.99,
                "published_date": "2024-01-01",
            }
        ],
        "answer": "Here is the official Python documentation.",
    }
    task = TaskRequested.create("python docs")
    mock_tool.execute.return_value = ToolInvocationResult.create_success(
        call_id=f"tavily-search-{task.task_id}",
        tool_name="tavily_search",
        output=expected_payload,
    )

    mock_client = AsyncMock(spec=McpClient)
    mock_client.list_tools.return_value = [mock_tool]

    with patch.object(McpClient, "from_streamable_http", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        result = await executor.execute(task)

    assert isinstance(result, ResearchResult)
    assert result.task_id == task.task_id
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://docs.python.org/3/"
    assert result.sources[0].domain == "docs.python.org"
    assert "Official Python 3 documentation" in result.output
    assert "Summary: Here is the official Python documentation." in result.output

    # Verify tool execution call
    mock_tool.execute.assert_called_once()
    req = mock_tool.execute.call_args[0][0]
    assert req.tool_name == "tavily_search"
    assert req.arguments == {"query": "python docs"}
    assert req.call_id == f"tavily-search-{task.task_id}"


@pytest.mark.anyio
async def test_mcp_tavily_executor_tool_error_fails_explicitly(caplog: pytest.LogCaptureFixture) -> None:
    error_message = "Rate limit exceeded on Tavily API"
    executor = McpTavilyResearchExecutor(url="https://mock.mcp/mcp/")

    tavily_tool_def = ToolDefinition(name="tavily_search", description="d", parameters={"type": "object"})
    mock_tool = AsyncMock(spec=McpTool)
    mock_tool.definition = tavily_tool_def
    mock_tool.execute.return_value = ToolInvocationResult.create_failure(
        call_id="tavily-search-t-err",
        tool_name="tavily_search",
        error=error_message,
    )

    mock_client = AsyncMock(spec=McpClient)
    mock_client.list_tools.return_value = [mock_tool]

    with patch.object(McpClient, "from_streamable_http", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        task = TaskRequested.create("rate limited")
        with pytest.raises(RuntimeError) as raised:
            await executor.execute(task)

    assert str(raised.value) == (
        f"Tavily MCP search failed for task {task.task_id} "
        f"(code=execution_failure): {error_message}"
    )
    assert error_message in caplog.text


@pytest.mark.anyio
async def test_mcp_tavily_executor_redacts_configured_bearer_from_tool_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "mcp-token-secret-123"
    error_message = f"Bearer {secret} rejected: ordinary provider error text"
    executor = McpTavilyResearchExecutor(
        url="https://mock.mcp/mcp/",
        headers={"Authorization": f"Bearer {secret}"},
    )

    tavily_tool_def = ToolDefinition(name="tavily_search", description="d", parameters={"type": "object"})
    mock_tool = AsyncMock(spec=McpTool)
    mock_tool.definition = tavily_tool_def
    mock_tool.execute.return_value = ToolInvocationResult.create_failure(
        call_id="tavily-search-t-redacted",
        tool_name="tavily_search",
        error=error_message,
    )

    mock_client = AsyncMock(spec=McpClient)
    mock_client.list_tools.return_value = [mock_tool]

    with patch.object(McpClient, "from_streamable_http", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        task = TaskRequested.create("redact credential")
        with pytest.raises(RuntimeError) as raised:
            await executor.execute(task)

    _assert_secrets_absent(str(raised.value), [secret])
    assert "[REDACTED]" in str(raised.value)
    assert "ordinary provider error text" in str(raised.value)
    _assert_secrets_absent(caplog.text, [secret])
    assert "[REDACTED]" in caplog.text
    assert "ordinary provider error text" in caplog.text
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize(
    "stage",
    [
        "client_creation",
        "connection_authentication",
        "list_tools",
        "tool_execution",
        "connection_close",
    ],
)
@pytest.mark.anyio
async def test_mcp_provider_exceptions_are_sanitized_without_chaining(
    stage: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = f"mcp-{stage}-canary"
    executor = McpTavilyResearchExecutor(
        headers={"Authorization": f"Basic {secret}"}
    )
    tool = _mock_tavily_tool()
    client = _mock_mcp_client(tool)
    factory_kwargs: dict[str, object] = {"return_value": client}

    if stage == "client_creation":
        factory_kwargs = {"side_effect": RuntimeError(secret)}
    elif stage == "connection_authentication":
        client.__aenter__.side_effect = RuntimeError(secret)
    elif stage == "list_tools":
        client.list_tools.side_effect = RuntimeError(secret)
    elif stage == "connection_close":
        client.__aexit__.side_effect = RuntimeError(secret)
    else:
        tool.execute.side_effect = RuntimeError(secret)

    with (
        patch.object(McpClient, "from_streamable_http", **factory_kwargs),
        pytest.raises(RuntimeError) as raised,
    ):
        await executor.execute(TaskRequested.create("safe error boundary"))

    message = str(raised.value)
    _assert_secrets_absent(message + caplog.text, [secret])
    assert stage in message
    assert "RuntimeError" in message
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize(
    ("header_name", "scheme"),
    [
        ("Authorization", "Basic"),
        ("Authorization", "Custom"),
        ("X-API-Key", ""),
        ("X-Tavily-API-Key", ""),
        ("X-Custom-Session", ""),
    ],
    ids=[
        "authorization-basic",
        "authorization-custom",
        "api-key",
        "tavily-api-key",
        "custom-header",
    ],
)
@pytest.mark.anyio
async def test_mcp_tool_error_redacts_configured_header_credentials(
    header_name: str,
    scheme: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = f"mcp-header-{header_name.lower()}-canary"
    header_value = f"{scheme} {secret}".strip()
    executor = McpTavilyResearchExecutor(headers={header_name: header_value})
    tool = _mock_tavily_tool()
    task = TaskRequested.create("redact configured header")
    tool.execute.return_value = ToolInvocationResult.create_failure(
        call_id=f"tavily-search-{task.task_id}",
        tool_name="tavily_search",
        error=f"Rejected {header_value}: request failed safely",
    )
    client = _mock_mcp_client(tool)

    with (
        patch.object(McpClient, "from_streamable_http", return_value=client),
        pytest.raises(RuntimeError) as raised,
    ):
        await executor.execute(task)

    message = str(raised.value)
    _assert_secrets_absent(message + caplog.text, [secret])
    assert "[REDACTED]" in message
    assert "Rejected" in message
    assert "request failed safely" in message
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.anyio
async def test_mcp_tool_error_redacts_url_userinfo_and_credential_query_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secrets = [
        "mcp-url-user-canary",
        "mcp-url-password-canary",
        "mcp-url-query-canary/encoded",
        "mcp-url-token-canary",
        "mcp-url-password-query-canary",
        "mcp-url-passwd-query-canary",
        "mcp-url-signature-query-canary",
        "mcp-url-sig-query-canary",
        "mcp-url-tavily-query-canary",
    ]
    url = (
        "https://mcp-url-user-canary:mcp-url-password-canary@mock.mcp/mcp/"
        "?api_key=mcp-url-query-canary%2Fencoded"
        "&access_token=mcp-url-token-canary"
        "&password=mcp-url-password-query-canary"
        "&passwd=mcp-url-passwd-query-canary"
        "&signature=mcp-url-signature-query-canary"
        "&sig=mcp-url-sig-query-canary"
        "&tavilyApiKey=mcp-url-tavily-query-canary"
        "&region=us-east-1"
        "&mode=compact"
    )
    executor = McpTavilyResearchExecutor(url=url)
    tool = _mock_tavily_tool()
    task = TaskRequested.create("redact URL credentials")
    tool.execute.return_value = ToolInvocationResult.create_failure(
        call_id=f"tavily-search-{task.task_id}",
        tool_name="tavily_search",
        error=f"Request failed at {url}: retry later",
    )
    client = _mock_mcp_client(tool)

    with (
        patch.object(McpClient, "from_streamable_http", return_value=client),
        pytest.raises(RuntimeError) as raised,
    ):
        await executor.execute(task)

    message = str(raised.value)
    _assert_secrets_absent(message + caplog.text, secrets)
    assert "[REDACTED]" in message
    assert "mock.mcp/mcp/" in message
    assert "region=us-east-1" in message
    assert "mode=compact" in message
    assert "retry later" in message


@pytest.mark.anyio
async def test_mcp_local_request_construction_error_is_not_relabelled() -> None:
    executor = McpTavilyResearchExecutor()
    tool = _mock_tavily_tool()
    client = _mock_mcp_client(tool)
    task = TaskRequested.create("preserve local programming errors")

    with (
        patch.object(McpClient, "from_streamable_http", return_value=client),
        patch.object(
            ToolInvocationRequest,
            "create",
            side_effect=ValueError("local request construction failed"),
        ),
        pytest.raises(ValueError, match="local request construction failed"),
    ):
        await executor.execute(task)

    client.__aexit__.assert_awaited_once()
    exit_args = client.__aexit__.await_args.args
    assert exit_args[0] is ValueError
    assert isinstance(exit_args[1], ValueError)


@pytest.mark.anyio
async def test_mcp_tavily_executor_unsupported_output_fails_explicitly() -> None:
    executor = McpTavilyResearchExecutor(url="https://mock.mcp/mcp/")

    tavily_tool_def = ToolDefinition(name="tavily_search", description="d", parameters={"type": "object"})
    mock_tool = AsyncMock(spec=McpTool)
    mock_tool.definition = tavily_tool_def
    mock_tool.execute.return_value = ToolInvocationResult.create_success(
        call_id="tavily-search-t-unsupp",
        tool_name="tavily_search",
        output="plain string instead of dict",
    )

    mock_client = AsyncMock(spec=McpClient)
    mock_client.list_tools.return_value = [mock_tool]

    with patch.object(McpClient, "from_streamable_http", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        task = TaskRequested.create("unsupported")
        with pytest.raises(TypeError, match="Unsupported Tavily MCP result format: expected dict, got str"):
            await executor.execute(task)


@pytest.mark.anyio
async def test_research_processor_with_mcp_tavily_executor() -> None:
    # Proves ResearchProcessor operates seamlessly with McpTavilyResearchExecutor without transport knowledge
    executor = MagicMock(spec=McpTavilyResearchExecutor)
    expected_result = ResearchResult(task_id="proc-task-1", output="Research output", sources=())
    executor.execute = AsyncMock(return_value=expected_result)

    bus = AsyncMock()
    processor = ResearchProcessor(
        agent_id="research-1",
        bus=bus,
        executor=executor,
        result_store=bus,
    )

    task = TaskRequested.create("test seamless processor")
    await processor.run_once(task)

    executor.execute.assert_called_once_with(task)
    bus.save.assert_called_once_with(expected_result)
