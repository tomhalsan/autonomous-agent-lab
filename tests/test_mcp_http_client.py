"""Tests for MCP Streamable HTTP configuration, transport, and client lifecycle."""

import socket

import agent_infrastructure.mcp.client as mcp_client_module
import anyio
import pytest
from agent_contracts import ToolInvocationRequest, ToolInvocationResult
from agent_infrastructure.mcp import McpClient, McpHttpClientConfig
from mcp.server.mcpserver import MCPServer


def _get_free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_mcp_http_client_config_validation_and_immutability() -> None:
    config = McpHttpClientConfig(
        url="https://mcp.tavily.com/mcp/",
        headers={"Authorization": "Bearer tvly-secret-12345"},
        timeout=15.0,
        terminate_on_close=True,
    )
    assert config.url == "https://mcp.tavily.com/mcp/"
    assert config.headers == {"Authorization": "Bearer tvly-secret-12345"}
    assert config.timeout == 15.0
    assert config.terminate_on_close is True

    # 1. Header values excluded from repr
    rep = repr(config)
    assert "tvly-secret-12345" not in rep
    assert "Authorization" not in rep

    # 2. Immutability
    with pytest.raises(AttributeError):
        config.url = "https://other.com"  # type: ignore[misc]

    # 3. Validation
    with pytest.raises(ValueError, match="url must be a non-empty string"):
        McpHttpClientConfig(url="   ")

    with pytest.raises(ValueError, match="url must be a valid http or https URL"):
        McpHttpClientConfig(url="ftp://invalid.com")

    with pytest.raises(ValueError, match="url must be a valid http or https URL"):
        McpHttpClientConfig(url="not-a-url")

    with pytest.raises(TypeError, match="headers must be a dict of strings if provided"):
        McpHttpClientConfig(url="https://api.com", headers={"bad": 123})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="timeout must be a positive number"):
        McpHttpClientConfig(url="https://api.com", timeout=0)

    with pytest.raises(ValueError, match="timeout must be a positive number"):
        McpHttpClientConfig(url="https://api.com", timeout=-5.0)

    with pytest.raises(TypeError, match="terminate_on_close must be a boolean"):
        McpHttpClientConfig(url="https://api.com", terminate_on_close="yes")  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_mcp_streamable_http_client_lifecycle_and_execution() -> None:
    server = MCPServer("TestHttpMcpServer")

    @server.tool()
    def multiply(x: int, y: int) -> dict[str, int]:
        return {"product": x * y}

    port = _get_free_port()

    async def _run_server() -> None:
        try:
            await server.run_streamable_http_async(host="127.0.0.1", port=port)
        except (anyio.get_cancelled_exc_class(), OSError, RuntimeError):
            pass

    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_server)
        await anyio.sleep(0.3)

        client = McpClient.from_streamable_http(
            url=f"http://127.0.0.1:{port}/mcp",
            headers={"X-Test": "unit"},
            timeout=5.0,
        )
        assert not client.is_active

        async with client:
            assert client.is_active

            # 1. tools/list
            tools = await client.list_tools()
            assert len(tools) == 1
            tool = tools[0]
            assert tool.definition.name == "multiply"
            assert tool.definition.metadata is None

            # 2. tools/call with structured result mapping
            req = ToolInvocationRequest.create(
                tool_name="multiply",
                arguments={"x": 6, "y": 7},
                call_id="call-mul-1",
            )
            result = await tool.execute(req)
            assert isinstance(result, ToolInvocationResult)
            assert result.success is True
            assert result.call_id == "call-mul-1"
            assert result.tool_name == "multiply"
            assert result.output == {"product": 42}

        # 3. Deterministic cleanup upon exit
        assert not client.is_active

        # 4. Calling after context exit fails explicitly
        res_closed = await tool.execute(req)
        assert res_closed.success is False
        assert "not connected or session has closed" in res_closed.error.message

        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_mcp_http_client_initialization_cancellation_cleans_acquired_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class ManagedResource:
        def __init__(self, name: str, enter_result: object) -> None:
            self.name = name
            self.enter_result = enter_result

        async def __aenter__(self) -> object:
            events.append(f"{self.name}.enter")
            return self.enter_result

        async def __aexit__(self, *exc_info: object) -> None:
            events.append(f"{self.name}.exit")

    class InitializingSession(ManagedResource):
        async def initialize(self) -> None:
            events.append("session.initialize")
            raise cancellation

    http_client = ManagedResource("http", None)
    http_client.enter_result = http_client
    transport = ManagedResource("transport", ("read", "write"))
    session = InitializingSession("session", None)
    session.enter_result = session

    monkeypatch.setattr(
        mcp_client_module.httpx2,
        "AsyncClient",
        lambda **kwargs: http_client,
    )
    monkeypatch.setattr(
        mcp_client_module,
        "streamable_http_client",
        lambda *args, **kwargs: transport,
    )
    monkeypatch.setattr(
        mcp_client_module,
        "ClientSession",
        lambda read_stream, write_stream: session,
    )

    client = McpClient.from_streamable_http(url="https://example.test/mcp")
    cancellation = anyio.get_cancelled_exc_class()("cancel during initialization")

    with pytest.raises(type(cancellation)) as raised:
        await client.__aenter__()

    assert raised.value is cancellation
    assert not client.is_active
    assert events == [
        "http.enter",
        "transport.enter",
        "session.enter",
        "session.initialize",
        "session.exit",
        "transport.exit",
        "http.exit",
    ]
