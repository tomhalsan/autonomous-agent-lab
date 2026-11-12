"""Tests for MCP Streamable HTTP configuration, transport, and client lifecycle."""

import socket
from typing import Any

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


class _TrackedAsyncContext:
    def __init__(self, context: Any, name: str, events: list[str]) -> None:
        self._context = context
        self._name = name
        self._events = events

    async def __aenter__(self) -> Any:
        result = await self._context.__aenter__()
        self._events.append(f"{self._name}.enter")
        return result

    async def __aexit__(self, *exc_info: object) -> Any:
        try:
            result = await self._context.__aexit__(*exc_info)
        except BaseException as exc:
            if len(exc_info) > 1 and _contains_exception(exc, exc_info[1]):
                self._events.append(f"{self._name}.exit")
            else:
                self._events.append(f"{self._name}.exit_failed")
            raise
        else:
            self._events.append(f"{self._name}.exit")
            return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)


def _contains_exception(error: BaseException, expected: BaseException) -> bool:
    if error is expected:
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(_contains_exception(nested, expected) for nested in error.exceptions)
    return False


def _track_mcp_contexts(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> type[Any]:
    original_http_client = mcp_client_module.httpx2.AsyncClient
    monkeypatch.setattr(
        mcp_client_module.httpx2,
        "AsyncClient",
        lambda **kwargs: _TrackedAsyncContext(
            original_http_client(**kwargs),
            "http",
            events,
        ),
    )

    original_transport = mcp_client_module.streamable_http_client
    monkeypatch.setattr(
        mcp_client_module,
        "streamable_http_client",
        lambda url, **kwargs: _TrackedAsyncContext(
            original_transport(url, **kwargs),
            "transport",
            events,
        ),
    )

    original_session = mcp_client_module.ClientSession
    monkeypatch.setattr(
        mcp_client_module,
        "ClientSession",
        lambda *args, **kwargs: _TrackedAsyncContext(
            original_session(*args, **kwargs),
            "session",
            events,
        ),
    )
    return original_session


async def _start_mcp_server(
    task_group: Any,
    server: MCPServer,
    port: int,
) -> None:
    async def run_server() -> None:
        try:
            await server.run_streamable_http_async(host="127.0.0.1", port=port)
        except (anyio.get_cancelled_exc_class(), OSError, RuntimeError):
            pass

    task_group.start_soon(run_server)
    await anyio.sleep(0.3)


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
async def test_mcp_http_client_initialization_failure_unwinds_acquired_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    session_class = _track_mcp_contexts(monkeypatch, events)
    initialization_error = RuntimeError("MCP initialization failed")

    async def fail_initialize(self: Any) -> None:
        events.append("session.initialize")
        raise initialization_error

    monkeypatch.setattr(session_class, "initialize", fail_initialize)
    server = MCPServer("InitializationFailureServer")
    port = _get_free_port()
    client = McpClient.from_streamable_http(url=f"http://127.0.0.1:{port}/mcp", timeout=5.0)

    async with anyio.create_task_group() as tg:
        await _start_mcp_server(tg, server, port)
        with pytest.raises(RuntimeError) as raised:
            await client.__aenter__()

        assert raised.value is initialization_error
        assert "cancel scope" not in str(raised.value)
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
        assert all(
            "attempted to exit a cancel scope" not in note.lower()
            for note in getattr(raised.value, "__notes__", [])
        )
        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_mcp_http_client_initialization_cancellation_unwinds_sdk_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    session_class = _track_mcp_contexts(monkeypatch, events)
    initialization_started = anyio.Event()
    initialization_cancellations: list[BaseException] = []

    async def wait_for_cancellation(self: Any) -> None:
        events.append("session.initialize")
        initialization_started.set()
        try:
            await anyio.sleep_forever()
        except BaseException as exc:
            initialization_cancellations.append(exc)
            raise

    monkeypatch.setattr(
        session_class,
        "initialize",
        wait_for_cancellation,
    )
    server = MCPServer("InitializationCancellationServer")
    port = _get_free_port()
    client = McpClient.from_streamable_http(url=f"http://127.0.0.1:{port}/mcp", timeout=5.0)
    caught_cancellations: list[BaseException] = []
    cancel_scopes: list[anyio.CancelScope] = []
    enter_finished = anyio.Event()

    async def enter_client() -> None:
        with anyio.CancelScope() as cancel_scope:
            cancel_scopes.append(cancel_scope)
            try:
                await client.__aenter__()
            except BaseException as exc:
                caught_cancellations.append(exc)
                raise
        enter_finished.set()

    async with anyio.create_task_group() as tg:
        await _start_mcp_server(tg, server, port)
        tg.start_soon(enter_client)
        await initialization_started.wait()
        cancel_scopes[0].cancel()
        await enter_finished.wait()

        assert len(caught_cancellations) == 1
        assert isinstance(caught_cancellations[0], anyio.get_cancelled_exc_class())
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
        assert caught_cancellations[0] is initialization_cancellations[0]
        assert all(
            "attempted to exit a cancel scope" not in note.lower()
            for note in getattr(caught_cancellations[0], "__notes__", [])
        )
        tg.cancel_scope.cancel()
