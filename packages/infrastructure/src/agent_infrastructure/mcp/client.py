import copy
import sys
from types import TracebackType
from typing import Any, Self

import anyio
import mcp.types as mcp_types
from agent_contracts import (
    ToolDefinition,
    ToolErrorCode,
    ToolInvocationRequest,
    ToolInvocationResult,
)
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent

from .server_config import McpServerConfig
from .tool import McpTool


def _map_mcp_tool_to_definition(tool: mcp_types.Tool) -> ToolDefinition:
    """Map an MCP v2 Tool object into a Tiny Agent Lab ToolDefinition."""
    name = tool.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"MCP tool name must be a non-empty string, got {name!r}")

    raw_desc = tool.description
    if raw_desc is not None and raw_desc.strip():
        description = raw_desc.strip()
    else:
        description = f"MCP tool '{name}' (no description provided by server)"

    parameters = copy.deepcopy(tool.input_schema) if tool.input_schema is not None else {}
    output_schema = (
        copy.deepcopy(tool.output_schema)
        if getattr(tool, "output_schema", None) is not None
        else None
    )

    # ToolDefinition.__post_init__ will validate schemas against the profile
    # and fail explicitly if unsupported keywords or invalid structures are present.
    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
        output_schema=output_schema,
        metadata=None,
    )


def _map_call_tool_result(
    result: CallToolResult, call_id: str, tool_name: str
) -> ToolInvocationResult:
    """Map an MCP v2 CallToolResult into a ToolInvocationResult."""
    if result.is_error is True:
        error_texts = [
            item.text for item in result.content if isinstance(item, TextContent)
        ]
        error_msg = (
            "\n".join(error_texts)
            if error_texts
            else "MCP tool reported an execution error"
        )
        return ToolInvocationResult.create_failure(
            call_id=call_id,
            tool_name=tool_name,
            error=error_msg,
            code=ToolErrorCode.EXECUTION_FAILURE,
            details={"is_error": True},
        )

    # Successful result: prioritize structured_content when present
    if result.structured_content is not None:
        structured = copy.deepcopy(result.structured_content)
        return ToolInvocationResult.create_success(
            call_id=call_id,
            tool_name=tool_name,
            output=structured,
        )

    # Content items check
    if not result.content:
        return ToolInvocationResult.create_success(
            call_id=call_id,
            tool_name=tool_name,
            output=None,
        )

    # Check that all content items are supported TextContent
    text_items: list[str] = []
    for item in result.content:
        if isinstance(item, TextContent):
            text_items.append(item.text)
        else:
            return ToolInvocationResult.create_failure(
                call_id=call_id,
                tool_name=tool_name,
                error=f"Unsupported MCP content item type: {type(item).__name__}",
                code=ToolErrorCode.EXECUTION_FAILURE,
            )

    output: Any = text_items[0] if len(text_items) == 1 else text_items

    return ToolInvocationResult.create_success(
        call_id=call_id,
        tool_name=tool_name,
        output=output,
    )


class McpClient:
    """Async context manager managing a stdio MCP client connection and session."""

    def __init__(self, config: McpServerConfig) -> None:
        if not isinstance(config, McpServerConfig):
            raise TypeError("config must be an instance of McpServerConfig")
        self._config = config
        self._stdio_ctx: Any = None
        self._session_ctx: Any = None
        self._session: ClientSession | None = None
        self._is_active = False

    @classmethod
    def from_stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> "McpClient":
        config = McpServerConfig(
            command=command,
            args=args if args is not None else [],
            env=env,
            cwd=cwd,
            timeout=timeout,
        )
        return cls(config)

    @property
    def is_active(self) -> bool:
        return self._is_active

    async def __aenter__(self) -> Self:
        params = StdioServerParameters(
            command=self._config.command,
            args=list(self._config.args),
            env=dict(self._config.env) if self._config.env is not None else None,
            cwd=self._config.cwd,
        )
        self._stdio_ctx = stdio_client(params)
        read_stream, write_stream = await self._stdio_ctx.__aenter__()

        self._session_ctx = ClientSession(read_stream, write_stream)
        session = await self._session_ctx.__aenter__()
        self._session = session
        try:
            with anyio.fail_after(self._config.timeout):
                await session.initialize()
        except Exception:
            await self.__aexit__(*sys.exc_info())
            raise
        self._is_active = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._is_active = False
        try:
            if self._session_ctx is not None:
                await self._session_ctx.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self._session = None
            self._session_ctx = None
            if self._stdio_ctx is not None:
                await self._stdio_ctx.__aexit__(exc_type, exc_val, exc_tb)
            self._stdio_ctx = None

    async def list_tools(self) -> list[McpTool]:
        """Discover tools from the MCP server and map them into Tiny Agent Lab Tool instances."""
        if not self._is_active or self._session is None:
            raise RuntimeError(
                "MCP client is not connected; enter the async context before listing tools"
            )

        with anyio.fail_after(self._config.timeout):
            result = await self._session.list_tools()

        adapted_tools: list[McpTool] = []
        for raw_tool in result.tools:
            definition = _map_mcp_tool_to_definition(raw_tool)
            adapted_tools.append(McpTool(definition=definition, client=self))
        return adapted_tools

    async def execute_tool(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        """Execute a tool invocation request across the MCP protocol boundary."""
        if not self._is_active or self._session is None:
            return ToolInvocationResult.create_failure(
                call_id=request.call_id,
                tool_name=request.tool_name,
                error="MCP client is not connected or session has closed",
                code=ToolErrorCode.UNAVAILABLE,
            )

        try:
            with anyio.fail_after(self._config.timeout):
                result = await self._session.call_tool(
                    name=request.tool_name,
                    arguments=request.arguments,
                )
        except TimeoutError:
            return ToolInvocationResult.create_failure(
                call_id=request.call_id,
                tool_name=request.tool_name,
                error=f"MCP tool execution timed out after {self._config.timeout}s",
                code=ToolErrorCode.TIMEOUT,
            )
        except MCPError as exc:
            code = ToolErrorCode.EXECUTION_FAILURE
            if exc.code == mcp_types.INVALID_PARAMS:
                code = ToolErrorCode.INVALID_ARGUMENTS
            elif exc.code == mcp_types.METHOD_NOT_FOUND:
                code = ToolErrorCode.UNAVAILABLE
            return ToolInvocationResult.create_failure(
                call_id=request.call_id,
                tool_name=request.tool_name,
                error=exc.message,
                code=code,
                details={"mcp_code": exc.code},
            )
        except (ConnectionError, OSError, anyio.ClosedResourceError) as exc:
            return ToolInvocationResult.create_failure(
                call_id=request.call_id,
                tool_name=request.tool_name,
                error=f"MCP transport connection error: {exc}",
                code=ToolErrorCode.UNAVAILABLE,
            )

        return _map_call_tool_result(
            result, call_id=request.call_id, tool_name=request.tool_name
        )
