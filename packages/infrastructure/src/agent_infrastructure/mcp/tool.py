from typing import TYPE_CHECKING

from agent_contracts import (
    Tool,
    ToolDefinition,
    ToolErrorCode,
    ToolInvocationRequest,
    ToolInvocationResult,
)

if TYPE_CHECKING:
    from .client import McpClient


class McpTool(Tool):
    """Adapts an MCP tool into the protocol-neutral Tiny Agent Lab Tool protocol."""

    def __init__(self, definition: ToolDefinition, client: "McpClient") -> None:
        self._definition = definition
        self._client = client

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        if request.tool_name != self._definition.name:
            return ToolInvocationResult.create_failure(
                call_id=request.call_id,
                tool_name=request.tool_name,
                error=(
                    f"Tool name mismatch: requested {request.tool_name!r}, "
                    f"but this tool is {self._definition.name!r}"
                ),
                code=ToolErrorCode.INVALID_ARGUMENTS,
            )
        return await self._client.execute_tool(request)
