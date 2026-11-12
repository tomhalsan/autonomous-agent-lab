"""Deterministic local MCP v2 stdio test server for Tiny Agent Lab integration tests."""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

server = MCPServer("TinyAgentLabTestServer")


class CalculationOutput(BaseModel):
    operation: str
    result: int


@server.tool()
def echo(message: str) -> str:
    """Echo a message back to the caller."""
    return f"echo: {message}"


@server.tool()
def calculate(a: int, b: int) -> CalculationOutput:
    """Calculate the sum of two integers with structured output."""
    return CalculationOutput(operation="add", result=a + b)


@server.tool()
def trigger_error(reason: str) -> str:
    """Predictable failure tool that always raises an error."""
    raise ToolError(f"Simulated tool error: {reason}")


if __name__ == "__main__":
    server.run(transport="stdio")
