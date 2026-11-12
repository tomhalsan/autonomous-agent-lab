from unittest.mock import AsyncMock

import mcp.types as mcp_types
import pytest
from agent_contracts import (
    Tool,
    ToolDefinition,
    ToolErrorCode,
    ToolInvocationRequest,
    ToolInvocationResult,
)
from agent_infrastructure import InMemoryToolRegistry
from agent_infrastructure.mcp import McpClient, McpServerConfig, McpTool
from agent_infrastructure.mcp.client import (
    _map_call_tool_result,
    _map_mcp_tool_to_definition,
)
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, ImageContent, TextContent


def test_mcp_server_config_validation_and_immutability() -> None:
    config = McpServerConfig(
        command="python",
        args=["-m", "demo"],
        env={"VAR": "1"},
        cwd="/tmp",
        timeout=15.0,
    )
    assert config.command == "python"
    assert config.args == ["-m", "demo"]
    assert config.env == {"VAR": "1"}
    assert config.cwd == "/tmp"
    assert config.timeout == 15.0

    # Immutability
    with pytest.raises(AttributeError):
        config.command = "node"  # type: ignore[misc]

    # Validation
    with pytest.raises(ValueError, match="command must be a non-empty string"):
        McpServerConfig(command="   ")
    with pytest.raises(TypeError, match="args must be a list of strings"):
        McpServerConfig(command="python", args="not-a-list")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="env must be a dict of strings"):
        McpServerConfig(command="python", env={"VAR": 123})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="cwd must be a non-empty string"):
        McpServerConfig(command="python", cwd="   ")
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        McpServerConfig(command="python", timeout=0)


def test_map_mcp_tool_preserves_exact_name_and_maps_schemas() -> None:
    raw = mcp_types.Tool(
        name="Exact_Name.v1-custom",
        description="A sample tool description.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        },
    )
    definition = _map_mcp_tool_to_definition(raw)
    assert definition.name == "Exact_Name.v1-custom"
    assert definition.description == "A sample tool description."
    assert definition.parameters == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    assert definition.output_schema == {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
    }
    assert definition.metadata is None


def test_map_mcp_tool_description_fallback() -> None:
    raw_none = mcp_types.Tool(
        name="fetch_data",
        description=None,
        input_schema={"type": "object"},
    )
    def_none = _map_mcp_tool_to_definition(raw_none)
    assert def_none.name == "fetch_data"
    assert def_none.description == "MCP tool 'fetch_data' (no description provided by server)"

    raw_empty = mcp_types.Tool(
        name="fetch_data",
        description="   ",
        input_schema={"type": "object"},
    )
    def_empty = _map_mcp_tool_to_definition(raw_empty)
    assert def_empty.description == "MCP tool 'fetch_data' (no description provided by server)"


def test_map_mcp_tool_invalid_name_fails() -> None:
    raw = mcp_types.Tool(
        name="   ",
        description="Description",
        input_schema={"type": "object"},
    )
    with pytest.raises(ValueError, match="MCP tool name must be a non-empty string"):
        _map_mcp_tool_to_definition(raw)


def test_map_mcp_tool_non_object_schema_fails_explicitly() -> None:
    # Tiny Agent Lab tool-schema profile requires input_schema type="object"
    raw = mcp_types.Tool(
        name="bad_schema_tool",
        description="Tool with invalid array input schema",
        input_schema={"type": "array"},
    )
    with pytest.raises(ValueError, match="parameters\\['type'\\] must be 'object'"):
        _map_mcp_tool_to_definition(raw)


def test_map_mcp_tool_registration_with_explicit_category() -> None:
    raw = mcp_types.Tool(
        name="search_documents",
        description="Search internal docs",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    definition = _map_mcp_tool_to_definition(raw)
    assert definition.metadata is None

    client = McpClient.from_stdio("python")
    tool = McpTool(definition=definition, client=client)
    assert isinstance(tool, Tool)

    registry = InMemoryToolRegistry()
    # Unclassified tool requires explicit category at registration
    registration = registry.register(tool, category="search")
    assert registration.category == "search"
    assert registration.name == "search_documents"
    assert registry.get("search_documents") == registration
    assert registry.find_by_category("search") == [registration]


@pytest.mark.anyio
async def test_mcp_tool_execute_validates_tool_name_match() -> None:
    definition = ToolDefinition(
        name="expected_tool",
        description="Expected tool description",
        parameters={"type": "object"},
    )
    client = McpClient.from_stdio("python")
    tool = McpTool(definition=definition, client=client)

    # Calling with mismatched tool name fails with INVALID_ARGUMENTS and preserves call_id
    req = ToolInvocationRequest.create(
        tool_name="different_tool",
        arguments={},
        call_id="call-mismatch-1",
    )
    result = await tool.execute(req)
    assert isinstance(result, ToolInvocationResult)
    assert result.call_id == "call-mismatch-1"
    assert result.tool_name == "different_tool"
    assert result.success is False
    assert result.error is not None
    assert result.error.code == ToolErrorCode.INVALID_ARGUMENTS
    assert "Tool name mismatch" in result.error.message


@pytest.mark.anyio
async def test_mcp_tool_execute_when_closed_returns_unavailable() -> None:
    definition = ToolDefinition(
        name="ping_tool",
        description="Ping tool description",
        parameters={"type": "object"},
    )
    client = McpClient.from_stdio("python")
    tool = McpTool(definition=definition, client=client)

    req = ToolInvocationRequest.create(
        tool_name="ping_tool",
        arguments={},
        call_id="call-closed-1",
    )
    result = await tool.execute(req)
    assert result.call_id == "call-closed-1"
    assert result.tool_name == "ping_tool"
    assert result.success is False
    assert result.error is not None
    assert result.error.code == ToolErrorCode.UNAVAILABLE
    assert "MCP client is not connected or session has closed" in result.error.message


def test_map_call_tool_result_structured_content_priority() -> None:
    res = CallToolResult(
        content=[TextContent(type="text", text='{"status": "ok"}')],
        structured_content={"status": "ok", "items": [1, 2]},
        is_error=False,
    )
    mapped = _map_call_tool_result(res, call_id="c-1", tool_name="fetch")
    assert mapped.success is True
    assert mapped.call_id == "c-1"
    assert mapped.tool_name == "fetch"
    assert mapped.output == {"status": "ok", "items": [1, 2]}


def test_map_call_tool_result_single_and_multiple_text_blocks() -> None:
    # Single text block
    res_single = CallToolResult(
        content=[TextContent(type="text", text="Single line result")],
        is_error=False,
    )
    mapped_single = _map_call_tool_result(res_single, call_id="c-single", tool_name="greet")
    assert mapped_single.success is True
    assert mapped_single.output == "Single line result"

    # Multiple text blocks preserved as list
    res_multi = CallToolResult(
        content=[
            TextContent(type="text", text="Block 1"),
            TextContent(type="text", text="Block 2"),
        ],
        is_error=False,
    )
    mapped_multi = _map_call_tool_result(res_multi, call_id="c-multi", tool_name="multi")
    assert mapped_multi.success is True
    assert mapped_multi.output == ["Block 1", "Block 2"]


def test_map_call_tool_result_unsupported_content_fails_explicitly() -> None:
    res_unsupported = CallToolResult(
        content=[ImageContent(type="image", data="base64data", mime_type="image/png")],
        is_error=False,
    )
    mapped = _map_call_tool_result(res_unsupported, call_id="c-img", tool_name="render")
    assert mapped.success is False
    assert mapped.error is not None
    assert mapped.error.code == ToolErrorCode.EXECUTION_FAILURE
    assert "Unsupported MCP content item type: ImageContent" in mapped.error.message


def test_map_call_tool_result_tool_reported_error() -> None:
    res_err = CallToolResult(
        content=[TextContent(type="text", text="Database connection timeout")],
        is_error=True,
    )
    mapped = _map_call_tool_result(res_err, call_id="c-err", tool_name="query")
    assert mapped.success is False
    assert mapped.error is not None
    assert mapped.error.code == ToolErrorCode.EXECUTION_FAILURE
    assert mapped.error.message == "Database connection timeout"
    assert mapped.error.details == {"is_error": True}


def test_map_call_tool_result_does_not_infer_invalid_arguments_from_error_text() -> None:
    res_val_err = CallToolResult(
        content=[TextContent(type="text", text="Error executing tool: validation error for args")],
        is_error=True,
    )
    mapped = _map_call_tool_result(res_val_err, call_id="c-val", tool_name="query")
    assert mapped.success is False
    assert mapped.error is not None
    assert mapped.error.code == ToolErrorCode.EXECUTION_FAILURE
    assert mapped.error.message == "Error executing tool: validation error for args"
    assert mapped.error.details == {"is_error": True}


@pytest.mark.anyio
async def test_mcp_client_reserves_invalid_arguments_for_protocol_mcp_error() -> None:
    client = McpClient.from_stdio("python")
    client._is_active = True
    client._session = AsyncMock()
    client._session.call_tool = AsyncMock(
        side_effect=MCPError(
            code=mcp_types.INVALID_PARAMS, message="Invalid parameters passed to tool"
        )
    )

    req = ToolInvocationRequest.create(
        tool_name="test_tool", arguments={"bad": "val"}, call_id="c-proto-1"
    )
    result = await client.execute_tool(req)
    assert result.success is False
    assert result.call_id == "c-proto-1"
    assert result.tool_name == "test_tool"
    assert result.error is not None
    assert result.error.code == ToolErrorCode.INVALID_ARGUMENTS
    assert result.error.message == "Invalid parameters passed to tool"
    assert result.error.details == {"mcp_code": -32602}
