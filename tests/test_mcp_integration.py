import sys
from pathlib import Path

import pytest
from agent_contracts import (
    Tool,
    ToolErrorCode,
    ToolInvocationRequest,
    ToolInvocationResult,
)
from agent_infrastructure import InMemoryToolRegistry
from agent_infrastructure.mcp import McpClient, McpTool

SERVER_SCRIPT = str(Path(__file__).parent / "fixtures" / "mcp_test_server.py")


@pytest.mark.anyio
async def test_mcp_stdio_end_to_end_protocol_round_trip() -> None:
    """Verifies live MCP stdio connection, discovery, protocol conformance, text execution,

    structured execution, failure mapping, and deterministic lifecycle cleanup.
    """
    client = McpClient.from_stdio(
        command=sys.executable,
        args=[SERVER_SCRIPT],
        timeout=10.0,
    )
    assert not client.is_active

    async with client:
        assert client.is_active

        # 1. Tool Discovery & Contract Conformance
        tools = await client.list_tools()
        assert len(tools) == 3
        assert all(isinstance(t, Tool) for t in tools)
        assert all(isinstance(t, McpTool) for t in tools)

        tool_map = {t.definition.name: t for t in tools}
        assert set(tool_map.keys()) == {"echo", "calculate", "trigger_error"}

        # 2. Exact Tool Name & Schema Assertions
        echo_tool = tool_map["echo"]
        assert echo_tool.definition.name == "echo"
        assert echo_tool.definition.description == "Echo a message back to the caller."
        assert echo_tool.definition.parameters["type"] == "object"
        assert "message" in echo_tool.definition.parameters["properties"]
        assert echo_tool.definition.metadata is None

        calc_tool = tool_map["calculate"]
        assert calc_tool.definition.name == "calculate"
        assert calc_tool.definition.parameters["type"] == "object"
        assert set(calc_tool.definition.parameters["properties"].keys()) == {"a", "b"}
        assert calc_tool.definition.output_schema is not None
        assert calc_tool.definition.output_schema["type"] == "object"
        assert set(calc_tool.definition.output_schema["properties"].keys()) == {
            "operation",
            "result",
        }
        assert calc_tool.definition.metadata is None

        err_tool = tool_map["trigger_error"]
        assert err_tool.definition.name == "trigger_error"
        assert err_tool.definition.metadata is None

        # 3. Text Invocation Round Trip
        req_echo = ToolInvocationRequest.create(
            tool_name="echo",
            arguments={"message": "hello Tiny Agent Lab"},
            call_id="call-echo-101",
        )
        res_echo = await echo_tool.execute(req_echo)
        assert isinstance(res_echo, ToolInvocationResult)
        assert res_echo.call_id == "call-echo-101"
        assert res_echo.tool_name == "echo"
        assert res_echo.success is True
        assert res_echo.error is None
        # Pinned MCP v2 MCPServer returns structured_content {'result': ...};
        # adapter mapping prioritizes structured_content as the exact output.
        assert res_echo.output == {"result": "echo: hello Tiny Agent Lab"}

        # 4. Structured Output Invocation Round Trip
        req_calc = ToolInvocationRequest.create(
            tool_name="calculate",
            arguments={"a": 3, "b": 4},
            call_id="call-calc-202",
        )
        res_calc = await calc_tool.execute(req_calc)
        assert isinstance(res_calc, ToolInvocationResult)
        assert res_calc.call_id == "call-calc-202"
        assert res_calc.tool_name == "calculate"
        assert res_calc.success is True
        assert res_calc.error is None
        assert res_calc.output == {"operation": "add", "result": 7}

        # Verify no MCP protocol types or Pydantic models escape into ToolInvocationResult.output
        for res in (res_echo, res_calc):
            assert type(res.output) is dict
            for k, v in res.output.items():
                assert type(k) is str
                assert type(v) in (dict, list, str, int, float, bool, type(None))

        # 5. Predictable Tool Failure Round Trip
        req_err = ToolInvocationRequest.create(
            tool_name="trigger_error",
            arguments={"reason": "simulated database crash"},
            call_id="call-err-303",
        )
        res_err = await err_tool.execute(req_err)
        assert isinstance(res_err, ToolInvocationResult)
        assert res_err.call_id == "call-err-303"
        assert res_err.tool_name == "trigger_error"
        assert res_err.success is False
        assert res_err.error is not None
        assert res_err.error.code == ToolErrorCode.EXECUTION_FAILURE
        assert "Simulated tool error: simulated database crash" in res_err.error.message

        # 6. Registry Compatibility
        registry = InMemoryToolRegistry()
        reg_echo = registry.register(echo_tool, category="utility")
        assert reg_echo.name == "echo"
        assert reg_echo.category == "utility"
        assert registry.get("echo") == reg_echo

    # 7. Deterministic Cleanup Verification
    assert not client.is_active

    # 8. Post-Closure Execution Fails with UNAVAILABLE
    res_post_close = await echo_tool.execute(req_echo)
    assert res_post_close.call_id == "call-echo-101"
    assert res_post_close.tool_name == "echo"
    assert res_post_close.success is False
    assert res_post_close.error is not None
    assert res_post_close.error.code == ToolErrorCode.UNAVAILABLE
    assert "MCP client is not connected or session has closed" in res_post_close.error.message
