"""Opt-in live external validation against the official Tavily remote MCP server.

Excluded from default CI via the 'external_provider' marker.
Run explicitly with:
    pytest tests/test_tavily_mcp_live.py -m external_provider -s
"""

import os

import pytest
from agent_contracts import TaskRequested
from agent_infrastructure.mcp import McpClient
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.mcp_tavily_executor import (
    DEFAULT_TAVILY_MCP_URL,
    McpTavilyResearchExecutor,
)


@pytest.mark.external_provider
@pytest.mark.anyio
async def test_live_tavily_mcp_server_contract() -> None:
    """Connect to official hosted Tavily MCP endpoint over Streamable HTTP and record live contract evidence."""
    endpoint_url = os.getenv("TAVILY_MCP_URL", DEFAULT_TAVILY_MCP_URL).strip()
    api_key = os.getenv("TAVILY_API_KEY", "").strip()

    print(f"\n[LIVE PROBE] Target endpoint: {endpoint_url}")

    # Phase 1: Attempt documented keyless access (no Authorization header)
    print("[LIVE PROBE] Phase 1: Testing keyless access without credentials...")
    client_keyless = McpClient.from_streamable_http(url=endpoint_url, timeout=15.0)

    keyless_succeeded = False
    active_headers: dict[str, str] | None = None
    try:
        async with client_keyless:
            tools = await client_keyless.list_tools()
            tool_names = [t.definition.name for t in tools]
            print(f"[LIVE PROBE] Keyless tool discovery succeeded. Published tools: {tool_names}")
            keyless_succeeded = True
    except (OSError, RuntimeError, TimeoutError, BaseExceptionGroup) as exc:
        print(
            f"[LIVE PROBE] Unauthenticated/keyless initialization did not succeed in this observed live validation: {exc}"
        )

    # Phase 2: If keyless failed and TAVILY_API_KEY is available, use Bearer header
    if not keyless_succeeded:
        if not api_key:
            pytest.skip(
                f"Unauthenticated/keyless initialization did not succeed against {endpoint_url} "
                "in this observed live validation; bearer-authenticated access succeeded when "
                "TAVILY_API_KEY is configured, but TAVILY_API_KEY is absent here."
            )
        print("[LIVE PROBE] Phase 2: Connecting with configured TAVILY_API_KEY (redacted in output)...")
        active_headers = {"Authorization": f"Bearer {api_key}"}

    # Phase 3: Detailed tool inspection and query execution
    client = McpClient.from_streamable_http(url=endpoint_url, headers=active_headers, timeout=20.0)
    async with client:
        tools = await client.list_tools()
        print(f"[LIVE PROBE] Total tools discovered: {len(tools)}")
        for t in tools:
            print(f"  - Tool: {t.definition.name!r}")
            print(f"    Description: {t.definition.description[:80]}...")
            print(f"    Parameters: {list(t.definition.parameters.get('properties', {}).keys())}")
            if t.definition.output_schema:
                print(f"    Output Schema: {t.definition.output_schema}")

        matching_tools = [t for t in tools if t.definition.name == "tavily_search"]
        assert matching_tools, "Expected exact 'tavily_search' tool on live Tavily MCP server"
        target_tool = matching_tools[0]
        exact_tool_name = target_tool.definition.name
        print(f"[LIVE PROBE] Selected exact search tool: {exact_tool_name!r}")

    # Phase 4: Execute via McpTavilyResearchExecutor and evaluate evidence
    executor = McpTavilyResearchExecutor(
        url=endpoint_url,
        headers=active_headers,
        timeout=25.0,
        tool_name=exact_tool_name,
    )
    task = TaskRequested.create("Python 3.12 release notes official docs")
    result = await executor.execute(task)

    assert isinstance(result, ResearchResult)
    print(f"[LIVE PROBE] ResearchResult task_id: {result.task_id}")
    print(f"[LIVE PROBE] Total candidate sources returned: {len(result.sources)}")
    print(f"[LIVE PROBE] Retained sources: {len(result.retained_sources)}")

    for idx, src in enumerate(result.retained_sources, 1):
        print(f"  [{idx}] {src.title} -> {src.url} (Domain: {src.domain}, Authority: {src.authority_tier})")
        assert src.url.startswith("http://") or src.url.startswith("https://")
        assert src.domain

    print("\n[LIVE PROBE SUCCESS] Full live Tavily MCP Streamable HTTP flow verified.")
