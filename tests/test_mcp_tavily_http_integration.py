"""Deterministic offline integration test for McpTavilyResearchExecutor using local Streamable HTTP server."""

import socket
from typing import Any

import anyio
import pytest
from agent_contracts import TaskRequested
from mcp.server.mcpserver import MCPServer
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.mcp_tavily_executor import McpTavilyResearchExecutor


def _get_free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.mark.anyio
async def test_mcp_tavily_streamable_http_end_to_end_protocol_round_trip() -> None:
    """Verifies live Streamable HTTP connection, tavily_search discovery, parameter passing,

    structured content mapping, and EvidenceEvaluator pipeline execution with zero public network.
    """
    server = MCPServer("LocalTavilyMockServer")

    @server.tool()
    def tavily_search(query: str) -> dict[str, Any]:
        return {
            "results": [
                {
                    "url": "https://kubernetes.io/docs/concepts/",
                    "title": "Kubernetes Concepts Overview",
                    "content": "Kubernetes is an open-source system for automating deployment, scaling, and management.",
                    "score": 0.98,
                    "published_date": "2024-02-15",
                }
            ],
            "answer": "Kubernetes coordinates a highly available cluster of connected computers.",
        }

    port = _get_free_port()

    async def _run_server() -> None:
        try:
            await server.run_streamable_http_async(host="127.0.0.1", port=port)
        except (anyio.get_cancelled_exc_class(), OSError, RuntimeError):
            pass

    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_server)
        await anyio.sleep(0.3)

        executor = McpTavilyResearchExecutor(
            url=f"http://127.0.0.1:{port}/mcp",
            headers={"Authorization": "Bearer mock-test-token"},
            timeout=5.0,
        )

        task = TaskRequested.create("kubernetes architecture overview")
        result = await executor.execute(task)

        assert isinstance(result, ResearchResult)
        assert result.task_id == task.task_id
        assert len(result.sources) == 1

        source = result.sources[0]
        assert source.url == "https://kubernetes.io/docs/concepts/"
        assert source.domain == "kubernetes.io"
        assert source.retained is True
        assert source.title == "Kubernetes Concepts Overview"

        # Evidence pipeline assertions
        assert "Research query: kubernetes architecture overview" in result.output
        assert "Summary: Kubernetes coordinates a highly available cluster" in result.output
        assert "[1] Kubernetes Concepts Overview (https://kubernetes.io/docs/concepts/)" in result.output
        assert "Domain: kubernetes.io | Authority: high (official_docs)" in result.output

        tg.cancel_scope.cancel()
