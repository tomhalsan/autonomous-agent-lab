"""Infrastructure adapter executing research queries via an external MCP server over Streamable HTTP."""

import copy
import logging

from agent_contracts import TaskRequested, ToolInvocationRequest
from agent_infrastructure.mcp import McpClient

from research_agent.domain.evaluator import EvidenceEvaluator
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.tavily_shared import process_tavily_search_payload

logger = logging.getLogger(__name__)

DEFAULT_TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"
DEFAULT_TAVILY_TOOL_NAME = "tavily_search"


class McpTavilyResearchExecutor:
    """Research executor that connects to a remote MCP server via Streamable HTTP,

    invokes the Tavily search tool, and delegates result evaluation to the shared pipeline.
    """

    def __init__(
        self,
        url: str = DEFAULT_TAVILY_MCP_URL,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        evaluator: EvidenceEvaluator | None = None,
        tool_name: str = DEFAULT_TAVILY_TOOL_NAME,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")
        self._url = url.strip()
        self._headers = copy.deepcopy(headers) if headers is not None else None
        self._timeout = float(timeout)
        self._evaluator = evaluator or EvidenceEvaluator()
        self._tool_name = tool_name

    @property
    def url(self) -> str:
        return self._url

    @property
    def tool_name(self) -> str:
        return self._tool_name

    def _redact_authorization_secret(self, message: str) -> str:
        if self._headers is None:
            return message

        authorization = next(
            (
                value
                for key, value in self._headers.items()
                if key.lower() == "authorization"
            ),
            None,
        )
        if authorization is None:
            return message

        parts = authorization.strip().split(maxsplit=1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return message

        secret = parts[1].split(maxsplit=1)[0]
        return message.replace(secret, "[REDACTED]") if secret else message

    async def execute(self, task: TaskRequested) -> ResearchResult:
        logger.info(
            "Executing MCP Tavily research search for task %s at %s",
            task.task_id,
            self._url,
        )
        client = McpClient.from_streamable_http(
            url=self._url,
            headers=self._headers,
            timeout=self._timeout,
        )

        async with client:
            tools = await client.list_tools()
            matching = [t for t in tools if t.definition.name == self._tool_name]
            if not matching:
                raise RuntimeError(
                    f"Expected Tavily search tool '{self._tool_name}' not found on MCP server"
                )
            if len(matching) > 1:
                raise RuntimeError(
                    f"Multiple tools found matching '{self._tool_name}' on MCP server"
                )
            tool = matching[0]

            req = ToolInvocationRequest.create(
                tool_name=self._tool_name,
                arguments={"query": task.prompt},
                call_id=f"tavily-search-{task.task_id}",
            )
            result = await tool.execute(req)

            if not result.success or result.error is not None:
                err_detail = (
                    result.error.message if result.error else "Unknown tool failure"
                )
                err_detail = self._redact_authorization_secret(err_detail)
                logger.error(
                    "Tavily MCP search failed for task %s: %s",
                    task.task_id,
                    err_detail,
                )
                raise RuntimeError(
                    f"Tavily MCP search failed for task {task.task_id}: {err_detail}"
                )

            if not isinstance(result.output, dict):
                raise TypeError(
                    f"Unsupported Tavily MCP result format: expected dict, got {type(result.output).__name__}"
                )

            return process_tavily_search_payload(
                task_id=task.task_id,
                query=task.prompt,
                payload=result.output,
                evaluator=self._evaluator,
            )
