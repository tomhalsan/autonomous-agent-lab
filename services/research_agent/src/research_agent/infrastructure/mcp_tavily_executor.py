"""Infrastructure adapter executing research queries via an external MCP server over Streamable HTTP."""

import copy
import logging
import sys
from urllib.parse import unquote, unquote_plus, urlsplit

from agent_contracts import TaskRequested, ToolInvocationRequest, ToolInvocationResult
from agent_infrastructure.mcp import McpClient, McpTool

from research_agent.domain.evaluator import EvidenceEvaluator
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.tavily_shared import process_tavily_search_payload

logger = logging.getLogger(__name__)

DEFAULT_TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"
DEFAULT_TAVILY_TOOL_NAME = "tavily_search"
_CREDENTIAL_QUERY_NAMES = frozenset(
    {
        "access_key",
        "access_token",
        "api_key",
        "api_token",
        "apikey",
        "auth",
        "auth_token",
        "authorization",
        "client_secret",
        "key",
        "password",
        "passwd",
        "secret",
        "sig",
        "signature",
        "tavilyapikey",
        "tavily_api_key",
        "token",
        "x_api_key",
    }
)


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

    def __repr__(self) -> str:
        return "McpTavilyResearchExecutor(url='[REDACTED]', headers='[REDACTED]')"

    def _credential_values(self) -> set[str]:
        values: set[str] = set()
        if self._headers is not None:
            for name, value in self._headers.items():
                if value:
                    values.add(value)
                    values.add(unquote(value))
                if name.lower().replace("_", "-") == "authorization":
                    parts = value.split()
                    values.update(parts[1:] if len(parts) > 1 else parts)

        try:
            parsed_url = urlsplit(self._url)
        except ValueError:
            parsed_url = None
        if parsed_url is not None:
            for value in (parsed_url.username, parsed_url.password):
                if value:
                    values.add(value)
                    values.add(unquote(value))

            for part in parsed_url.query.split("&"):
                name, separator, value = part.partition("=")
                normalized_name = unquote_plus(name).lower().replace("-", "_")
                if separator and normalized_name in _CREDENTIAL_QUERY_NAMES:
                    values.add(value)
                    values.add(unquote_plus(value))

        return {value for value in values if value}

    def _redact_credentials(self, message: str) -> str:
        for value in sorted(self._credential_values(), key=len, reverse=True):
            message = message.replace(value, "[REDACTED]")
        return message

    @staticmethod
    def _provider_failure(stage: str, exc: Exception) -> str:
        return (
            f"Tavily MCP {stage} failed "
            f"(error_type={type(exc).__name__})"
        )

    async def execute(self, task: TaskRequested) -> ResearchResult:
        logger.info("Executing MCP Tavily research search for task %s", task.task_id)
        client: McpClient | None = None
        creation_failure: str | None = None
        try:
            client = McpClient.from_streamable_http(
                url=self._url,
                headers=self._headers,
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001 - provider errors may contain credentials
            creation_failure = self._provider_failure("client_creation", exc)
        if creation_failure is not None:
            logger.error("%s for task %s", creation_failure, task.task_id)
            raise RuntimeError(creation_failure) from None
        if client is None:
            raise RuntimeError("Tavily MCP client creation returned no client") from None

        matching: list[McpTool] = []
        tool_result: ToolInvocationResult | None = None
        operation_failure: str | None = None
        try:
            await client.__aenter__()
        except Exception as exc:  # noqa: BLE001 - provider errors may contain credentials
            operation_failure = self._provider_failure(
                "connection_authentication", exc
            )
        else:
            try:
                try:
                    tools = await client.list_tools()
                except Exception as exc:  # noqa: BLE001 - provider errors may contain credentials
                    operation_failure = self._provider_failure("list_tools", exc)
                else:
                    matching = [
                        tool
                        for tool in tools
                        if tool.definition.name == self._tool_name
                    ]
                    if len(matching) == 1:
                        tool = matching[0]
                        req = ToolInvocationRequest.create(
                            tool_name=self._tool_name,
                            arguments={"query": task.prompt},
                            call_id=f"tavily-search-{task.task_id}",
                        )
                        try:
                            tool_result = await tool.execute(req)
                        except Exception as exc:  # noqa: BLE001 - provider errors may contain credentials
                            operation_failure = self._provider_failure(
                                "tool_execution", exc
                            )
            finally:
                active_exception = sys.exception()
                exit_exception = active_exception
                if exit_exception is None and operation_failure is not None:
                    exit_exception = RuntimeError(operation_failure)
                try:
                    await client.__aexit__(
                        type(exit_exception) if exit_exception is not None else None,
                        exit_exception,
                        (
                            exit_exception.__traceback__
                            if exit_exception is not None
                            else None
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - provider errors may contain credentials
                    close_failure = self._provider_failure("connection_close", exc)
                    if active_exception is not None:
                        logger.error(
                            "%s while handling a local failure for task %s",
                            close_failure,
                            task.task_id,
                        )
                    elif operation_failure is None:
                        operation_failure = close_failure
                    else:
                        logger.error(
                            "%s after %s for task %s",
                            close_failure,
                            operation_failure,
                            task.task_id,
                        )

        if operation_failure is not None:
            logger.error("%s for task %s", operation_failure, task.task_id)
            raise RuntimeError(operation_failure) from None

        if not matching:
            raise RuntimeError(
                f"Expected Tavily search tool '{self._tool_name}' not found on MCP server"
            )
        if len(matching) > 1:
            raise RuntimeError(
                f"Multiple tools found matching '{self._tool_name}' on MCP server"
            )
        if tool_result is None:
            raise RuntimeError("Tavily MCP tool execution returned no result")

        if not tool_result.success or tool_result.error is not None:
            error = tool_result.error
            error_code = error.code.value if error is not None else "unknown"
            err_detail = (
                self._redact_credentials(error.message)
                if error is not None
                else "Unknown tool failure"
            )
            logger.error(
                "Tavily MCP search failed for task %s (code=%s): %s",
                task.task_id,
                error_code,
                err_detail,
            )
            raise RuntimeError(
                f"Tavily MCP search failed for task {task.task_id} "
                f"(code={error_code}): {err_detail}"
            ) from None

        if not isinstance(tool_result.output, dict):
            raise TypeError(
                f"Unsupported Tavily MCP result format: expected dict, got {type(tool_result.output).__name__}"
            )

        return process_tavily_search_payload(
            task_id=task.task_id,
            query=task.prompt,
            payload=tool_result.output,
            evaluator=self._evaluator,
        )
