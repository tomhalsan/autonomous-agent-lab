import logging
import os
from typing import Any

from agent_contracts import TaskRequested
from tavily import AsyncTavilyClient

from research_agent.domain.models import ResearchResult

logger = logging.getLogger(__name__)


class TavilyResearchExecutor:
    """Infrastructure adapter that executes research queries using the Tavily search API."""

    def __init__(
        self,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._api_key = (api_key or os.getenv("TAVILY_API_KEY", "")).strip()
        if client is not None:
            self._client = client
        else:
            if not self._api_key:
                raise ValueError(
                    "TAVILY_API_KEY environment variable is required for TavilyResearchExecutor"
                )
            self._client = AsyncTavilyClient(api_key=self._api_key)

    async def execute(self, task: TaskRequested) -> ResearchResult:
        logger.info("Executing Tavily research search for task %s", task.task_id)
        try:
            response = await self._client.search(query=task.prompt)
        except Exception as exc:
            err_msg = str(exc)
            if self._api_key and self._api_key in err_msg:
                err_msg = err_msg.replace(self._api_key, "[REDACTED]")
            logger.error(
                "Tavily search failed for task %s: %s",
                task.task_id,
                err_msg,
            )
            raise RuntimeError(
                f"Tavily search failed for task {task.task_id}: {err_msg}"
            ) from exc

        results = response.get("results") or []
        if not results:
            return ResearchResult(
                task_id=task.task_id,
                output=f"No research results found for query: {task.prompt}",
            )

        lines: list[str] = [f"Research query: {task.prompt}"]
        answer = response.get("answer")
        if answer:
            lines.append(f"Summary: {answer}")

        lines.append("Sources:")
        for idx, item in enumerate(results, start=1):
            title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            content = (item.get("content") or "").strip()
            header = f"[{idx}] {title} ({url})" if url else f"[{idx}] {title}"
            lines.append(header)
            if content:
                lines.append(f"    {content}")

        return ResearchResult(
            task_id=task.task_id,
            output="\n".join(lines),
        )

