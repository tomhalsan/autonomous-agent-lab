import logging
import os
from typing import Any

from agent_contracts import TaskRequested
from tavily import AsyncTavilyClient

from research_agent.domain.evaluator import EvidenceEvaluator
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.tavily_shared import process_tavily_search_payload

logger = logging.getLogger(__name__)


class TavilyResearchExecutor:
    """Infrastructure adapter that executes research queries using the Tavily search API

    and evaluates candidate evidence against domain-quality heuristics.

    Note on responsibility placement: For Issue #9, evaluation is executed within this adapter
    to preserve existing ResearchExecutor and ResearchProcessor contracts. Application-layer
    orchestration may be reconsidered if evidence evaluation expands or more retrieval providers
    are added.
    """

    def __init__(
        self,
        api_key: str | None = None,
        client: Any | None = None,
        evaluator: EvidenceEvaluator | None = None,
    ) -> None:
        self._api_key = (api_key or os.getenv("TAVILY_API_KEY", "")).strip()
        self._evaluator = evaluator or EvidenceEvaluator()

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

        return process_tavily_search_payload(
            task_id=task.task_id,
            query=task.prompt,
            payload=response,
            evaluator=self._evaluator,
        )
