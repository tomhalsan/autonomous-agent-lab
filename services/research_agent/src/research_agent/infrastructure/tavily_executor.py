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
            creation_error_type: str | None = None
            try:
                self._client = AsyncTavilyClient(api_key=self._api_key)
            except Exception as exc:  # noqa: BLE001 - do not retain provider errors that may contain credentials
                creation_error_type = type(exc).__name__
                logger.error(
                    "Tavily client creation failed (error_type=%s)",
                    creation_error_type,
                )
            if creation_error_type is not None:
                raise RuntimeError(
                    "Tavily client creation failed "
                    f"(error_type={creation_error_type})"
                ) from None

    def __repr__(self) -> str:
        return "TavilyResearchExecutor(api_key='[REDACTED]')"

    async def execute(self, task: TaskRequested) -> ResearchResult:
        logger.info("Executing Tavily research search for task %s", task.task_id)
        failure_message: str | None = None
        try:
            response = await self._client.search(query=task.prompt)
        except Exception as exc:  # noqa: BLE001 - sanitize all provider errors before propagation
            error_type = type(exc).__name__
            logger.error(
                "Tavily search failed for task %s (error_type=%s)",
                task.task_id,
                error_type,
            )
            failure_message = (
                f"Tavily search failed for task {task.task_id} "
                f"(error_type={error_type})"
            )
        else:
            return process_tavily_search_payload(
                task_id=task.task_id,
                query=task.prompt,
                payload=response,
                evaluator=self._evaluator,
            )

        raise RuntimeError(failure_message) from None
