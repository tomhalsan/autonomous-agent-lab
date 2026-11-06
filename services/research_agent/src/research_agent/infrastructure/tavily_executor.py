import logging
import os
from typing import Any

from agent_contracts import TaskRequested
from tavily import AsyncTavilyClient

from research_agent.domain.evaluator import EvidenceEvaluator
from research_agent.domain.evidence import CandidateEvidence
from research_agent.domain.models import ResearchResult

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

        raw_results = response.get("results") or []
        if not raw_results:
            return ResearchResult(
                task_id=task.task_id,
                output=f"No research results found for query: {task.prompt}",
                sources=(),
            )

        # 1. Map provider-specific search results into technology-neutral CandidateEvidence
        candidates: list[CandidateEvidence] = []
        for item in raw_results:
            raw_score = item.get("score")
            score_val = float(raw_score) if isinstance(raw_score, (int, float)) else None
            candidates.append(
                CandidateEvidence(
                    url=str(item.get("url") or "").strip(),
                    title=str(item.get("title") or "Untitled").strip(),
                    content=str(item.get("content") or "").strip(),
                    raw_score=score_val,
                    published_date=str(item["published_date"]) if item.get("published_date") else None,
                )
            )

        # 2. Evaluate, filter, and rank evidence using domain-quality heuristics
        evaluated_evidence = self._evaluator.evaluate(candidates, query=task.prompt)
        retained = [item for item in evaluated_evidence if item.retained]

        if not retained:
            logger.warning(
                "All %d candidate evidence items filtered for task %s",
                len(evaluated_evidence),
                task.task_id,
            )
            return ResearchResult(
                task_id=task.task_id,
                output=f"No verified research evidence passed quality criteria for query: {task.prompt}",
                sources=tuple(evaluated_evidence),
            )

        # 3. Format grounded evidence preserving domain-quality indicators alongside URLs
        lines: list[str] = [f"Research query: {task.prompt}"]
        answer = response.get("answer")
        if answer:
            lines.append(f"Summary: {answer}")

        lines.append("Sources:")
        for idx, item in enumerate(retained, start=1):
            tier_str = item.authority_tier.value
            type_str = item.source_type.value
            overlap_str = (
                f"cross-domain overlap ({item.cross_domain_overlap_count} domains)"
                if item.cross_domain_overlap_count > 0
                else "no cross-domain overlap observed"
            )
            indicators = (
                f"Domain: {item.domain} | Authority: {tier_str} ({type_str}) | "
                f"Relevance: {item.relevance_signal} | Overlap: {overlap_str}"
            )
            header = f"[{idx}] {item.title} ({item.url})" if item.url else f"[{idx}] {item.title}"
            lines.append(header)
            lines.append(f"    {indicators}")
            if item.content:
                lines.append(f"    {item.content}")

        return ResearchResult(
            task_id=task.task_id,
            output="\n".join(lines),
            sources=tuple(evaluated_evidence),
        )
