"""Shared pure processing and evidence mapping for Tavily search payloads."""

import logging
from typing import Any

from research_agent.domain.evaluator import EvidenceEvaluator
from research_agent.domain.evidence import CandidateEvidence
from research_agent.domain.models import ResearchResult

logger = logging.getLogger(__name__)


def process_tavily_search_payload(
    task_id: str,
    query: str,
    payload: dict[str, Any],
    evaluator: EvidenceEvaluator,
) -> ResearchResult:
    """Map raw Tavily search payload into CandidateEvidence, evaluate domain quality,

    and format the resulting ResearchResult.

    Used identically by both direct Tavily SDK and MCP-mediated search paths.
    """
    raw_results = payload.get("results") or []
    if not raw_results:
        return ResearchResult(
            task_id=task_id,
            output=f"No research results found for query: {query}",
            sources=(),
        )

    # 1. Map provider-specific search results into technology-neutral CandidateEvidence
    candidates: list[CandidateEvidence] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
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
    evaluated_evidence = evaluator.evaluate(candidates, query=query)
    retained = [item for item in evaluated_evidence if item.retained]

    if not retained:
        logger.warning(
            "All %d candidate evidence items filtered for task %s",
            len(evaluated_evidence),
            task_id,
        )
        return ResearchResult(
            task_id=task_id,
            output=f"No verified research evidence passed quality criteria for query: {query}",
            sources=tuple(evaluated_evidence),
        )

    # 3. Format grounded evidence preserving domain-quality indicators alongside URLs
    lines: list[str] = [f"Research query: {query}"]
    answer = payload.get("answer")
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
        task_id=task_id,
        output="\n".join(lines),
        sources=tuple(evaluated_evidence),
    )
