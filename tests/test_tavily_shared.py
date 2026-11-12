"""Tests for shared Tavily search payload processing and evidence evaluation."""

from research_agent.domain.evaluator import EvidenceEvaluator
from research_agent.domain.evidence import AuthorityTier
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.tavily_shared import process_tavily_search_payload


def test_process_tavily_payload_empty_results() -> None:
    evaluator = EvidenceEvaluator()
    res = process_tavily_search_payload(
        task_id="t-1",
        query="what is quantum computing",
        payload={"results": []},
        evaluator=evaluator,
    )
    assert isinstance(res, ResearchResult)
    assert res.task_id == "t-1"
    assert res.output == "No research results found for query: what is quantum computing"
    assert res.sources == ()


def test_process_tavily_payload_preserves_urls_snippets_scores_dates() -> None:
    evaluator = EvidenceEvaluator()
    payload = {
        "results": [
            {
                "url": "https://docs.python.org/3/whatsnew/3.12.html",
                "title": "What's New In Python 3.12",
                "content": "Python 3.12 features improved error messages and faster comprehension.",
                "score": 0.98,
                "published_date": "2023-10-02",
            }
        ],
        "answer": "Python 3.12 was released with notable performance upgrades.",
    }
    res = process_tavily_search_payload(
        task_id="t-2",
        query="python 3.12 features",
        payload=payload,
        evaluator=evaluator,
    )
    assert res.task_id == "t-2"
    assert len(res.sources) == 1
    source = res.sources[0]
    assert source.url == "https://docs.python.org/3/whatsnew/3.12.html"
    assert source.title == "What's New In Python 3.12"
    assert source.content == "Python 3.12 features improved error messages and faster comprehension."
    assert source.domain == "docs.python.org"
    assert source.authority_tier == AuthorityTier.HIGH
    assert source.retained is True

    # Output formatting checks
    assert "Research query: python 3.12 features" in res.output
    assert "Summary: Python 3.12 was released with notable performance upgrades." in res.output
    assert "[1] What's New In Python 3.12 (https://docs.python.org/3/whatsnew/3.12.html)" in res.output
    assert "Domain: docs.python.org | Authority: high (official_docs)" in res.output
    assert "Python 3.12 features improved error messages" in res.output


def test_process_tavily_payload_handles_malformed_items_gracefully() -> None:
    evaluator = EvidenceEvaluator()
    payload = {
        "results": [
            "not a dict item",  # non-dict
            {
                "url": "https://rfc-editor.org/rfc/rfc9110.html",
                "title": "HTTP Semantics",
                "content": "The Hypertext Transfer Protocol (HTTP) is a stateless application-level protocol.",
                "score": "invalid_score",  # invalid float
                "published_date": None,
            },
        ]
    }
    res = process_tavily_search_payload(
        task_id="t-3",
        query="http semantics rfc",
        payload=payload,
        evaluator=evaluator,
    )
    assert res.task_id == "t-3"
    assert len(res.sources) == 1
    assert res.sources[0].url == "https://rfc-editor.org/rfc/rfc9110.html"
    assert res.sources[0].domain == "rfc-editor.org"


def test_process_tavily_payload_no_retained_evidence() -> None:
    evaluator = EvidenceEvaluator()
    payload = {
        "results": [
            {
                "url": "javascript:alert(1)",  # invalid scheme rejected by evaluator
                "title": "XSS Link",
                "content": "Malicious payload link.",
            }
        ]
    }
    res = process_tavily_search_payload(
        task_id="t-4",
        query="security test",
        payload=payload,
        evaluator=evaluator,
    )
    assert res.task_id == "t-4"
    assert "No verified research evidence passed quality criteria" in res.output
    assert len(res.sources) == 1
    assert res.sources[0].retained is False
