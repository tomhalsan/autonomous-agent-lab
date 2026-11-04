import re

from orchestrator.domain.capability import AgentCapability

DEFAULT_RESEARCH_KEYWORDS: frozenset[str] = frozenset({
    "analyse",
    "analysed",
    "analyses",
    "analysing",
    "analysis",
    "analyze",
    "analyzed",
    "analyzes",
    "analyzing",
    "assess",
    "assessed",
    "assesses",
    "assessing",
    "assessment",
    "assessments",
    "benchmark",
    "benchmarked",
    "benchmarking",
    "benchmarks",
    "comparative",
    "compare",
    "compared",
    "compares",
    "comparing",
    "comparison",
    "comparisons",
    "evaluate",
    "evaluated",
    "evaluates",
    "evaluating",
    "evaluation",
    "evaluations",
    "exploration",
    "explorations",
    "explore",
    "explored",
    "explores",
    "exploring",
    "investigate",
    "investigated",
    "investigates",
    "investigating",
    "investigation",
    "investigations",
    "research",
    "researched",
    "researches",
    "researching",
    "studied",
    "studies",
    "study",
    "studying",
    "survey",
    "surveyed",
    "surveying",
    "surveys",
})


class DeterministicCapabilityRouter:
    """Routes prompts to agent capabilities using deterministic keyword matching."""

    def __init__(self, keywords: frozenset[str] | set[str] | None = None) -> None:
        self._keywords = (
            frozenset(k.lower() for k in keywords)
            if keywords is not None
            else DEFAULT_RESEARCH_KEYWORDS
        )
        escaped_keywords = [
            re.escape(k) for k in sorted(self._keywords, key=len, reverse=True)
        ]
        pattern = r"\b(" + "|".join(escaped_keywords) + r")\b"
        self._pattern = re.compile(pattern, re.IGNORECASE)

    def route(self, prompt: str) -> AgentCapability:
        if self._pattern.search(prompt):
            return AgentCapability.RESEARCH
        return AgentCapability.GENERAL

