import re
from collections.abc import Mapping
from enum import Enum

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


class ResearchIntentCategory(str, Enum):
    """Semantic intent categories for Routing v2."""

    RESEARCH_AND_INVESTIGATION = "research_and_investigation"
    FRESHNESS_AND_CURRENT_INFO = "freshness_and_current_info"
    SOURCES_AND_EVIDENCE = "sources_and_evidence"


RESEARCH_INTENT_PATTERNS: Mapping[ResearchIntentCategory, list[str]] = {
    ResearchIntentCategory.RESEARCH_AND_INVESTIGATION: [
        r"\bresearch(ed|es|ing)?\b",
        r"\binvestigat(e|ed|es|ing|ion|ions)\b",
        r"\b(conduct|perform|do|undertake)\b.*\bresearch\b",
        r"\bdeep dive\b",
        r"\bliterature review\b",
        r"\b(detailed|systematic|comprehensive)\s+study\b",
        r"\bstudy of\b",
        r"\b(perform|conduct|undertake)\s+(a\s+)?study\b",
        r"\bstud(y|ies|ied|ying)\b",
        r"\bbenchmark(s|ed|ing)?\b",
        r"\bsurvey(s|ed|ing)?\b",
        r"\bcompar(e|ed|es|ing|ison|isons|ative)\b",
        r"\banaly(ze|zed|zes|zing|sis|ses|ce|ced|ces|cing)\b",
        r"\bevaluat(e|ed|es|ing|ion|ions)\b",
        r"\bassess(ed|es|ing|ment|ments)?\b",
        r"\bexplor(e|ed|es|ing|ation|ations)\b",
    ],
    ResearchIntentCategory.FRESHNESS_AND_CURRENT_INFO: [
        r"\blatest\s+(developments?|news|trends?|updates?|findings?|releases?|advancements?|breakthroughs?|reports?|data|information|research|status|state)\b",
        r"\blatest\s+(on|about|regarding)\b",
        r"\bwhat('s|\s+is|\s+are)\s+(the\s+)?latest\b",
        r"\bcurrent\s+(information|state|status|trends?|market|data|developments?|landscape|adoption|events?|news|progress|statistics)\b",
        r"\bcurrent\s+(information|data|details|status)\s+(about|on|regarding|for)\b",
        r"\bwhat('s|\s+is|\s+are)\s+(the\s+)?current\s+(state|status|situation|landscape)\b",
        r"\brecent\s+(developments?|news|trends?|updates?|advancements?|breakthroughs?|changes|events?|reports?|findings?)\b",
        r"\bup-?to-?date\s+(information|data|details|status|overview|developments?|news)\b",
        r"\bstate-?of-?the-?art\b",
        r"\bas of\s+(today|now|recently|202[4-9])\b",
        r"\breal-?time\s+(information|updates?|data|status)\b",
    ],
    ResearchIntentCategory.SOURCES_AND_EVIDENCE: [
        r"\b(find|search\s+for|look\s+up|gather|collect|provide|retrieve|cite|give\s+me)\s+(sources?|citations?|references?|papers?|articles?|literature)\b(?!\s+code\b)",
        r"\b(sources?|citations?|references?|literature)\s+(about|on|for|regarding)\b",
        r"\b(find|search\s+for|look\s+up|gather|collect|provide|retrieve)\s+evidence\b",
        r"\bevidence\s+(for|against|about|on|regarding|of)\b",
        r"\bempirical\s+evidence\b",
        r"\b(find|search\s+for|look\s+up|retrieve|gather)\s+(current|recent|up-?to-?date)\s+information\b",
    ],
}


class DeterministicCapabilityRouter:
    """Routes prompts to agent capabilities using intent-aware deterministic matching."""

    def __init__(
        self,
        keywords: frozenset[str] | set[str] | None = None,
        intent_patterns: Mapping[ResearchIntentCategory, list[str]] | None = None,
    ) -> None:
        self._custom_keywords = (
            frozenset(k.lower() for k in keywords) if keywords is not None else None
        )
        if self._custom_keywords is not None:
            escaped_keywords = [
                re.escape(k) for k in sorted(self._custom_keywords, key=len, reverse=True)
            ]
            pattern_str = r"\b(" + "|".join(escaped_keywords) + r")\b"
            self._custom_pattern: re.Pattern[str] | None = re.compile(pattern_str, re.IGNORECASE)
            self._category_patterns: dict[ResearchIntentCategory, re.Pattern[str]] = {}
        else:
            self._custom_pattern = None
            raw_patterns = intent_patterns or RESEARCH_INTENT_PATTERNS
            self._category_patterns = {
                category: re.compile(r"|".join(f"({p})" for p in patterns), re.IGNORECASE)
                for category, patterns in raw_patterns.items()
            }

    def detect_intent(self, prompt: str) -> ResearchIntentCategory | None:
        """Detect the research intent category for a prompt, or return None if general."""
        if self._custom_pattern is not None:
            if self._custom_pattern.search(prompt):
                return ResearchIntentCategory.RESEARCH_AND_INVESTIGATION
            return None

        for category, pattern in self._category_patterns.items():
            if pattern.search(prompt):
                return category
        return None

    def route(self, prompt: str) -> AgentCapability:
        """Route prompt to AgentCapability.RESEARCH or AgentCapability.GENERAL."""
        if self.detect_intent(prompt) is not None:
            return AgentCapability.RESEARCH
        return AgentCapability.GENERAL
