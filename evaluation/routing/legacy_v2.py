"""Frozen historical baseline router for Routing v2.

Preserved in the evaluation boundary for benchmark comparison and historical reproduction.
Does not run in production.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum

from orchestrator.domain.assessment import (
    RoutingAssessment,
    RoutingClarity,
)
from orchestrator.domain.capability import AgentCapability


class LegacyResearchIntentCategory(str, Enum):
    """Semantic intent categories from historical Routing v2."""

    RESEARCH_AND_INVESTIGATION = "research_and_investigation"
    FRESHNESS_AND_CURRENT_INFO = "freshness_and_current_info"
    SOURCES_AND_EVIDENCE = "sources_and_evidence"


LEGACY_RESEARCH_INTENT_PATTERNS: Mapping[LegacyResearchIntentCategory, list[str]] = {
    LegacyResearchIntentCategory.RESEARCH_AND_INVESTIGATION: [
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
    LegacyResearchIntentCategory.FRESHNESS_AND_CURRENT_INFO: [
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
    LegacyResearchIntentCategory.SOURCES_AND_EVIDENCE: [
        r"\b(find|search\s+for|look\s+up|gather|collect|provide|retrieve|cite|give\s+me)\s+(sources?|citations?|references?|papers?|articles?|literature)\b(?!\s+code\b)",
        r"\b(sources?|citations?|references?|literature)\s+(about|on|for|regarding)\b",
        r"\b(find|search\s+for|look\s+up|gather|collect|provide|retrieve)\s+evidence\b",
        r"\bevidence\s+(for|against|about|on|regarding|of)\b",
        r"\bempirical\s+evidence\b",
        r"\b(find|search\s+for|look\s+up|retrieve|gather)\s+(current|recent|up-?to-?date)\s+information\b",
    ],
}


class LegacyRoutingV2Router:
    """Frozen historical baseline router for Routing v2 (Phase A baseline)."""

    def __init__(self) -> None:
        self._category_patterns = {
            category: re.compile(r"|".join(f"({p})" for p in patterns), re.IGNORECASE)
            for category, patterns in LEGACY_RESEARCH_INTENT_PATTERNS.items()
        }

    def detect_intent(self, prompt: str) -> LegacyResearchIntentCategory | None:
        """Detect research intent category using historical Routing v2 patterns."""
        for category, pattern in self._category_patterns.items():
            if pattern.search(prompt):
                return category
        return None

    def route(self, prompt: str) -> AgentCapability:
        """Route prompt based purely on intent detection (Routing v2 behavior)."""
        if self.detect_intent(prompt) is not None:
            return AgentCapability.RESEARCH
        return AgentCapability.GENERAL

    def assess(self, prompt: str) -> RoutingAssessment:
        """Produce a RoutingAssessment for evaluation reporting."""
        intent = self.detect_intent(prompt)
        if intent is not None:
            return RoutingAssessment(
                provisional_capability=AgentCapability.RESEARCH,
                clarity=RoutingClarity.CLEAR,
                signals=(),
                reason=f"Legacy Routing v2 intent: {intent.value}",
            )
        return RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.CLEAR,
            signals=(),
            reason="Legacy Routing v2 general fallback",
        )
