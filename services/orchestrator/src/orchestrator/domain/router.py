"""Deterministic capability router with transparent RoutingAssessment and ambiguity analysis."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum

from orchestrator.domain.assessment import (
    RoutingAssessment,
    RoutingClarity,
    RoutingSignal,
    RoutingSignalType,
    SignalStrength,
)
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
    """Semantic intent categories for Routing v2 compatibility."""

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
        r"\blatest\s+(developments?|news|trends?|updates?|findings?|releases?|advancements?|breakthroughs?|reports?|data|information|research|status|state|cves?|vulnerabilities?|pricing)\b",
        r"\blatest\s+(on|about|regarding)\b",
        r"\bwhat('s|\s+is|\s+are)\s+(the\s+)?latest\b",
        r"\bcurrent\s+(information|state|status|trends?|market|data|developments?|landscape|adoption|events?|news|progress|statistics|prime\s+minister|president|ceo|leader|pricing|features?|release\s+status|guidance|standards?)\b",
        r"\bcurrent\s+(information|data|details|status)\s+(about|on|regarding|for)\b",
        r"\bwhat('s|\s+is|\s+are)\s+(the\s+)?current\s+(state|status|situation|landscape)\b",
        r"\brecent\s+(developments?|news|trends?|updates?|advancements?|breakthroughs?|changes|events?|reports?|findings?|outages?|cves?)\b",
        r"\bmost\s+recent\s+(cves?|vulnerabilities?|developments?|releases?|announcements?|outages?)\b",
        r"\bup-?to-?date\s+(information|data|details|status|overview|developments?|news|pricing)\b",
        r"\bstate-?of-?the-?art\b",
        r"\bas of\s+(today|now|recently|202[4-9])\b",
        r"\breal-?time\s+(information|updates?|data|status)\b",
        r"\b(this\s+week|this\s+month|trending\s+.*?\s+this\s+month)\b",
    ],
    ResearchIntentCategory.SOURCES_AND_EVIDENCE: [
        r"\b(find|search\s+for|look\s+up|gather|collect|provide|retrieve|cite|give\s+me)\s+(sources?|citations?|references?|papers?|articles?|literature)\b(?!\s+code\b)",
        r"\b(sources?|citations?|references?|literature)\s+(about|on|for|regarding)\b",
        r"\b(find|search\s+for|look\s+up|gather|collect|provide|retrieve)\s+evidence\b",
        r"\bevidence\s+(for|against|about|on|regarding|of)\b",
        r"\b(primary\s+sources?|empirical\s+(studies|evidence)|peer-?reviewed|medical\s+literature|whitepapers?)\b",
        r"\b(find|search\s+for|look\s+up|retrieve|gather)\s+(current|recent|up-?to-?date)\s+information\b",
    ],
}


# --- Negative Guards for False Freshness Matches and Diagnostic Evidence ---

NEGATIVE_CURRENT_PROGRAM_STATE = re.compile(
    r"\bcurrent_[a-zA-Z0-9_]+\b|\bcurrent\s+(variable|pointer|index|iteration|loop|thread|directory|dir|folder|branch|commit|function|line|value|item|element|node|state|session|user)\b|\b(in|of|within|check|examine|read|inspect)\s+(the\s+)?current\s+(file|function|class|method|module|component|script|buffer|context|session|execution)\b",
    re.IGNORECASE,
)

NEGATIVE_CURRENT_CAMEL_CASE = re.compile(
    r"\bcurrent[A-Z][a-zA-Z0-9_]*\b",
)

NEGATIVE_CURRENT_ELECTRICAL_OR_PHYSICS = re.compile(
    r"\b(electric|electrical|direct|alternating|eddy|leakage|induced)\s+current\b|\bcurrent\s+(flow|flows|density|draw|rating|measurement|leakage)\b|\bcurrent\s+of\s+\d+[\d\.,]*\s*(a|ma|amps?|amperes?)\b|\b\d+[\d\.,]*\s*(a|ma|amps?|amperes?)\s+(of\s+)?current\b|\b(calculate|compute|determine)\s+(the\s+)?(electrical\s+|electric\s+)?current\b|\bfind\s+(the\s+)?(electrical|electric)\s+current\b|\bfind\s+(the\s+)?current\s+(in|through|across|flowing|draw)\b|\bcurrent\s+and\s+voltage\b|\bvoltage\s+and\s+current\b|\bcurrent\s+(in|through)\s+(the|a|this)\s+(resistor|circuit|wire|conductor|capacitor|inductor|diode)\b",
    re.IGNORECASE,
)

NEGATIVE_PACKAGE_INSTALL_OR_UPGRADE = re.compile(
    r"\b(install|reinstall|uninstall|upgrade|update)\s+(the\s+)?(latest|newest)\s+(version|release|build)\b|\b(pip|npm|npx|yarn|pnpm|brew|apt|apt-get|cargo|gem|pacman)\s+(install|upgrade|update|add)\b",
    re.IGNORECASE,
)

NEGATIVE_PERSONAL_ACTIVITY = re.compile(
    r"\b(i|we|my|our)\s+(started|learned|learning|built|worked|wrote|tried|began|finished|completed|am|was|have\s+been)\b",
    re.IGNORECASE,
)

NEGATIVE_DIAGNOSTIC_EVIDENCE = re.compile(
    r"\bevidence\s+(of|for)\s+.*?\b(in|from|within)\s+(this|the)\s+(heap\s+dump|memory\s+dump|log|logs|trace|traceback|dump|diff|file|snippet|profile|code)\b|\bevidence\s+(of|for)\s+(memory\s+leaks?|bugs?|errors?|exceptions?|crashes?|failures?|deadlocks?|data\s+races?|leaks?)\b",
    re.IGNORECASE,
)


def _is_false_freshness(prompt: str, match_text: str) -> bool:
    """Determine whether a freshness match actually refers to program state, electrical current, personal activity, or package install."""
    if NEGATIVE_PACKAGE_INSTALL_OR_UPGRADE.search(prompt):
        return True
    if NEGATIVE_PERSONAL_ACTIVITY.search(prompt):
        return True
    if "current" in match_text.lower():
        lower_match = match_text.lower()
        if any(
            kw in lower_match
            for kw in (
                "supported",
                "pricing",
                "guidance",
                "standards",
                "release",
                "version",
                "prime minister",
                "president",
                "cve",
                "vulnerabilities",
                "outages",
                "market",
            )
        ):
            return False
        if NEGATIVE_CURRENT_PROGRAM_STATE.search(prompt) or NEGATIVE_CURRENT_CAMEL_CASE.search(prompt):
            return True
        if NEGATIVE_CURRENT_ELECTRICAL_OR_PHYSICS.search(prompt):
            return True
    return False


# --- Generalized Deterministic Signal Detection Patterns ---

STRONG_RESEARCH_PATTERNS: list[tuple[RoutingSignalType, re.Pattern[str]]] = [
    (
        RoutingSignalType.EXPLICIT_SOURCES_OR_CITATIONS,
        re.compile(
            r"\b(find|search\s+for|look\s+up|gather|collect|provide|retrieve|cite|give\s+me)\s+(sources?|citations?|references?|papers?|articles?|literature)\b(?!\s+code\b)|\b(sources?|citations?|references?|literature)\s+(about|on|for|regarding)\b|\b(find|search\s+for|look\s+up|gather|collect|provide|retrieve)\s+evidence\b|\bevidence\s+(for|against|about|on|regarding|of)\b|\b(primary\s+sources?|empirical\s+(studies|evidence)|peer-?reviewed(\s+literature|\s+evidence|\s+papers?)?|medical\s+literature|whitepapers?)\b|\bpublished\s+benchmarks?\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoutingSignalType.FRESHNESS_OR_CURRENT_INFO,
        re.compile(
            r"\b(is\s+|are\s+)?(currently|presently|now)\s+supported\b|\bsupported\s+(by|in)\s+(the\s+)?(latest|current|newest|recent)\b|\b(latest|current|recent|newest)\s+([a-zA-Z0-9_\-\.]+\s+){0,3}(developments?|news|trends?|updates?|findings?|releases?|advancements?|advances|breakthroughs?|reports?|data|information|research|status|state|cves?|vulnerabilities?|pricing|version|build|lts|guidance|standards?|edition|algorithms?|protocols?|approaches?|tools?|techniques?|papers?|systems?|architectures?|benchmarks?|recommendations?)\b|\b(current|latest|present)\s+status\s+of\s+(the\s+)?([a-zA-Z0-9_\-\.]+\s+){0,3}(release|version|build)\b|\b(today(\x27s|\x19s|\x27s)?|current|latest|recent|up-?to-?date)\s+([a-zA-Z0-9_\-\.]+\s+){0,3}(pricing|prices?|rates?|costs?|tiers?|plans?|fees?)\b|\b(pricing|pricing\s+tiers?|rates?|costs?)\s+as\s+of\b|\b(latest|recent|most\s+recent|new|current)\s+([a-zA-Z0-9_\-\.]+\s+){0,3}(cves?|vulnerabilities?|security\s+advisories?|exploits?|bulletins?)\b|\bwho\s+is\s+(the\s+)?current\b|\bcurrent\s+(prime\s+minister|president|vice\s+president|ceo|cto|cfo|governor|senator|chancellor|director|leader)\b|\bcurrent\s+([a-zA-Z0-9_\-\.]+\s+){0,3}(information|state|status|trends?|market|data|developments?|landscape|adoption|events?|news|progress|statistics|guidance|standards?)\b|\b(find|search\s+for|look\s+up|retrieve|gather)\s+(current|recent|up-?to-?date)\s+information\b|\b(new\s+features?|breaking\s+changes|deprecations?|announcements?)\s+.*?\s+(latest|recent|newest)\s+(release|version)\b|\b(current|latest|recent)\s+features?\s+(and|&)\s+pricing\b|\b(latest|recent|current)\s+([a-zA-Z0-9_\-\.]+\s+){0,3}(market\s+trends?|stock\s+trends?|stock\s+market|market\s+conditions?)\b|\btrending\s+.*?\s+this\s+month\b|\b(recent|latest|current)\s+([a-zA-Z0-9_\-\.]+\s+){0,3}(outages?|downtime|service\s+status|availability)\b|\bas of\s+(today|now|recently|this\s+month|this\s+year|202[4-9])\b|\b([a-zA-Z0-9_\-\.]+\s+){0,3}(trends?|developments?|news|releases?|outages?|cves?|vulnerabilities?|announcements?)\s+(this\s+week|this\s+month)\b|\b(how\s+many\s+(people|humans)|population|inhabitants)\b.*?\b(alive|living|on\s+earth|in\s+the\s+world|globally|worldwide)\b.*?\b(right\s+now|today|currently|at\s+present)\b|\bhow\s+many\s+people\s+are\s+alive\b|\bwhat(\x27s|\s+is|\s+are)\s+(the\s+)?latest\b|\bwhat(\x27s|\s+is|\s+are)\s+(the\s+)?current\s+(state|status|situation|landscape)\b|\bup-?to-?date\s+(information|data|details|status|overview|developments?|news|pricing)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoutingSignalType.CHANGING_EXTERNAL_INFO,
        re.compile(
            r"\bpricing\s+tiers?\b|\bpricing\s+plans?\b|\bcosts?\s+as\s+of\b|\bstock\s+market\s+trends?\b|\bfinancial\s+market\s+trends?\b|\bnew\s+features?\s+(and\s+breaking\s+changes\s+)?(were\s+)?added\s+in\s+the\s+latest\s+release\b|\bproduct\s+announcements?\s+at\s+their\s+most\s+recent\b|\bmanaged-cloud\s+capabilities\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoutingSignalType.OUTSIDE_AUTHORITY_OR_PROVIDER,
        re.compile(
            r"\b(un|cop\d+|ieee|acm|w3c|ietf|iso|who|cdc|nist)\b|\b(climate|developer|standards?|security|tech)\s+(conference|summit|symposium|convention|forum)\b|\b(national\s+vulnerability\s+database|nvd|cve\s+database|security\s+advisories|github\s+advisories)\b|\b(nist|iso|ietf|w3c|owasp|cis|gdpr|hipaa|fda|sec|fcc|epa)\s+(guidance|standards?|framework|regulations?|specifications?|compliance|benchmarks?)\b|\b(developer\s+day|developer\s+conference|keynote\s+announcements?|product\s+announcements?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoutingSignalType.EXPLICIT_EXTERNAL_RETRIEVAL,
        re.compile(
            r"\b(search\s+for\s+the\s+latest|find\s+recent\s+real-?world|search\s+the\s+web|live\s+web\s+search)\b",
            re.IGNORECASE,
        ),
    ),
]

# Structural markers for supplied local artifacts (requires actual structure/code, not mere mention)
SUPPLIED_ARTIFACT_STRUCTURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"```[\s\S]+?```"),  # Fenced code block
    re.compile(
        r"(Traceback\s+\(most\s+recent\s+call\s+last\):|Exception\s+in\s+thread\s+[\"'].*?[\"']|[\w\.]+Exception:\s+.*|at\s+[\w\.\$]+\([\w\.]+\.java:\d+\)|at\s+[\w\.\$]+\s*\([^)]+:\d+:\d+\))"
    ),  # Stack trace
    re.compile(
        r"(@@\s+-\d+.*?\+\d+.*?@@|---\s+[a-zA-Z0-9_\-\.\/]+\s+\+\+\+\s+[a-zA-Z0-9_\-\.\/]+)"
    ),  # Unified diff
    re.compile(
        r"(\[[a-zA-Z0-9_\-\.\:\s]{2,40}\][\s\S]*?[a-zA-Z0-9_\-]+\s*=\s*[\S]+|[a-zA-Z0-9_\.\-]+\s*[:=]\s*[\S]+\s*\n\s*[a-zA-Z0-9_\.\-]+\s*[:=]\s*[\S]+)"
    ),  # Configuration content (INI/TOML/key-value)
    re.compile(
        r"(\{\s*\"[a-zA-Z0-9_\-]+\"\s*:\s*[\s\S]+?\}|\[\s*\{\s*\"[a-zA-Z0-9_\-]+\"|\[\s*[A-Z][a-zA-Z0-9_\s]*:\s*\d+[\s\S]*?\])"
    ),  # Structured data (JSON/mapping list)
    re.compile(
        r"(\bCREATE\s+TABLE\s+\w+|\bSELECT\b[\s\S]*?\bFROM\b[\s\S]*?(\bWHERE\b|;|\Z)|\bINSERT\s+INTO\b[\s\S]*?\bVALUES\b)"
    ),  # SQL DDL/query
    re.compile(
        r"(#instances|\bClass\s+Name\b[\s\S]*?\b(Shallow|Retained)\s+Heap\b|\binstances\s+#bytes\s+class\s+name\b|\b[0-9]+:\s+[0-9]+\s+[0-9]+\s+[\w\.\$\[]+)"
    ),  # Memory/heap dump histogram
    re.compile(
        r"(\bdef\s+\w+\s*\([^)]*\)\s*:|\bclass\s+\w+[\s:(]|\bfunction\s+\w+\s*\(|\binterface\s+\w+[\s<\{]|\btype\s+\w+\s*=|^\s*<\?php|\^[\(\[\w\+\-\*\?\\\.\]\)]+\$|\{\s*\*\*[a-zA-Z0-9_]+|[a-zA-Z0-9_]+\s*\|\s*[a-zA-Z0-9_]+)"
    ),  # Code construct or inline expression
]

STRONG_GENERAL_PATTERNS: list[tuple[RoutingSignalType, re.Pattern[str]]] = [
    (
        RoutingSignalType.LOCAL_CODE_OR_DEBUGGING,
        re.compile(
            r"\b(write|create|implement)\s+(a\s+)?(python|typescript|javascript|bash|shell|sql|go|rust|c|c\+\+|java)\s+(script|function|program|algorithm|class|method|module|simulation|tool|cli)\b|\b(binary\s+search|sorting\s+algorithms?|depth-?first|breadth-?first|dijkstra|fibonacci|linked\s+list|binary\s+tree)\b|\b(debug|fix\s+(a\s+)?bug|fix\s+this|traceback|nullpointerexception|type\s+error|generic\s+constraints?|catastrophic\s+backtracking)\b|\b(code\s+review|review\s+this\s+(pr\s+diff|code|function|implementation|script)|optimize\s+this\s+(regular\s+expression|regex|query|function|algorithm))\b|\b(explain\s+why\s+the\s+loop\s+never\s+terminates|infinite\s+loop|loop\s+termination)\b|\b(check\s+installed\s+package\s+versions|restart\s+worker\s+container|deploy\s+application\s+to\s+cluster)\b|\b(install|upgrade)\s+(the\s+)?(latest|newest)\s+version\s+of\b|\b(install|reinstall|uninstall|upgrade)\s+(the\s+)?([a-zA-Z0-9_\-\.]+\s+){0,3}(package|library|module|tool|dependency|dependencies)\b|\b(pip|npm|yarn|cargo|brew|apt|apt-get)\s+(install|upgrade|add|update)\b|\b(heap\s+dump|memory\s+leak|memory\s+dump)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoutingSignalType.TRANSFORMATION_OR_REWRITE,
        re.compile(
            r"\b(refactor|rewrite|convert)\s+(this|the)\b|\btranslate\s+['\"].*?['\"]\s+into\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoutingSignalType.ESTABLISHED_CONCEPTUAL_EXPLANATION,
        re.compile(
            r"\b(explain|describe)\s+how\s+[a-zA-Z0-9_\-\s]{3,35}\s+(works?|operates?|handles?|elects?|scales?)\b|\b(explain|describe)\s+the\s+difference\s+between\s+[a-zA-Z0-9_\-\s]{3,30}\s+and\s+[a-zA-Z0-9_\-\s]{3,30}\b|\b(explain|describe)\s+the\s+(concept|principles?|fundamentals?)\s+of\s+[a-zA-Z0-9_\-\s]{3,35}\b|\bhow\s+does\s+(the\s+)?[a-zA-Z0-9_\-\s]{3,35}\s+(algorithm|protocol|flow|mechanism)\s+(work|elect|handle|operate|scale)\b|\bexplain\s+(first|second|third|1st|2nd|3rd)\s+normal\s+form\b|\bhow\s+do\s+you\s+(reverse|sort|filter|map|traverse)\s+a\s+(string|array|list|dictionary)\b|\b(what\s+is|calculate|compute)\s+\d+[\d\.,\s]*\s*(%|\*|\/|\+|\-|\s*percent\s+of)\s*\d+[\d\.,\s]*\b|\b(boiling|melting|freezing)\s+point\s+of\b|\b(speed\s+of\s+light|speed\s+of\s+sound|gravitational\s+constant|atomic\s+(number|mass|weight))\b|\b(calculate|compute|determine)\s+(the\s+)?(electrical\s+|electric\s+)?current\b|\bfind\s+(the\s+)?(electrical|electric)\s+current\b|\bvoltage\s+drop\s+given\s+a\s+current\b|\b(calculate|compute)\s+voltage\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoutingSignalType.ALL_DATA_SUPPLIED_INLINE,
        re.compile(
            r"\bhere\s+(is|are)\s+(a\s+)?(the\s+)?(survey|numbers|figures|dataset|stats|statistics|results|list|scores?|data|table)\b",
            re.IGNORECASE,
        ),
    ),
]

WEAK_RESEARCH_VERB_PATTERN = re.compile(
    r"\b(analyse|analyses|analysis|analyze|analyzed|analyzes|analyzing|assess|assessed|assesses|assessing|assessment|benchmark|benchmarked|benchmarks|benchmarking|comparative|compare|compared|compares|comparing|comparison|conduct|conducted|conducting|conducts|evaluate|evaluated|evaluates|evaluating|evaluation|explore|explored|explores|exploring|investigate|investigated|investigates|investigating|investigation|research|researched|researches|researching|study|studied|studies|studying|survey|surveyed|surveys|surveying)\b",
    re.IGNORECASE,
)

COMPOUND_TASK_PATTERN = re.compile(
    r"\b(search\s+for|find)\s+.*?\s+and\s+(write|build|create|generate)\b",
    re.IGNORECASE,
)

AMBIGUOUS_BORDERLINE_PATTERN = re.compile(
    r"\b(advancements?|advancing)\b|\bhow\s+is\s+[a-zA-Z0-9_\-\s]{2,30}\s+impacting\s+[a-zA-Z0-9_\-\s]{2,30}\b|\b(should\s+(i|we)\s+use|choosing\s+between|choose\s+between)\s+[a-zA-Z0-9_\-]+\s+or\s+[a-zA-Z0-9_\-]+\b|\bbest\s+practices\s+for\s+[a-zA-Z0-9_\-\s]{3,35}\b|\b(pros\s+and\s+cons|trade-?offs)\s+of\s+[a-zA-Z0-9_\-\s]{3,35}\b",
    re.IGNORECASE,
)


def _format_signal_evidence(signals: list[RoutingSignal]) -> str:
    """Format signal types and evidence concisely for dynamic rationale generation."""
    return ", ".join(f"{s.signal_type.value} ({s.evidence!r})" for s in signals)


class DeterministicCapabilityRouter:
    """Routes prompts to agent capabilities with transparent assessment and ambiguity analysis."""

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

    def assess(self, prompt: str) -> RoutingAssessment:
        """Produce an inspectable RoutingAssessment containing capability, clarity, signals, and rationale."""
        # 0. Custom keyword override mode (preserves custom keyword testing)
        if self._custom_pattern is not None:
            m = self._custom_pattern.search(prompt)
            if m:
                sig = RoutingSignal(
                    signal_type=RoutingSignalType.EXPLICIT_EXTERNAL_RETRIEVAL,
                    target_capability=AgentCapability.RESEARCH,
                    strength=SignalStrength.STRONG,
                    evidence=m.group(0),
                )
                return RoutingAssessment(
                    provisional_capability=AgentCapability.RESEARCH,
                    clarity=RoutingClarity.CLEAR,
                    signals=(sig,),
                    reason=f"Custom keyword match: {m.group(0)!r}",
                )
            return RoutingAssessment(
                provisional_capability=AgentCapability.GENERAL,
                clarity=RoutingClarity.CLEAR,
                signals=(),
                reason="No custom keyword match",
            )

        # 1. Collect all deterministic signals
        signals: list[RoutingSignal] = []

        # Check supplied artifact structural indicators (requires actual structure/content)
        for pattern in SUPPLIED_ARTIFACT_STRUCTURE_PATTERNS:
            match = pattern.search(prompt)
            if match:
                signals.append(
                    RoutingSignal(
                        signal_type=RoutingSignalType.SUPPLIED_LOCAL_ARTIFACT,
                        target_capability=AgentCapability.GENERAL,
                        strength=SignalStrength.STRONG,
                        evidence=match.group(0).strip()[:80],
                    )
                )
                break  # Record one supplied artifact signal

        # Strong Research signals (with negative guards for false freshness and diagnostic evidence)
        for sig_type, pattern in STRONG_RESEARCH_PATTERNS:
            match = pattern.search(prompt)
            if match:
                evidence = match.group(0)
                if sig_type == RoutingSignalType.FRESHNESS_OR_CURRENT_INFO and _is_false_freshness(
                    prompt, evidence
                ):
                    continue  # Suppress false freshness on local state, install commands, personal activity, etc.
                if (
                    sig_type == RoutingSignalType.EXPLICIT_SOURCES_OR_CITATIONS
                    and NEGATIVE_DIAGNOSTIC_EVIDENCE.search(prompt)
                ):
                    continue  # Suppress diagnostic evidence search on local artifacts/debugging
                signals.append(
                    RoutingSignal(
                        signal_type=sig_type,
                        target_capability=AgentCapability.RESEARCH,
                        strength=SignalStrength.STRONG,
                        evidence=evidence,
                    )
                )

        # Strong General signals
        for sig_type, pattern in STRONG_GENERAL_PATTERNS:
            match = pattern.search(prompt)
            if match:
                signals.append(
                    RoutingSignal(
                        signal_type=sig_type,
                        target_capability=AgentCapability.GENERAL,
                        strength=SignalStrength.STRONG,
                        evidence=match.group(0),
                    )
                )

        # Compound mixed task signal
        compound_match = COMPOUND_TASK_PATTERN.search(prompt)
        if compound_match:
            signals.append(
                RoutingSignal(
                    signal_type=RoutingSignalType.COMPOUND_MIXED_TASK,
                    target_capability=AgentCapability.RESEARCH,
                    strength=SignalStrength.STRONG,
                    evidence=compound_match.group(0),
                )
            )

        # Weak research verbs (always recorded as contextual signals, never independently dictate RESEARCH)
        weak_match = WEAK_RESEARCH_VERB_PATTERN.search(prompt)
        if weak_match:
            signals.append(
                RoutingSignal(
                    signal_type=RoutingSignalType.WEAK_RESEARCH_VERB,
                    target_capability=AgentCapability.RESEARCH,
                    strength=SignalStrength.WEAK,
                    evidence=weak_match.group(0),
                )
            )

        # Partition collected signals
        strong_research = [
            s
            for s in signals
            if s.target_capability == AgentCapability.RESEARCH
            and s.strength == SignalStrength.STRONG
        ]
        strong_general = [
            s
            for s in signals
            if s.target_capability == AgentCapability.GENERAL
            and s.strength == SignalStrength.STRONG
        ]
        has_compound = any(
            s.signal_type == RoutingSignalType.COMPOUND_MIXED_TASK for s in signals
        )
        signals_tuple = tuple(signals)

        # --- Rule Evaluation Order ---

        # Case 1: Compound Task Policy
        # If task combines external retrieval with local code generation, external retrieval
        # is a prerequisite step.
        if has_compound:
            comp_evidence = next(
                s.evidence for s in signals if s.signal_type == RoutingSignalType.COMPOUND_MIXED_TASK
            )
            return RoutingAssessment(
                provisional_capability=AgentCapability.RESEARCH,
                clarity=RoutingClarity.CLEAR,
                signals=signals_tuple,
                reason=f"Compound task policy: external retrieval ({comp_evidence!r}) is an essential prerequisite before local synthesis.",
            )

        # Case 2: Conflicting Strong Signals
        # Meaningful deterministic signals point toward different capabilities
        # (e.g. established conceptual trade-offs combined with changing cloud vendor capabilities or external guidance).
        if strong_research and strong_general:
            research_summary = _format_signal_evidence(strong_research)
            general_summary = _format_signal_evidence(strong_general)
            return RoutingAssessment(
                provisional_capability=AgentCapability.RESEARCH,
                clarity=RoutingClarity.AMBIGUOUS,
                signals=signals_tuple,
                reason=f"Conflicting signals detected: strong general ({general_summary}) conflicts with strong research ({research_summary}).",
            )

        # Case 3: Strong General Signals Present (No strong research signals)
        # Weak research verbs must never determine RESEARCH when supplied artifacts
        # or local tasks are present.
        if strong_general and not strong_research:
            general_summary = _format_signal_evidence(strong_general)
            return RoutingAssessment(
                provisional_capability=AgentCapability.GENERAL,
                clarity=RoutingClarity.CLEAR,
                signals=signals_tuple,
                reason=f"Strong general signals detected ({general_summary}); weak research verbs do not determine routing.",
            )

        # Case 4: Ambiguous / Borderline Prompts (Insufficient Evidence / Dual Interpretation)
        ambiguous_match = AMBIGUOUS_BORDERLINE_PATTERN.search(prompt)
        if ambiguous_match and not strong_research and not strong_general:
            matched_str = ambiguous_match.group(0).lower()
            if "advancement" in matched_str or "impacting" in matched_str:
                return RoutingAssessment(
                    provisional_capability=AgentCapability.RESEARCH,
                    clarity=RoutingClarity.AMBIGUOUS,
                    signals=signals_tuple,
                    reason=f"Ambiguous borderline intent ({matched_str}): plausible dual interpretation of historical theory vs. active research advancements.",
                )
            return RoutingAssessment(
                provisional_capability=AgentCapability.GENERAL,
                clarity=RoutingClarity.AMBIGUOUS,
                signals=signals_tuple,
                reason=f"Ambiguous architectural or technology-choice inquiry ({matched_str}) answerable from general knowledge or external research.",
            )

        # Case 5: Clear Strong Research Signals (No general counter-signals)
        if strong_research:
            research_summary = _format_signal_evidence(strong_research)
            return RoutingAssessment(
                provisional_capability=AgentCapability.RESEARCH,
                clarity=RoutingClarity.CLEAR,
                signals=signals_tuple,
                reason=f"Strong research signals detected: {research_summary}.",
            )

        # Case 6: Weak Research Verbs Alone (No strong signals)
        # Enforce weak-verb policy: weak verbs must never independently determine RESEARCH.
        # Provisional capability defaults to GENERAL.
        if weak_match:
            verb = weak_match.group(0).lower()
            return RoutingAssessment(
                provisional_capability=AgentCapability.GENERAL,
                clarity=RoutingClarity.AMBIGUOUS,
                signals=signals_tuple,
                reason=f"Weak research verb ({verb!r}) without strong external research evidence defaults provisionally to general capability (ambiguous).",
            )

        # Case 7: Default General
        return RoutingAssessment(
            provisional_capability=AgentCapability.GENERAL,
            clarity=RoutingClarity.CLEAR,
            signals=signals_tuple,
            reason="No research signals detected; defaulting to general capability.",
        )

    def route(self, prompt: str) -> AgentCapability:
        """Route prompt to AgentCapability.RESEARCH or AgentCapability.GENERAL.

        Delegates to self.assess(prompt).provisional_capability.
        """
        return self.assess(prompt).provisional_capability
