"""Evidence quality evaluator implementing deterministic heuristics for authority, relevance, and filtering."""

import ipaddress
import re
from collections import Counter
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

from research_agent.domain.evidence import (
    AuthorityTier,
    CandidateEvidence,
    EvaluatedEvidence,
    FilterReason,
    SourceType,
)

# Standard English stop words to exclude when extracting significant keywords
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
    }
)

# Strong positive authority indicators (TLDs and domain substrings)
ACADEMIC_GOV_TLDS = (".edu", ".gov", ".mil", ".ac.uk", ".gov.uk")

STANDARDS_DOMAINS = frozenset(
    {
        "ietf.org",
        "rfc-editor.org",
        "w3.org",
        "iso.org",
        "nist.gov",
        "ieee.org",
        "acm.org",
        "oasis-open.org",
    }
)

ACADEMIC_RESEARCH_DOMAINS = frozenset(
    {
        "arxiv.org",
        "biorxiv.org",
        "medrxiv.org",
        "nature.com",
        "science.org",
        "ncbi.nlm.nih.gov",
        "semanticscholar.org",
    }
)

OFFICIAL_DOCS_DOMAINS = frozenset(
    {
        "docs.python.org",
        "python.org",
        "doc.rust-lang.org",
        "rust-lang.org",
        "go.dev",
        "golang.org",
        "kubernetes.io",
        "docker.com",
        "docs.docker.com",
        "redis.io",
        "postgresql.org",
        "sqlite.org",
        "apache.org",
        "kernel.org",
        "developer.mozilla.org",
    }
)

TECHNICAL_REFERENCE_DOMAINS = frozenset(
    {
        "github.com",
        "gitlab.com",
        "stackoverflow.com",
        "wikipedia.org",
        "archive.org",
    }
)

# Explicit low-value, promotional, spam, and clickbait patterns
PROMOTIONAL_QUERY_KEYS = frozenset(
    {
        "aff",
        "affiliate",
        "aff_id",
        "referrer",
        "sponsor",
        "sponsored",
        "campaign_id",
        "partner",
    }
)

PROMOTIONAL_TEXT_PATTERNS = re.compile(
    r"\b("
    r"buy\s+now|order\s+now|add\s+to\s+cart|"
    r"discount\s+code|coupon\s+code|promo\s+code|"
    r"best\s+price\s+guarantee|lowest\s+price|"
    r"affiliate\s+disclosure|sponsored\s+review|paid\s+partnership|"
    r"100%\s+free\s+download\s+crack|torrent\s+download|"
    r"subscribe\s+today\s+for\s+only|exclusive\s+discount"
    r")\b",
    re.IGNORECASE,
)

KNOWN_LOW_VALUE_DOMAINS = frozenset(
    {
        "pinterest.com",
        "coupons.com",
        "retailmenot.com",
        "bestdeals.com",
    }
)

# Common compound public suffixes for conservative root-domain diversity capping.
# Note: This is a focused, deterministic heuristic, not a full Public Suffix List (PSL).
COMPOUND_PUBLIC_SUFFIXES = frozenset(
    {
        "co.uk",
        "ac.uk",
        "gov.uk",
        "com.au",
        "co.jp",
    }
)


def extract_keywords(text: str) -> set[str]:
    """Extract lowercase alpha-numeric keywords from text, omitting common stop words."""
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS and not w.isdigit()}


def get_root_domain(domain: str) -> str:
    """Extract root domain for diversity capping (e.g., 'docs.python.org' -> 'python.org').

    Handles common compound public suffixes (e.g. 'bbc.co.uk' -> 'bbc.co.uk',
    'research.ox.ac.uk' -> 'ox.ac.uk') so unrelated domains are not falsely collapsed.
    """
    parts = domain.lower().split(".")
    if len(parts) >= 3:
        two_suffix = ".".join(parts[-2:])
        if two_suffix in COMPOUND_PUBLIC_SUFFIXES:
            return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain.lower()


class EvidenceEvaluator:
    """Evaluates candidate research evidence against deterministic quality, authority, and relevance heuristics.

    Preserves Clean Architecture independence: depends solely on the standard library and domain evidence models.
    """

    def __init__(
        self,
        max_per_root_domain: int = 2,
        min_snippet_chars: int = 25,
    ) -> None:
        self.max_per_root_domain = max_per_root_domain
        self.min_snippet_chars = min_snippet_chars

    def evaluate(
        self,
        candidates: list[CandidateEvidence],
        query: str,
    ) -> list[EvaluatedEvidence]:
        """Evaluate, filter, and deterministically rank a collection of candidate evidence."""
        if not candidates:
            return []

        query_keywords = extract_keywords(query)
        seen_urls: set[str] = set()
        seen_contents: set[str] = set()

        # Phase 1: Individual evaluation (validation, authority, promotional/spam check, relevance)
        raw_evaluated: list[EvaluatedEvidence] = []
        url_to_raw_score: dict[str, float | None] = {}
        for candidate in candidates:
            item = self._evaluate_single(candidate, query_keywords, seen_urls, seen_contents)
            raw_evaluated.append(item)
            if item.url not in url_to_raw_score:
                url_to_raw_score[item.url] = candidate.raw_score

        # Phase 2: Cross-domain query keyword overlap analysis across retained candidates
        # Count how many distinct domains mention each query keyword in retained evidence
        domain_keyword_counts: Counter[str] = Counter()
        domain_by_keyword: dict[str, set[str]] = {kw: set() for kw in query_keywords}
        for item in raw_evaluated:
            if item.retained:
                item_keywords = extract_keywords(f"{item.title} {item.content}")
                for kw in query_keywords:
                    if kw in item_keywords:
                        domain_by_keyword[kw].add(item.domain)

        for domains in domain_by_keyword.values():
            if len(domains) >= 2:
                for d in domains:
                    domain_keyword_counts[d] += 1

        # Phase 3: Domain diversity capping and cross-domain overlap annotation
        root_domain_counts: Counter[str] = Counter()
        final_evaluated: list[EvaluatedEvidence] = []

        for item in raw_evaluated:
            if not item.retained:
                final_evaluated.append(item)
                continue

            root_domain = get_root_domain(item.domain)
            root_domain_counts[root_domain] += 1
            if root_domain_counts[root_domain] > self.max_per_root_domain:
                # Filter out domain flooding to preserve diversity of perspectives
                rejection = EvaluatedEvidence(
                    url=item.url,
                    title=item.title,
                    content=item.content,
                    domain=item.domain,
                    authority_tier=item.authority_tier,
                    source_type=item.source_type,
                    quality_signals=(*item.quality_signals, "domain-cap-exceeded"),
                    relevance_signal=item.relevance_signal,
                    recency_indicator=item.recency_indicator,
                    rank_score=round(item.rank_score * 0.5, 3),
                    retained=False,
                    filter_reason=FilterReason.DOMAIN_CAP_EXCEEDED,
                    rationale=f"Filtered: Exceeded maximum {self.max_per_root_domain} results from root domain '{root_domain}'.",
                    cross_domain_overlap_count=0,
                )
                final_evaluated.append(rejection)
                continue

            # Check for cross-domain query keyword overlap
            shared_terms = domain_keyword_counts.get(item.domain, 0)
            has_keyword_overlap = shared_terms > 0
            signals = list(item.quality_signals)
            if has_keyword_overlap:
                signals.append(f"cross-domain-keyword-overlap({shared_terms}-terms)")
            else:
                signals.append("no-cross-domain-overlap")

            # Look up each item's own raw provider relevance score
            item_raw_score = url_to_raw_score.get(item.url)

            # Calculate explainable rank score
            rank_score = self._compute_rank_score(
                authority_tier=item.authority_tier,
                relevance_signal=item.relevance_signal,
                has_keyword_overlap=has_keyword_overlap,
                recency_indicator=item.recency_indicator,
                raw_score=item_raw_score,
            )

            rationale = self._format_rationale(
                item=item,
                rank_score=rank_score,
                has_keyword_overlap=has_keyword_overlap,
                shared_terms=shared_terms,
            )

            updated = EvaluatedEvidence(
                url=item.url,
                title=item.title,
                content=item.content,
                domain=item.domain,
                authority_tier=item.authority_tier,
                source_type=item.source_type,
                quality_signals=tuple(signals),
                relevance_signal=item.relevance_signal,
                recency_indicator=item.recency_indicator,
                rank_score=rank_score,
                retained=True,
                filter_reason=None,
                rationale=rationale,
                cross_domain_overlap_count=shared_terms,
            )
            final_evaluated.append(updated)

        # Deterministic sorting: retained first, then highest rank_score, then domain, then url
        final_evaluated.sort(
            key=lambda x: (
                0 if x.retained else 1,
                -x.rank_score,
                x.domain,
                x.url,
            )
        )
        return final_evaluated

    def _evaluate_single(
        self,
        candidate: CandidateEvidence,
        query_keywords: set[str],
        seen_urls: set[str],
        seen_contents: set[str],
    ) -> EvaluatedEvidence:
        """Validate and classify an individual candidate evidence item."""
        url = candidate.url.strip()
        title = candidate.title.strip()
        content = candidate.content.strip()

        # 1. URL validation & safe parsing
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return EvaluatedEvidence(
                url=url,
                title=title,
                content=content,
                domain="invalid",
                authority_tier=AuthorityTier.LOW_VALUE,
                source_type=SourceType.MALFORMED,
                quality_signals=("malformed-url",),
                relevance_signal="low",
                recency_indicator="unknown",
                rank_score=0.0,
                retained=False,
                filter_reason=FilterReason.MALFORMED_URL,
                rationale="Filtered: Invalid or unsupported URL scheme.",
            )

        # Reject embedded credentials in URL (e.g., user:pass@host)
        if parsed.username or parsed.password:
            return EvaluatedEvidence(
                url="[REDACTED_UNSAFE_URL]",
                title=title,
                content=content,
                domain="unsafe",
                authority_tier=AuthorityTier.LOW_VALUE,
                source_type=SourceType.MALFORMED,
                quality_signals=("unsafe-credentials-in-url",),
                relevance_signal="low",
                recency_indicator="unknown",
                rank_score=0.0,
                retained=False,
                filter_reason=FilterReason.MALFORMED_URL,
                rationale="Filtered: URL contained userinfo/credentials.",
            )

        # Extract and normalize domain
        raw_host = parsed.netloc.split(":")[0].lower().removeprefix("www.")

        # Reject localhost and private IP addresses
        try:
            ip = ipaddress.ip_address(raw_host)
            if ip.is_private or ip.is_loopback:
                return EvaluatedEvidence(
                    url=url,
                    title=title,
                    content=content,
                    domain=raw_host,
                    authority_tier=AuthorityTier.LOW_VALUE,
                    source_type=SourceType.MALFORMED,
                    quality_signals=("private-network-address",),
                    relevance_signal="low",
                    recency_indicator="unknown",
                    rank_score=0.0,
                    retained=False,
                    filter_reason=FilterReason.MALFORMED_URL,
                    rationale="Filtered: URL targets private or loopback IP address.",
                )
        except ValueError:
            pass  # Normal hostname, not an IP address

        # 2. Duplicate detection
        normalized_url = f"{parsed.scheme}://{raw_host}{parsed.path}".rstrip("/")
        if normalized_url in seen_urls:
            return EvaluatedEvidence(
                url=url,
                title=title,
                content=content,
                domain=raw_host,
                authority_tier=AuthorityTier.LOW_VALUE,
                source_type=SourceType.UNKNOWN,
                quality_signals=("duplicate-url",),
                relevance_signal="low",
                recency_indicator="unknown",
                rank_score=0.0,
                retained=False,
                filter_reason=FilterReason.DUPLICATE,
                rationale="Filtered: Duplicate URL in retrieval candidates.",
            )
        seen_urls.add(normalized_url)

        normalized_content_sample = re.sub(r"\s+", " ", content[:120].lower()).strip()
        if normalized_content_sample and normalized_content_sample in seen_contents:
            return EvaluatedEvidence(
                url=url,
                title=title,
                content=content,
                domain=raw_host,
                authority_tier=AuthorityTier.LOW_VALUE,
                source_type=SourceType.UNKNOWN,
                quality_signals=("duplicate-content",),
                relevance_signal="low",
                recency_indicator="unknown",
                rank_score=0.0,
                retained=False,
                filter_reason=FilterReason.DUPLICATE,
                rationale="Filtered: Duplicate snippet content already evaluated from another source.",
            )
        if normalized_content_sample:
            seen_contents.add(normalized_content_sample)

        # 3. Low information density
        if len(content) < self.min_snippet_chars:
            return EvaluatedEvidence(
                url=url,
                title=title,
                content=content,
                domain=raw_host,
                authority_tier=AuthorityTier.LOW_VALUE,
                source_type=SourceType.UNKNOWN,
                quality_signals=("low-information-density",),
                relevance_signal="low",
                recency_indicator="unknown",
                rank_score=0.0,
                retained=False,
                filter_reason=FilterReason.LOW_QUALITY,
                rationale=f"Filtered: Content snippet shorter than {self.min_snippet_chars} characters.",
            )

        # 4. Check for promotional / affiliate / spam patterns
        query_params = parse_qs(parsed.query)
        has_promo_query = any(k.lower() in PROMOTIONAL_QUERY_KEYS for k in query_params)
        has_promo_text = bool(
            PROMOTIONAL_TEXT_PATTERNS.search(title)
            or PROMOTIONAL_TEXT_PATTERNS.search(content)
        )
        is_known_low_value = raw_host in KNOWN_LOW_VALUE_DOMAINS

        if has_promo_query or has_promo_text or is_known_low_value:
            signals = ["promotional-content"]
            if has_promo_query:
                signals.append("affiliate-query-params")
            if has_promo_text:
                signals.append("commercial-call-to-action")
            if is_known_low_value:
                signals.append("known-low-value-domain")

            return EvaluatedEvidence(
                url=url,
                title=title,
                content=content,
                domain=raw_host,
                authority_tier=AuthorityTier.LOW_VALUE,
                source_type=SourceType.PROMOTIONAL,
                quality_signals=tuple(signals),
                relevance_signal="low",
                recency_indicator="unknown",
                rank_score=0.0,
                retained=False,
                filter_reason=FilterReason.PROMOTIONAL,
                rationale="Filtered: Detected promotional marketing, commercial affiliate, or low-information content.",
            )

        # 5. Authority classification
        authority_tier, source_type, auth_signals = self._classify_authority(raw_host, parsed.path)

        # 6. Relevance classification
        relevance_signal, rel_signals, keyword_match_count = self._evaluate_relevance(
            candidate, query_keywords
        )

        # Only filter for IRRELEVANT if an explicit provider score is present and indicates negligible relevance (< 0.15)
        if candidate.raw_score is not None and candidate.raw_score < 0.15 and keyword_match_count == 0:
            return EvaluatedEvidence(
                url=url,
                title=title,
                content=content,
                domain=raw_host,
                authority_tier=authority_tier,
                source_type=source_type,
                quality_signals=(*auth_signals, "negligible-provider-relevance"),
                relevance_signal="low",
                recency_indicator="unknown",
                rank_score=0.0,
                retained=False,
                filter_reason=FilterReason.IRRELEVANT,
                rationale="Filtered: Explicit search provider score indicates negligible relevance and zero lexical overlap.",
            )

        # 7. Recency indicator
        recency_indicator = self._evaluate_recency(candidate.published_date)

        # Preliminary score before cross-domain overlap
        rank_score = self._compute_rank_score(
            authority_tier=authority_tier,
            relevance_signal=relevance_signal,
            has_keyword_overlap=False,
            recency_indicator=recency_indicator,
            raw_score=candidate.raw_score,
        )

        all_signals = (*auth_signals, *rel_signals, f"recency:{recency_indicator}")

        return EvaluatedEvidence(
            url=url,
            title=title,
            content=content,
            domain=raw_host,
            authority_tier=authority_tier,
            source_type=source_type,
            quality_signals=all_signals,
            relevance_signal=relevance_signal,
            recency_indicator=recency_indicator,
            rank_score=rank_score,
            retained=True,
            filter_reason=None,
            rationale="",  # Enriched in Phase 3
        )

    def _classify_authority(
        self, host: str, path: str
    ) -> tuple[AuthorityTier, SourceType, list[str]]:
        """Classify domain authority into High, Medium, or Neutral tiers with explainable signals."""
        signals: list[str] = []

        # Government & Academic TLDs
        if any(host.endswith(tld) for tld in ACADEMIC_GOV_TLDS):
            signals.append("institutional-domain")
            return AuthorityTier.HIGH, SourceType.ACADEMIC_GOV, signals

        # Standards bodies
        if host in STANDARDS_DOMAINS or any(host.endswith("." + d) for d in STANDARDS_DOMAINS):
            signals.append("standards-body")
            return AuthorityTier.HIGH, SourceType.STANDARDS, signals

        # Academic repositories
        if host in ACADEMIC_RESEARCH_DOMAINS or any(host.endswith("." + d) for d in ACADEMIC_RESEARCH_DOMAINS):
            signals.append("academic-research-repository")
            return AuthorityTier.HIGH, SourceType.ACADEMIC_GOV, signals

        # Explicit official documentation domains (unknown docs.* / developer.* subdomains do NOT receive high authority)
        if (
            host in OFFICIAL_DOCS_DOMAINS
            or any(host.endswith("." + d) for d in OFFICIAL_DOCS_DOMAINS)
            or (host.endswith(".github.io") and ("doc" in path or "spec" in path))
        ):
            signals.append("official-documentation")
            return AuthorityTier.HIGH, SourceType.OFFICIAL_DOCS, signals

        # Established technical reference
        if host in TECHNICAL_REFERENCE_DOMAINS or any(host.endswith("." + d) for d in TECHNICAL_REFERENCE_DOMAINS):
            signals.append("technical-reference-platform")
            return AuthorityTier.MEDIUM, SourceType.TECHNICAL_REFERENCE, signals

        # Standard commercial or editorial web (neutral/unknown, not filtered)
        signals.append("general-web-domain")
        return AuthorityTier.NEUTRAL, SourceType.GENERAL_EDITORIAL, signals

    def _evaluate_relevance(
        self, candidate: CandidateEvidence, query_keywords: set[str]
    ) -> tuple[str, list[str], int]:
        """Assess query relevance using keyword overlap and raw provider score."""
        signals: list[str] = []
        if not query_keywords:
            return "medium", ["unconstrained-query"], 0

        text_to_search = f"{candidate.title} {candidate.url} {candidate.content}".lower()
        matched = [kw for kw in query_keywords if kw in text_to_search]
        match_count = len(matched)
        ratio = match_count / len(query_keywords)

        signals.append(f"matched-{match_count}/{len(query_keywords)}-keywords")

        if ratio >= 0.5 or match_count >= 3:
            signals.append("high-query-overlap")
            return "high", signals, match_count
        if ratio >= 0.25 or match_count >= 1:
            signals.append("moderate-query-overlap")
            return "medium", signals, match_count

        signals.append("low-query-overlap")
        return "low", signals, match_count

    def _evaluate_recency(self, published_date: str | None) -> str:
        """Determine recency indicator from explicit publication date metadata without content scanning."""
        if published_date and published_date.strip():
            raw = published_date.strip()
            # Attempt to extract year from explicit publication metadata
            year_match = re.search(r"\b(19\d\d|20\d\d)\b", raw)
            if year_match:
                year = int(year_match.group(1))
                current_year = datetime.now(UTC).year
                if current_year - year <= 2:
                    return "recent"
                return "historical"

        return "unknown"

    def _compute_rank_score(
        self,
        authority_tier: AuthorityTier,
        relevance_signal: str,
        has_keyword_overlap: bool,
        recency_indicator: str,
        raw_score: float | None = None,
    ) -> float:
        """Compute an explainable, deterministic ranking score from decomposable factors."""
        # Factor 1: Authority weight (0.0 to 0.40)
        auth_weights = {
            AuthorityTier.HIGH: 0.40,
            AuthorityTier.MEDIUM: 0.28,
            AuthorityTier.NEUTRAL: 0.18,
            AuthorityTier.LOW_VALUE: 0.0,
        }
        auth_score = auth_weights.get(authority_tier, 0.18)

        # Factor 2: Relevance weight (0.0 to 0.40)
        rel_weights = {
            "high": 0.40,
            "medium": 0.25,
            "low": 0.10,
        }
        rel_score = rel_weights.get(relevance_signal, 0.25)
        # Factor in raw provider score moderately if provided
        if raw_score is not None and isinstance(raw_score, (int, float)):
            clamped_raw = max(0.0, min(float(raw_score), 1.0))
            rel_score = (rel_score * 0.7) + (clamped_raw * 0.40 * 0.3)

        # Factor 3: Cross-domain query keyword overlap weight (0.05 to 0.15)
        # Represents retrieval presence across distinct domains without asserting consensus
        overlap_score = 0.15 if has_keyword_overlap else 0.05

        # Factor 4: Recency weight (0.0 to 0.05)
        recency_weights = {
            "recent": 0.05,
            "unknown": 0.02,
            "historical": 0.01,
        }
        recency_score = recency_weights.get(recency_indicator, 0.02)

        composite = auth_score + rel_score + overlap_score + recency_score
        return round(min(max(composite, 0.0), 1.0), 3)

    def _format_rationale(
        self,
        item: EvaluatedEvidence,
        rank_score: float,
        has_keyword_overlap: bool,
        shared_terms: int,
    ) -> str:
        """Format an explainable rationale sentence explaining the evaluation outcome."""
        tier_desc = {
            AuthorityTier.HIGH: "High-authority verified institutional or documentation source",
            AuthorityTier.MEDIUM: "Established technical reference platform",
            AuthorityTier.NEUTRAL: "Standard web domain without promotional signals",
            AuthorityTier.LOW_VALUE: "Low-value or commercial promotional domain",
        }.get(item.authority_tier, "Web source")

        parts = [f"Retained (Rank Score: {rank_score:.2f})", f"{tier_desc} ({item.domain})"]

        if item.relevance_signal == "high":
            parts.append("strong query keyword overlap")
        elif item.relevance_signal == "medium":
            parts.append("moderate query relevance")

        if has_keyword_overlap:
            parts.append(f"cross-domain keyword overlap ({shared_terms} shared terms across domains)")
        else:
            parts.append("no cross-domain keyword overlap observed")

        return "; ".join(parts) + "."

        return "; ".join(parts) + "."
