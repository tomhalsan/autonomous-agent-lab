"""Domain models for research evidence quality, authority evaluation, and filtering."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AuthorityTier(str, Enum):
    """Categorical source authority tier.

    Represents provenance and institutional standing, not a proof of universal truth.
    """

    HIGH = "high"  # Official docs, standards bodies, academic, governmental
    MEDIUM = "medium"  # Established technical publications, verified references
    NEUTRAL = "neutral"  # Standard public web, blogs, general news (non-promotional)
    LOW_VALUE = "low_value"  # Promotional, spam, content farms, ad-heavy, clickbait


class SourceType(str, Enum):
    """Specific structural classification of evidence sources."""

    ACADEMIC_GOV = "academic_gov"
    OFFICIAL_DOCS = "official_docs"
    STANDARDS = "standards"
    TECHNICAL_REFERENCE = "technical_reference"
    GENERAL_EDITORIAL = "general_editorial"
    COMMERCIAL_EDITORIAL = "commercial_editorial"
    PROMOTIONAL = "promotional"
    CONTENT_FARM = "content_farm"
    MALFORMED = "malformed"
    UNKNOWN = "unknown"


class FilterReason(str, Enum):
    """Explicit rationale for why a candidate evidence item was filtered out."""

    PROMOTIONAL = "promotional_content"
    LOW_QUALITY = "low_information_density"
    SPAM_OR_FARM = "content_farm_or_spam"
    MALFORMED_URL = "malformed_or_unsafe_url"
    DUPLICATE = "duplicate_content"
    IRRELEVANT = "zero_relevance"
    DOMAIN_CAP_EXCEEDED = "domain_diversity_cap_exceeded"


@dataclass(frozen=True)
class CandidateEvidence:
    """Technology-neutral representation of a candidate search result retrieved from an external provider."""

    url: str
    title: str
    content: str
    raw_score: float | None = None
    published_date: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.url, str):
            raise TypeError("url must be a string")
        if not isinstance(self.title, str):
            raise TypeError("title must be a string")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")


@dataclass(frozen=True)
class EvaluatedEvidence:
    """Evidence item evaluated for authority, relevance, provenance, and filtering."""

    url: str
    title: str
    content: str
    domain: str
    authority_tier: AuthorityTier
    source_type: SourceType
    quality_signals: tuple[str, ...]
    relevance_signal: str  # "high", "medium", "low"
    recency_indicator: str  # "recent", "historical", "unknown"
    rank_score: float  # Deterministic composite score (0.0 to 1.0)
    retained: bool  # True if passed filtering; False if filtered out
    filter_reason: FilterReason | None = None
    rationale: str = ""
    cross_domain_overlap_count: int = 0

    def __post_init__(self) -> None:
        if not self.domain.strip():
            raise ValueError("domain must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary with enum values expanded."""
        data = asdict(self)
        data["authority_tier"] = self.authority_tier.value
        data["source_type"] = self.source_type.value
        data["filter_reason"] = (
            self.filter_reason.value if self.filter_reason else None
        )
        data["quality_signals"] = list(self.quality_signals)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluatedEvidence":
        auth_raw = data.get("authority_tier", AuthorityTier.NEUTRAL.value)
        source_raw = data.get("source_type", SourceType.UNKNOWN.value)
        filter_raw = data.get("filter_reason")

        try:
            authority_tier = AuthorityTier(str(auth_raw))
        except ValueError:
            authority_tier = AuthorityTier.NEUTRAL

        try:
            source_type = SourceType(str(source_raw))
        except ValueError:
            source_type = SourceType.UNKNOWN

        filter_reason: FilterReason | None = None
        if filter_raw is not None:
            try:
                filter_reason = FilterReason(str(filter_raw))
            except ValueError:
                filter_reason = None

        overlap_val = data.get(
            "cross_domain_overlap_count", data.get("corroboration_count", 0)
        )

        return cls(
            url=str(data.get("url", "")),
            title=str(data.get("title", "")),
            content=str(data.get("content", "")),
            domain=str(data.get("domain", "")),
            authority_tier=authority_tier,
            source_type=source_type,
            quality_signals=tuple(str(s) for s in data.get("quality_signals", ())),
            relevance_signal=str(data.get("relevance_signal", "medium")),
            recency_indicator=str(data.get("recency_indicator", "unknown")),
            rank_score=float(data.get("rank_score", 0.0)),
            retained=bool(data.get("retained", True)),
            filter_reason=filter_reason,
            rationale=str(data.get("rationale", "")),
            cross_domain_overlap_count=int(overlap_val),
        )

