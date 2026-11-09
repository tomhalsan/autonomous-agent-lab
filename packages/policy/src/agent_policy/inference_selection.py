"""Deterministic inference provider selection policy and decision contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from agent_contracts import InferenceHealthStatus, InferenceProviderHealth


class ProviderLocality(str, Enum):
    """Provider execution locality classification."""

    LOCAL = "local"
    CLOUD = "cloud"


class CostTier(str, Enum):
    """Coarse provider cost classification tier."""

    FREE = "free"
    LOW = "low"
    STANDARD = "standard"

    @property
    def rank(self) -> int:
        """Numeric rank for deterministic comparison (lower rank = lower cost)."""
        return _COST_TIER_RANKS[self]


_COST_TIER_RANKS: dict[CostTier, int] = {
    CostTier.FREE: 0,
    CostTier.LOW: 1,
    CostTier.STANDARD: 2,
}


class SelectionStrategy(str, Enum):
    """Strategy guiding provider candidate ordering within the same health tier."""

    PRIORITY = "priority"
    LOCAL_FIRST = "local_first"
    CLOUD_FIRST = "cloud_first"
    COST_FIRST = "cost_first"


@dataclass(frozen=True)
class ProviderDescriptor:
    """Immutable provider descriptor supplied by caller/configuration.

    Encapsulates provider identity, execution locality, and coarse cost tier
    without encoding vendor-specific knowledge in the policy engine.
    """

    provider_name: str
    locality: ProviderLocality
    cost_tier: CostTier = CostTier.STANDARD

    def __post_init__(self) -> None:
        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        object.__setattr__(self, "provider_name", self.provider_name.strip().lower())

        if not isinstance(self.locality, ProviderLocality):
            raise TypeError(
                f"locality must be a ProviderLocality, got {type(self.locality).__name__}"
            )
        if not isinstance(self.cost_tier, CostTier):
            raise TypeError(
                f"cost_tier must be a CostTier, got {type(self.cost_tier).__name__}"
            )


@dataclass(frozen=True)
class RankedProviderCandidate:
    """An eligible provider candidate ranked by health status, preference strategy, and configured priority."""

    provider_name: str
    health_status: InferenceHealthStatus
    priority_index: int
    locality: ProviderLocality | None = None
    cost_tier: CostTier | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        object.__setattr__(self, "provider_name", self.provider_name.strip().lower())

        if not isinstance(self.health_status, InferenceHealthStatus):
            raise TypeError(
                f"health_status must be an InferenceHealthStatus, got {type(self.health_status).__name__}"
            )
        if not self.health_status.is_available:
            raise ValueError(
                f"Ranked candidate must be available (HEALTHY or DEGRADED), got {self.health_status.value}"
            )

        if (
            type(self.priority_index) is bool
            or not isinstance(self.priority_index, int)
            or self.priority_index < 0
        ):
            raise ValueError(
                f"priority_index must be a non-negative integer, got {self.priority_index!r}"
            )

        if self.locality is not None and not isinstance(self.locality, ProviderLocality):
            raise TypeError(
                f"locality must be a ProviderLocality or None, got {type(self.locality).__name__}"
            )
        if self.cost_tier is not None and not isinstance(self.cost_tier, CostTier):
            raise TypeError(
                f"cost_tier must be a CostTier or None, got {type(self.cost_tier).__name__}"
            )


@dataclass(frozen=True)
class ExcludedProviderCandidate:
    """A configured provider excluded from selection due to health constraints or missing data."""

    provider_name: str
    health_status: InferenceHealthStatus | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        object.__setattr__(self, "provider_name", self.provider_name.strip().lower())

        if self.health_status is not None and not isinstance(
            self.health_status, InferenceHealthStatus
        ):
            raise TypeError(
                f"health_status must be an InferenceHealthStatus or None, got {type(self.health_status).__name__}"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


@dataclass(frozen=True)
class ProviderSelectionDecision:
    """Immutable record of deterministic provider selection."""

    selected_provider: str | None
    status: InferenceHealthStatus | None
    reason: str
    ranked_candidates: tuple[RankedProviderCandidate, ...] = ()
    excluded_candidates: tuple[ExcludedProviderCandidate, ...] = ()
    strategy: SelectionStrategy = SelectionStrategy.PRIORITY

    def __post_init__(self) -> None:
        if isinstance(self.ranked_candidates, list):
            object.__setattr__(self, "ranked_candidates", tuple(self.ranked_candidates))
        if isinstance(self.excluded_candidates, list):
            object.__setattr__(self, "excluded_candidates", tuple(self.excluded_candidates))

        if not isinstance(self.strategy, SelectionStrategy):
            raise TypeError(
                f"strategy must be a SelectionStrategy, got {type(self.strategy).__name__}"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        # Validate elements types
        for cand in self.ranked_candidates:
            if not isinstance(cand, RankedProviderCandidate):
                raise TypeError(
                    f"ranked_candidates items must be RankedProviderCandidate, got {type(cand).__name__}"
                )
        for excl in self.excluded_candidates:
            if not isinstance(excl, ExcludedProviderCandidate):
                raise TypeError(
                    f"excluded_candidates items must be ExcludedProviderCandidate, got {type(excl).__name__}"
                )

        if self.selected_provider is not None:
            if not isinstance(self.selected_provider, str) or not self.selected_provider.strip():
                raise ValueError("selected_provider must be a non-empty string when provided")
            normalized_selected = self.selected_provider.strip().lower()
            object.__setattr__(self, "selected_provider", normalized_selected)

            if not isinstance(self.status, InferenceHealthStatus):
                raise TypeError(
                    f"status must be an InferenceHealthStatus when selected_provider is present, got {type(self.status).__name__}"
                )

            # Invariant: A selected provider requires non-empty ranked_candidates
            if not self.ranked_candidates:
                raise ValueError(
                    "ranked_candidates cannot be empty when a selected_provider is present"
                )

            # Invariant: selected_provider and status must match the first ranked candidate
            first_candidate = self.ranked_candidates[0]
            if normalized_selected != first_candidate.provider_name:
                raise ValueError(
                    f"selected_provider {normalized_selected!r} does not match first ranked candidate {first_candidate.provider_name!r}"
                )
            if self.status != first_candidate.health_status:
                raise ValueError(
                    f"status {self.status.value!r} does not match first ranked candidate status {first_candidate.health_status.value!r}"
                )
        else:
            # Invariant: selected_provider=None requires status=None and empty ranked_candidates
            if self.status is not None:
                raise ValueError(
                    f"status must be None when selected_provider is None, got {self.status.value!r}"
                )
            if self.ranked_candidates:
                raise ValueError(
                    f"ranked_candidates must be empty when selected_provider is None, got {len(self.ranked_candidates)} candidates"
                )

        # Invariant: Normalized provider identities must be unique within ranked_candidates
        ranked_names = [c.provider_name for c in self.ranked_candidates]
        if len(set(ranked_names)) != len(ranked_names):
            raise ValueError(
                f"Duplicate provider identity detected in ranked_candidates: {ranked_names}"
            )

        # Invariant: Normalized provider identities must be unique within excluded_candidates
        excluded_names = [e.provider_name for e in self.excluded_candidates]
        if len(set(excluded_names)) != len(excluded_names):
            raise ValueError(
                f"Duplicate provider identity detected in excluded_candidates: {excluded_names}"
            )

        # Invariant: No provider may appear in both ranked and excluded candidates
        intersection = set(ranked_names).intersection(set(excluded_names))
        if intersection:
            raise ValueError(
                f"Provider(s) cannot appear in both ranked and excluded candidates: {sorted(intersection)}"
            )


def parse_provider_priority(
    raw_priority: str | None,
    default: Sequence[str] | str | None = None,
) -> tuple[str, ...]:
    """Parse, normalize, and deduplicate a comma-separated provider priority string.

    Rules:
    - Rejects invalid types with TypeError.
    - An explicitly blank or comma-only string raises ValueError.
    - Normalizes case and whitespace; deduplicates while preserving first occurrence.
    - If raw_priority is None, uses supplied default (or raises ValueError if None).
    """
    if raw_priority is not None and not isinstance(raw_priority, str):
        raise TypeError(
            f"raw_priority must be a string or None, got {type(raw_priority).__name__}"
        )

    if raw_priority is None:
        if default is None:
            raise ValueError(
                "Provider priority configuration is required when default is not provided"
            )
        if isinstance(default, str):
            return parse_provider_priority(default, default=None)
        if isinstance(default, (list, tuple)):
            if isinstance(default, tuple) and not default:
                raise ValueError("Default provider priority sequence must not be empty")
            if isinstance(default, list) and not default:
                raise ValueError("Default provider priority sequence must not be empty")

            tokens: list[str] = []
            for item in default:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(
                        f"Default provider priority items must be non-empty strings, got {item!r}"
                    )
                tok = item.strip().lower()
                if tok not in tokens:
                    tokens.append(tok)
            if not tokens:
                raise ValueError(
                    "Default provider priority must contain at least one non-empty provider name"
                )
            return tuple(tokens)
        raise TypeError(
            f"default must be a string, sequence of strings, or None, got {type(default).__name__}"
        )

    # raw_priority is a string
    raw_tokens = raw_priority.split(",")
    valid_tokens = [t.strip().lower() for t in raw_tokens if t.strip()]
    if not valid_tokens:
        raise ValueError(
            f"Provider priority configuration must contain at least one non-empty provider name, got {raw_priority!r}"
        )

    deduped: list[str] = []
    for tok in valid_tokens:
        if tok not in deduped:
            deduped.append(tok)

    return tuple(deduped)


def _locality_rank_local_first(locality: ProviderLocality | None) -> int:
    """Explicit conservative ordering for LOCAL_FIRST: known local (0), unknown (1), known cloud (2)."""
    if locality == ProviderLocality.LOCAL:
        return 0
    if locality is None:
        return 1
    return 2


def _locality_rank_cloud_first(locality: ProviderLocality | None) -> int:
    """Explicit conservative ordering for CLOUD_FIRST: known cloud (0), unknown (1), known local (2)."""
    if locality == ProviderLocality.CLOUD:
        return 0
    if locality is None:
        return 1
    return 2


def _cost_rank_cost_first(cost_tier: CostTier | None) -> int:
    """Explicit conservative ordering for COST_FIRST: FREE (0), LOW (1), STANDARD (2), unknown (3)."""
    if cost_tier is not None:
        return cost_tier.rank
    return 3


def _sort_candidates(
    candidates: list[RankedProviderCandidate],
    strategy: SelectionStrategy,
) -> list[RankedProviderCandidate]:
    if strategy == SelectionStrategy.PRIORITY:
        return sorted(candidates, key=lambda c: c.priority_index)
    if strategy == SelectionStrategy.LOCAL_FIRST:
        return sorted(
            candidates,
            key=lambda c: (_locality_rank_local_first(c.locality), c.priority_index),
        )
    if strategy == SelectionStrategy.CLOUD_FIRST:
        return sorted(
            candidates,
            key=lambda c: (_locality_rank_cloud_first(c.locality), c.priority_index),
        )
    if strategy == SelectionStrategy.COST_FIRST:
        return sorted(
            candidates,
            key=lambda c: (_cost_rank_cost_first(c.cost_tier), c.priority_index),
        )
    raise ValueError(f"Unsupported strategy: {strategy!r}")


def _build_selection_reason(
    selected: RankedProviderCandidate,
    strategy: SelectionStrategy,
) -> str:
    """Build a concise, accurate reason describing health, degradation, and strategy."""
    if selected.health_status == InferenceHealthStatus.HEALTHY:
        health_clause = "healthy"
    else:
        health_clause = "degraded; no healthy providers available"

    if strategy == SelectionStrategy.PRIORITY:
        strategy_clause = "by configured priority"
    elif strategy == SelectionStrategy.LOCAL_FIRST:
        if selected.locality == ProviderLocality.LOCAL:
            strategy_clause = "by local-first strategy"
        elif selected.locality is None:
            strategy_clause = "by priority under local-first strategy (unclassified locality)"
        else:
            strategy_clause = "by priority under local-first strategy (no local candidate in selected health tier)"
    elif strategy == SelectionStrategy.CLOUD_FIRST:
        if selected.locality == ProviderLocality.CLOUD:
            strategy_clause = "by cloud-first strategy"
        elif selected.locality is None:
            strategy_clause = "by priority under cloud-first strategy (unclassified locality)"
        else:
            strategy_clause = "by priority under cloud-first strategy (no cloud candidate in selected health tier)"
    elif strategy == SelectionStrategy.COST_FIRST:
        if selected.cost_tier is not None:
            strategy_clause = "by cost-first strategy"
        else:
            strategy_clause = "by priority under cost-first strategy (unclassified cost)"
    else:
        strategy_clause = f"by {strategy.value} strategy"

    return f"Selected provider {selected.provider_name!r} ({health_clause}) {strategy_clause}"


class DeterministicProviderSelector:
    """Pure deterministic inference provider selector ranking by health, preference strategy, and priority."""

    def select(
        self,
        preferences: Sequence[str],
        health_snapshots: Sequence[InferenceProviderHealth],
        descriptors: Sequence[ProviderDescriptor] | None = None,
        strategy: SelectionStrategy = SelectionStrategy.PRIORITY,
    ) -> ProviderSelectionDecision:
        """Select an eligible inference provider based on health status, preference strategy, and priority order.

        Semantics:
        - Input health_snapshots must not contain duplicate normalized provider identities.
        - HEALTHY outranks DEGRADED regardless of strategy or configured priority.
        - Under PRIORITY, configured priority index breaks ties within the same health class.
        - Under LOCAL_FIRST, local providers are preferred within the same health class; ties broken by priority index.
        - Under CLOUD_FIRST, cloud providers are preferred within the same health class; ties broken by priority index.
        - Under COST_FIRST, lower cost tiers are preferred within the same health class; ties broken by priority index.
        - Missing or unclassified metadata is handled safely without assumptions.
        - RATE_LIMITED, QUOTA_EXHAUSTED, and UNAVAILABLE providers are excluded.
        - Missing health is represented by health_status=None, never fabricated as UNAVAILABLE.
        - If no eligible providers exist, selected_provider is None.
        """
        if not isinstance(strategy, SelectionStrategy):
            raise TypeError(
                f"strategy must be a SelectionStrategy, got {type(strategy).__name__}"
            )

        if isinstance(preferences, (str, bytes)) or not isinstance(preferences, (list, tuple)):
            raise TypeError(
                f"preferences must be a sequence of strings (list or tuple), got {type(preferences).__name__}"
            )

        if isinstance(health_snapshots, (str, bytes)) or not isinstance(
            health_snapshots, (list, tuple)
        ):
            raise TypeError(
                f"health_snapshots must be a sequence of InferenceProviderHealth, got {type(health_snapshots).__name__}"
            )

        descriptors_by_name: dict[str, ProviderDescriptor] = {}
        if descriptors is not None:
            if isinstance(descriptors, (str, bytes)) or not isinstance(
                descriptors, (list, tuple)
            ):
                raise TypeError(
                    f"descriptors must be a sequence of ProviderDescriptor, got {type(descriptors).__name__}"
                )
            for desc in descriptors:
                if not isinstance(desc, ProviderDescriptor):
                    raise TypeError(
                        f"descriptors items must be ProviderDescriptor instances, got {type(desc).__name__}"
                    )
                norm_desc_name = desc.provider_name
                if norm_desc_name in descriptors_by_name:
                    raise ValueError(
                        f"Duplicate provider descriptor provided for {norm_desc_name!r}"
                    )
                descriptors_by_name[norm_desc_name] = desc

        # Validate preferences
        configured_providers: list[str] = []
        for p in preferences:
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"preference entries must be non-empty strings, got {p!r}")
            norm_p = p.strip().lower()
            if norm_p not in configured_providers:
                configured_providers.append(norm_p)

        if not configured_providers:
            raise ValueError("preferences must contain at least one provider identity")

        # Validate snapshots and build lookup map; reject duplicates
        snapshots_by_name: dict[str, InferenceProviderHealth] = {}
        for snap in health_snapshots:
            if not isinstance(snap, InferenceProviderHealth):
                raise TypeError(
                    f"health_snapshots items must be InferenceProviderHealth instances, got {type(snap).__name__}"
                )
            snap_name = snap.provider_name.strip().lower()
            if snap_name in snapshots_by_name:
                raise ValueError(
                    f"Duplicate health snapshot provided for provider {snap_name!r}"
                )
            snapshots_by_name[snap_name] = snap

        healthy_candidates: list[RankedProviderCandidate] = []
        degraded_candidates: list[RankedProviderCandidate] = []
        excluded_candidates: list[ExcludedProviderCandidate] = []

        for idx, provider_name in enumerate(configured_providers):
            if provider_name not in snapshots_by_name:
                excluded_candidates.append(
                    ExcludedProviderCandidate(
                        provider_name=provider_name,
                        health_status=None,
                        reason="No health snapshot provided",
                    )
                )
            else:
                snap = snapshots_by_name[provider_name]
                desc = descriptors_by_name.get(provider_name)
                locality = desc.locality if desc is not None else None
                cost_tier = desc.cost_tier if desc is not None else None

                if snap.status == InferenceHealthStatus.HEALTHY:
                    healthy_candidates.append(
                        RankedProviderCandidate(
                            provider_name=provider_name,
                            health_status=snap.status,
                            priority_index=idx,
                            locality=locality,
                            cost_tier=cost_tier,
                        )
                    )
                elif snap.status == InferenceHealthStatus.DEGRADED:
                    degraded_candidates.append(
                        RankedProviderCandidate(
                            provider_name=provider_name,
                            health_status=snap.status,
                            priority_index=idx,
                            locality=locality,
                            cost_tier=cost_tier,
                        )
                    )
                else:
                    # RATE_LIMITED, QUOTA_EXHAUSTED, UNAVAILABLE
                    reason_msg = snap.reason.strip() if snap.reason else f"Provider is {snap.status.value}"
                    excluded_candidates.append(
                        ExcludedProviderCandidate(
                            provider_name=provider_name,
                            health_status=snap.status,
                            reason=reason_msg,
                        )
                    )

        # Ranked candidates: HEALTHY candidates ordered by strategy, then DEGRADED candidates ordered by strategy
        ranked_healthy = _sort_candidates(healthy_candidates, strategy)
        ranked_degraded = _sort_candidates(degraded_candidates, strategy)
        ranked = tuple(ranked_healthy + ranked_degraded)
        excluded = tuple(excluded_candidates)

        if ranked:
            selected = ranked[0]
            reason = _build_selection_reason(
                selected=selected,
                strategy=strategy,
            )

            return ProviderSelectionDecision(
                selected_provider=selected.provider_name,
                status=selected.health_status,
                reason=reason,
                ranked_candidates=ranked,
                excluded_candidates=excluded,
                strategy=strategy,
            )

        return ProviderSelectionDecision(
            selected_provider=None,
            status=None,
            reason=(
                "No eligible inference providers available (all configured providers are "
                "unavailable, rate-limited, quota-exhausted, or missing health data)"
            ),
            ranked_candidates=(),
            excluded_candidates=excluded,
            strategy=strategy,
        )
