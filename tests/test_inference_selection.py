"""Unit tests for pure deterministic inference provider selection policy."""

from dataclasses import FrozenInstanceError

import pytest
from agent_contracts import InferenceHealthStatus, InferenceProviderHealth
from agent_policy import (
    CostTier,
    DeterministicProviderSelector,
    ExcludedProviderCandidate,
    ProviderDescriptor,
    ProviderLocality,
    ProviderSelectionDecision,
    RankedProviderCandidate,
    SelectionStrategy,
    parse_provider_priority,
)


def _make_health(
    provider_name: str,
    status: InferenceHealthStatus,
    reason: str = "",
) -> InferenceProviderHealth:
    return InferenceProviderHealth(
        provider_name=provider_name,
        status=status,
        reason=reason or f"Status is {status.value}",
    )


# --- 1. Health-First, Configured-Priority-Second Ordering ---


def test_healthy_outranks_degraded_regardless_of_configured_priority() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["cloudflare", "gemini"]

    # cloudflare is priority 0 but DEGRADED; gemini is priority 1 but HEALTHY
    snapshots = [
        _make_health("cloudflare", InferenceHealthStatus.DEGRADED),
        _make_health("gemini", InferenceHealthStatus.HEALTHY),
    ]

    decision = selector.select(preferences, snapshots)

    assert decision.selected_provider == "gemini"
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert decision.reason == "Selected provider 'gemini' (healthy) by configured priority"
    assert len(decision.ranked_candidates) == 2
    assert decision.ranked_candidates[0].provider_name == "gemini"
    assert decision.ranked_candidates[0].health_status == InferenceHealthStatus.HEALTHY
    assert decision.ranked_candidates[0].priority_index == 1

    assert decision.ranked_candidates[1].provider_name == "cloudflare"
    assert decision.ranked_candidates[1].health_status == InferenceHealthStatus.DEGRADED
    assert decision.ranked_candidates[1].priority_index == 0


def test_configured_priority_breaks_ties_within_healthy_class() -> None:
    selector = DeterministicProviderSelector()

    # Preference order A: gemini then cloudflare
    snapshots = [
        _make_health("cloudflare", InferenceHealthStatus.HEALTHY),
        _make_health("gemini", InferenceHealthStatus.HEALTHY),
    ]

    dec1 = selector.select(["gemini", "cloudflare"], snapshots)
    assert dec1.selected_provider == "gemini"
    assert dec1.ranked_candidates[0].provider_name == "gemini"
    assert dec1.ranked_candidates[1].provider_name == "cloudflare"

    # Preference order B: cloudflare then gemini
    dec2 = selector.select(["cloudflare", "gemini"], snapshots)
    assert dec2.selected_provider == "cloudflare"
    assert dec2.ranked_candidates[0].provider_name == "cloudflare"
    assert dec2.ranked_candidates[1].provider_name == "gemini"


def test_degraded_remains_eligible_and_breaks_ties_by_priority() -> None:
    selector = DeterministicProviderSelector()
    snapshots = [
        _make_health("p1", InferenceHealthStatus.DEGRADED),
        _make_health("p2", InferenceHealthStatus.DEGRADED),
    ]

    decision = selector.select(["p1", "p2"], snapshots)
    assert decision.selected_provider == "p1"
    assert decision.status == InferenceHealthStatus.DEGRADED
    assert decision.ranked_candidates[0].provider_name == "p1"
    assert decision.ranked_candidates[1].provider_name == "p2"
    assert (
        decision.reason
        == "Selected provider 'p1' (degraded; no healthy providers available) by configured priority"
    )


def test_degraded_selected_when_higher_priority_unavailable() -> None:
    selector = DeterministicProviderSelector()
    snapshots = [
        _make_health("p1", InferenceHealthStatus.UNAVAILABLE, reason="endpoint offline"),
        _make_health("p2", InferenceHealthStatus.DEGRADED),
    ]

    decision = selector.select(["p1", "p2"], snapshots)
    assert decision.selected_provider == "p2"
    assert decision.status == InferenceHealthStatus.DEGRADED
    assert (
        decision.reason
        == "Selected provider 'p2' (degraded; no healthy providers available) by configured priority"
    )
    assert len(decision.ranked_candidates) == 1
    assert decision.ranked_candidates[0].provider_name == "p2"
    assert len(decision.excluded_candidates) == 1
    assert decision.excluded_candidates[0].provider_name == "p1"
    assert decision.excluded_candidates[0].health_status == InferenceHealthStatus.UNAVAILABLE


# --- 1b. Concise Reason Generation Tests ---


def test_reason_healthy_selected_by_priority() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["gemini", "cloudflare"]
    snapshots = [
        _make_health("gemini", InferenceHealthStatus.HEALTHY),
        _make_health("cloudflare", InferenceHealthStatus.HEALTHY),
    ]

    decision = selector.select(preferences, snapshots)

    assert decision.selected_provider == "gemini"
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert decision.reason == "Selected provider 'gemini' (healthy) by configured priority"


def test_reason_lower_priority_healthy_selected_over_higher_priority_degraded() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["cloudflare", "gemini"]
    snapshots = [
        _make_health("cloudflare", InferenceHealthStatus.DEGRADED),
        _make_health("gemini", InferenceHealthStatus.HEALTHY),
    ]

    decision = selector.select(preferences, snapshots)

    assert decision.selected_provider == "gemini"
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert decision.reason == "Selected provider 'gemini' (healthy) by configured priority"


def test_reason_lower_priority_healthy_selected_when_higher_priority_excluded() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["cloudflare", "gemini"]
    snapshots = [
        _make_health("cloudflare", InferenceHealthStatus.RATE_LIMITED),
        _make_health("gemini", InferenceHealthStatus.HEALTHY),
    ]

    decision = selector.select(preferences, snapshots)

    assert decision.selected_provider == "gemini"
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert decision.reason == "Selected provider 'gemini' (healthy) by configured priority"


def test_reason_degraded_provider_selected_when_no_healthy_providers_exist() -> None:
    selector = DeterministicProviderSelector()

    # Case A: degraded provider is index 0
    decision_top = selector.select(
        ["cloudflare", "gemini"],
        [
            _make_health("cloudflare", InferenceHealthStatus.DEGRADED),
            _make_health("gemini", InferenceHealthStatus.DEGRADED),
        ],
    )
    assert decision_top.selected_provider == "cloudflare"
    assert decision_top.status == InferenceHealthStatus.DEGRADED
    assert (
        decision_top.reason
        == "Selected provider 'cloudflare' (degraded; no healthy providers available) by configured priority"
    )

    # Case B: degraded provider is lower-priority (higher-priority was excluded)
    decision_lower = selector.select(
        ["cloudflare", "gemini"],
        [
            _make_health("cloudflare", InferenceHealthStatus.UNAVAILABLE),
            _make_health("gemini", InferenceHealthStatus.DEGRADED),
        ],
    )
    assert decision_lower.selected_provider == "gemini"
    assert decision_lower.status == InferenceHealthStatus.DEGRADED
    assert (
        decision_lower.reason
        == "Selected provider 'gemini' (degraded; no healthy providers available) by configured priority"
    )


# --- 2. Exclusion of Ineligible and Missing Health States ---


def test_rate_limited_quota_exhausted_and_unavailable_are_excluded() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["p_rl", "p_qe", "p_un", "p_ok"]
    snapshots = [
        _make_health("p_rl", InferenceHealthStatus.RATE_LIMITED, "Too many requests"),
        _make_health("p_qe", InferenceHealthStatus.QUOTA_EXHAUSTED, "Daily tokens exceeded"),
        _make_health("p_un", InferenceHealthStatus.UNAVAILABLE, "HTTP 503"),
        _make_health("p_ok", InferenceHealthStatus.HEALTHY),
    ]

    decision = selector.select(preferences, snapshots)

    assert decision.selected_provider == "p_ok"
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert len(decision.ranked_candidates) == 1

    excluded_map = {e.provider_name: e for e in decision.excluded_candidates}
    assert len(excluded_map) == 3

    assert excluded_map["p_rl"].health_status == InferenceHealthStatus.RATE_LIMITED
    assert "Too many requests" in excluded_map["p_rl"].reason

    assert excluded_map["p_qe"].health_status == InferenceHealthStatus.QUOTA_EXHAUSTED
    assert "Daily tokens exceeded" in excluded_map["p_qe"].reason

    assert excluded_map["p_un"].health_status == InferenceHealthStatus.UNAVAILABLE
    assert "HTTP 503" in excluded_map["p_un"].reason


def test_missing_health_is_represented_as_unknown_health_none() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["p_has_health", "p_missing_health"]
    snapshots = [
        _make_health("p_has_health", InferenceHealthStatus.HEALTHY),
        # p_missing_health is intentionally omitted from snapshots
    ]

    decision = selector.select(preferences, snapshots)

    assert decision.selected_provider == "p_has_health"
    assert len(decision.excluded_candidates) == 1
    excluded = decision.excluded_candidates[0]
    assert excluded.provider_name == "p_missing_health"
    # Invariant: Must be None, never fabricated as UNAVAILABLE
    assert excluded.health_status is None
    assert excluded.reason == "No health snapshot provided"


def test_all_providers_ineligible_returns_none_selection() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["p1", "p2", "p3"]
    snapshots = [
        _make_health("p1", InferenceHealthStatus.RATE_LIMITED),
        _make_health("p2", InferenceHealthStatus.UNAVAILABLE),
        # p3 missing snapshot
    ]

    decision = selector.select(preferences, snapshots)

    assert decision.selected_provider is None
    assert decision.status is None
    assert decision.ranked_candidates == ()
    assert len(decision.excluded_candidates) == 3
    assert "No eligible inference providers available" in decision.reason


# --- 3. Duplicate Health Snapshots Rejection ---


def test_duplicate_normalized_health_snapshots_raise_value_error() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["gemini", "cloudflare"]

    # Exact duplicate
    snapshots_exact = [
        _make_health("gemini", InferenceHealthStatus.HEALTHY),
        _make_health("gemini", InferenceHealthStatus.DEGRADED),
    ]
    with pytest.raises(ValueError, match="Duplicate health snapshot provided for provider 'gemini'"):
        selector.select(preferences, snapshots_exact)

    # Case-insensitive / whitespace duplicate
    snapshots_casing = [
        _make_health("Cloudflare", InferenceHealthStatus.HEALTHY),
        _make_health("  cloudflare  ", InferenceHealthStatus.HEALTHY),
    ]
    with pytest.raises(ValueError, match="Duplicate health snapshot provided for provider 'cloudflare'"):
        selector.select(preferences, snapshots_casing)


# --- 4. Vendor Neutrality and Opaque Provider Identities ---


def test_opaque_vendor_neutral_provider_identities() -> None:
    selector = DeterministicProviderSelector()
    # Arbitrary non-vendor names
    preferences = ["custom-local-llm", "enterprise-gateway-v2", "embedded-model"]
    snapshots = [
        _make_health("custom-local-llm", InferenceHealthStatus.DEGRADED),
        _make_health("enterprise-gateway-v2", InferenceHealthStatus.HEALTHY),
        _make_health("embedded-model", InferenceHealthStatus.UNAVAILABLE),
    ]

    decision = selector.select(preferences, snapshots)
    assert decision.selected_provider == "enterprise-gateway-v2"
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert decision.ranked_candidates[0].provider_name == "enterprise-gateway-v2"
    assert decision.ranked_candidates[1].provider_name == "custom-local-llm"
    assert decision.excluded_candidates[0].provider_name == "embedded-model"


# --- 5. Configuration Parser (`parse_provider_priority`) ---


def test_parse_provider_priority_normalization_and_deduplication() -> None:
    raw = "  Gemini , CLOUDFLARE,  gemini  , stub, Cloudflare "
    result = parse_provider_priority(raw)
    assert result == ("gemini", "cloudflare", "stub")


def test_parse_provider_priority_handles_none_with_default() -> None:
    # String default
    res1 = parse_provider_priority(None, default="stub")
    assert res1 == ("stub",)

    # Sequence default
    res2 = parse_provider_priority(None, default=["Gemini", "Cloudflare"])
    assert res2 == ("gemini", "cloudflare")

    # None without default raises ValueError
    with pytest.raises(ValueError, match="Provider priority configuration is required"):
        parse_provider_priority(None, default=None)


def test_parse_provider_priority_blank_and_comma_only_fail() -> None:
    with pytest.raises(ValueError, match="must contain at least one non-empty provider name"):
        parse_provider_priority("")

    with pytest.raises(ValueError, match="must contain at least one non-empty provider name"):
        parse_provider_priority("   ")

    with pytest.raises(ValueError, match="must contain at least one non-empty provider name"):
        parse_provider_priority(",")

    with pytest.raises(ValueError, match="must contain at least one non-empty provider name"):
        parse_provider_priority(", , ,,,")


def test_parse_provider_priority_invalid_types_raise_type_error() -> None:
    with pytest.raises(TypeError, match="raw_priority must be a string or None"):
        parse_provider_priority(123)  # type: ignore

    with pytest.raises(TypeError, match="raw_priority must be a string or None"):
        parse_provider_priority(True)  # type: ignore

    with pytest.raises(TypeError, match="raw_priority must be a string or None"):
        parse_provider_priority(["gemini"])  # type: ignore

    with pytest.raises(TypeError, match="default must be a string, sequence of strings, or None"):
        parse_provider_priority(None, default=456)  # type: ignore


# --- 6. Decision and Candidate Invariant Verification ---


def test_decision_invariants_prevent_contradictory_state() -> None:
    # 1. selected_provider=None requires status=None
    with pytest.raises(ValueError, match="status must be None when selected_provider is None"):
        ProviderSelectionDecision(
            selected_provider=None,
            status=InferenceHealthStatus.HEALTHY,
            reason="Contradictory state",
        )

    # 2. selected_provider=None requires empty ranked_candidates
    with pytest.raises(ValueError, match="ranked_candidates must be empty when selected_provider is None"):
        ProviderSelectionDecision(
            selected_provider=None,
            status=None,
            reason="Contradictory state",
            ranked_candidates=(
                RankedProviderCandidate("gemini", InferenceHealthStatus.HEALTHY, 0),
            ),
        )

    # 3. Selected provider requires non-empty ranked_candidates
    with pytest.raises(ValueError, match="ranked_candidates cannot be empty when a selected_provider is present"):
        ProviderSelectionDecision(
            selected_provider="gemini",
            status=InferenceHealthStatus.HEALTHY,
            reason="Missing candidates",
            ranked_candidates=(),
        )

    # 4. selected_provider and status must match first ranked candidate
    with pytest.raises(ValueError, match="selected_provider 'gemini' does not match first ranked candidate 'cloudflare'"):
        ProviderSelectionDecision(
            selected_provider="gemini",
            status=InferenceHealthStatus.HEALTHY,
            reason="Mismatch candidate",
            ranked_candidates=(
                RankedProviderCandidate("cloudflare", InferenceHealthStatus.HEALTHY, 0),
            ),
        )

    with pytest.raises(ValueError, match="status 'degraded' does not match first ranked candidate status 'healthy'"):
        ProviderSelectionDecision(
            selected_provider="gemini",
            status=InferenceHealthStatus.DEGRADED,
            reason="Mismatch status",
            ranked_candidates=(
                RankedProviderCandidate("gemini", InferenceHealthStatus.HEALTHY, 0),
            ),
        )

    # 5. Duplicate names within ranked_candidates
    with pytest.raises(ValueError, match="Duplicate provider identity detected in ranked_candidates"):
        ProviderSelectionDecision(
            selected_provider="gemini",
            status=InferenceHealthStatus.HEALTHY,
            reason="Duplicate ranked",
            ranked_candidates=(
                RankedProviderCandidate("gemini", InferenceHealthStatus.HEALTHY, 0),
                RankedProviderCandidate("gemini", InferenceHealthStatus.HEALTHY, 1),
            ),
        )

    # 6. Duplicate names within excluded_candidates
    with pytest.raises(ValueError, match="Duplicate provider identity detected in excluded_candidates"):
        ProviderSelectionDecision(
            selected_provider="gemini",
            status=InferenceHealthStatus.HEALTHY,
            reason="Duplicate excluded",
            ranked_candidates=(
                RankedProviderCandidate("gemini", InferenceHealthStatus.HEALTHY, 0),
            ),
            excluded_candidates=(
                ExcludedProviderCandidate("cloudflare", InferenceHealthStatus.UNAVAILABLE, "err1"),
                ExcludedProviderCandidate("cloudflare", InferenceHealthStatus.UNAVAILABLE, "err2"),
            ),
        )

    # 7. Overlap between ranked and excluded candidates
    with pytest.raises(ValueError, match="cannot appear in both ranked and excluded"):
        ProviderSelectionDecision(
            selected_provider="gemini",
            status=InferenceHealthStatus.HEALTHY,
            reason="Overlap",
            ranked_candidates=(
                RankedProviderCandidate("gemini", InferenceHealthStatus.HEALTHY, 0),
            ),
            excluded_candidates=(
                ExcludedProviderCandidate("gemini", InferenceHealthStatus.UNAVAILABLE, "err"),
            ),
        )


def test_candidate_invariants_and_immutability() -> None:
    # Ranked candidate validation
    with pytest.raises(ValueError, match="provider_name must be a non-empty string"):
        RankedProviderCandidate("", InferenceHealthStatus.HEALTHY, 0)

    with pytest.raises(ValueError, match="Ranked candidate must be available"):
        RankedProviderCandidate("p1", InferenceHealthStatus.UNAVAILABLE, 0)

    with pytest.raises(ValueError, match="Ranked candidate must be available"):
        RankedProviderCandidate("p1", InferenceHealthStatus.RATE_LIMITED, 0)

    with pytest.raises(ValueError, match="priority_index must be a non-negative integer"):
        RankedProviderCandidate("p1", InferenceHealthStatus.HEALTHY, -1)

    with pytest.raises(ValueError, match="priority_index must be a non-negative integer"):
        RankedProviderCandidate("p1", InferenceHealthStatus.HEALTHY, True)  # type: ignore

    # Immutability
    candidate = RankedProviderCandidate("gemini", InferenceHealthStatus.HEALTHY, 0)
    with pytest.raises(FrozenInstanceError):
        candidate.provider_name = "cloudflare"  # type: ignore

    exclusion = ExcludedProviderCandidate("cf", InferenceHealthStatus.RATE_LIMITED, "Rate limited")
    with pytest.raises(FrozenInstanceError):
        exclusion.reason = "New reason"  # type: ignore


# --- 7. Selector Input Validation ---


def test_selector_input_validation() -> None:
    selector = DeterministicProviderSelector()

    # Preferences must be a list/tuple
    with pytest.raises(TypeError, match="preferences must be a sequence of strings"):
        selector.select("gemini,cloudflare", [])  # type: ignore

    with pytest.raises(ValueError, match="preference entries must be non-empty strings"):
        selector.select([""], [])

    with pytest.raises(ValueError, match="preferences must contain at least one provider identity"):
        selector.select([], [])

    # Snapshots must be list/tuple
    with pytest.raises(TypeError, match="health_snapshots must be a sequence of InferenceProviderHealth"):
        selector.select(["gemini"], "not-a-list")  # type: ignore

    with pytest.raises(TypeError, match="health_snapshots items must be InferenceProviderHealth"):
        selector.select(["gemini"], [{"provider_name": "gemini"}])  # type: ignore


# --- 8. Strategy-Guided Selection & Provider-Neutral Descriptors ---


def test_strategy_local_first_prefers_healthy_local_over_higher_priority_cloud() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["cloud_provider", "local_provider"]
    snapshots = [
        _make_health("cloud_provider", InferenceHealthStatus.HEALTHY),
        _make_health("local_provider", InferenceHealthStatus.HEALTHY),
    ]
    descriptors = [
        ProviderDescriptor("cloud_provider", ProviderLocality.CLOUD, CostTier.STANDARD),
        ProviderDescriptor("local_provider", ProviderLocality.LOCAL, CostTier.FREE),
    ]

    decision = selector.select(
        preferences,
        snapshots,
        descriptors=descriptors,
        strategy=SelectionStrategy.LOCAL_FIRST,
    )

    assert decision.selected_provider == "local_provider"
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert decision.strategy == SelectionStrategy.LOCAL_FIRST
    assert decision.ranked_candidates[0].provider_name == "local_provider"
    assert decision.ranked_candidates[0].locality == ProviderLocality.LOCAL
    assert decision.ranked_candidates[0].cost_tier == CostTier.FREE
    assert decision.ranked_candidates[1].provider_name == "cloud_provider"
    assert decision.reason == "Selected provider 'local_provider' (healthy) by local-first strategy"


def test_strategy_cloud_first_prefers_healthy_cloud_over_higher_priority_local() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["local_provider", "cloud_provider"]
    snapshots = [
        _make_health("local_provider", InferenceHealthStatus.HEALTHY),
        _make_health("cloud_provider", InferenceHealthStatus.HEALTHY),
    ]
    descriptors = [
        ProviderDescriptor("local_provider", ProviderLocality.LOCAL, CostTier.FREE),
        ProviderDescriptor("cloud_provider", ProviderLocality.CLOUD, CostTier.STANDARD),
    ]

    decision = selector.select(
        preferences,
        snapshots,
        descriptors=descriptors,
        strategy=SelectionStrategy.CLOUD_FIRST,
    )

    assert decision.selected_provider == "cloud_provider"
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert decision.strategy == SelectionStrategy.CLOUD_FIRST
    assert decision.ranked_candidates[0].provider_name == "cloud_provider"
    assert decision.ranked_candidates[0].locality == ProviderLocality.CLOUD
    assert decision.ranked_candidates[1].provider_name == "local_provider"
    assert decision.reason == "Selected provider 'cloud_provider' (healthy) by cloud-first strategy"


def test_strategy_cost_first_prefers_lower_cost_over_higher_priority_expensive() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["provider_expensive", "provider_cheap", "provider_free"]
    snapshots = [
        _make_health("provider_expensive", InferenceHealthStatus.HEALTHY),
        _make_health("provider_cheap", InferenceHealthStatus.HEALTHY),
        _make_health("provider_free", InferenceHealthStatus.HEALTHY),
    ]
    descriptors = [
        ProviderDescriptor("provider_expensive", ProviderLocality.CLOUD, CostTier.STANDARD),
        ProviderDescriptor("provider_cheap", ProviderLocality.CLOUD, CostTier.LOW),
        ProviderDescriptor("provider_free", ProviderLocality.LOCAL, CostTier.FREE),
    ]

    decision = selector.select(
        preferences,
        snapshots,
        descriptors=descriptors,
        strategy=SelectionStrategy.COST_FIRST,
    )

    assert decision.selected_provider == "provider_free"
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert decision.strategy == SelectionStrategy.COST_FIRST
    assert [c.provider_name for c in decision.ranked_candidates] == [
        "provider_free",
        "provider_cheap",
        "provider_expensive",
    ]
    assert [c.cost_tier for c in decision.ranked_candidates] == [
        CostTier.FREE,
        CostTier.LOW,
        CostTier.STANDARD,
    ]
    assert decision.reason == "Selected provider 'provider_free' (healthy) by cost-first strategy"


@pytest.mark.parametrize(
    ("strategy", "healthy_desc", "degraded_desc"),
    [
        (
            SelectionStrategy.PRIORITY,
            ProviderDescriptor("p_cloud", ProviderLocality.CLOUD, CostTier.STANDARD),
            ProviderDescriptor("p_local", ProviderLocality.LOCAL, CostTier.FREE),
        ),
        (
            SelectionStrategy.LOCAL_FIRST,
            ProviderDescriptor("p_cloud", ProviderLocality.CLOUD, CostTier.STANDARD),
            ProviderDescriptor("p_local", ProviderLocality.LOCAL, CostTier.FREE),
        ),
        (
            SelectionStrategy.CLOUD_FIRST,
            ProviderDescriptor("p_local", ProviderLocality.LOCAL, CostTier.FREE),
            ProviderDescriptor("p_cloud", ProviderLocality.CLOUD, CostTier.STANDARD),
        ),
        (
            SelectionStrategy.COST_FIRST,
            ProviderDescriptor("p_paid", ProviderLocality.CLOUD, CostTier.STANDARD),
            ProviderDescriptor("p_free", ProviderLocality.LOCAL, CostTier.FREE),
        ),
    ],
)
def test_health_precedence_intact_under_all_strategies(
    strategy: SelectionStrategy,
    healthy_desc: ProviderDescriptor,
    degraded_desc: ProviderDescriptor,
) -> None:
    selector = DeterministicProviderSelector()
    # Preference order puts the degraded candidate first:
    preferences = [degraded_desc.provider_name, healthy_desc.provider_name]
    snapshots = [
        _make_health(degraded_desc.provider_name, InferenceHealthStatus.DEGRADED),
        _make_health(healthy_desc.provider_name, InferenceHealthStatus.HEALTHY),
    ]
    descriptors = [degraded_desc, healthy_desc]

    decision = selector.select(
        preferences,
        snapshots,
        descriptors=descriptors,
        strategy=strategy,
    )

    assert decision.selected_provider == healthy_desc.provider_name
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert decision.ranked_candidates[0].provider_name == healthy_desc.provider_name
    assert decision.ranked_candidates[1].provider_name == degraded_desc.provider_name
    assert "(healthy)" in decision.reason


def test_complete_locality_ordering_including_unknown_metadata() -> None:
    selector = DeterministicProviderSelector()
    # Preferences: cloud (priority 0), unknown (priority 1), local (priority 2)
    preferences = ["p_cloud", "p_unknown", "p_local"]
    snapshots = [
        _make_health("p_cloud", InferenceHealthStatus.HEALTHY),
        _make_health("p_unknown", InferenceHealthStatus.HEALTHY),
        _make_health("p_local", InferenceHealthStatus.HEALTHY),
    ]
    descriptors = [
        ProviderDescriptor("p_cloud", ProviderLocality.CLOUD),
        ProviderDescriptor("p_local", ProviderLocality.LOCAL),
        # p_unknown has NO descriptor
    ]

    # LOCAL_FIRST deliberate ordering: known local (0), unknown (1), known cloud (2)
    dec_local = selector.select(
        preferences,
        snapshots,
        descriptors=descriptors,
        strategy=SelectionStrategy.LOCAL_FIRST,
    )
    assert [c.provider_name for c in dec_local.ranked_candidates] == ["p_local", "p_unknown", "p_cloud"]
    assert dec_local.selected_provider == "p_local"

    # CLOUD_FIRST deliberate ordering: known cloud (0), unknown (1), known local (2)
    dec_cloud = selector.select(
        preferences,
        snapshots,
        descriptors=descriptors,
        strategy=SelectionStrategy.CLOUD_FIRST,
    )
    assert [c.provider_name for c in dec_cloud.ranked_candidates] == ["p_cloud", "p_unknown", "p_local"]
    assert dec_cloud.selected_provider == "p_cloud"


def test_complete_cost_tier_ordering_including_unknown_metadata() -> None:
    selector = DeterministicProviderSelector()
    # Preferences order puts unknown first, then standard, low, free:
    preferences = ["p_unknown", "p_standard", "p_low", "p_free"]
    snapshots = [
        _make_health("p_unknown", InferenceHealthStatus.HEALTHY),
        _make_health("p_standard", InferenceHealthStatus.HEALTHY),
        _make_health("p_low", InferenceHealthStatus.HEALTHY),
        _make_health("p_free", InferenceHealthStatus.HEALTHY),
    ]
    descriptors = [
        ProviderDescriptor("p_standard", ProviderLocality.CLOUD, CostTier.STANDARD),
        ProviderDescriptor("p_low", ProviderLocality.CLOUD, CostTier.LOW),
        ProviderDescriptor("p_free", ProviderLocality.LOCAL, CostTier.FREE),
        # p_unknown has NO descriptor
    ]

    # COST_FIRST deliberate ordering: FREE (0), LOW (1), STANDARD (2), unknown (3)
    decision = selector.select(
        preferences,
        snapshots,
        descriptors=descriptors,
        strategy=SelectionStrategy.COST_FIRST,
    )
    assert [c.provider_name for c in decision.ranked_candidates] == [
        "p_free",
        "p_low",
        "p_standard",
        "p_unknown",
    ]
    assert decision.selected_provider == "p_free"


def test_priority_behavior_when_all_metadata_is_unknown() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["p_first", "p_second", "p_third"]
    snapshots = [
        _make_health("p_first", InferenceHealthStatus.HEALTHY),
        _make_health("p_second", InferenceHealthStatus.HEALTHY),
        _make_health("p_third", InferenceHealthStatus.HEALTHY),
    ]

    # When all metadata is unknown, every strategy must preserve priority ordering:
    for strategy, expected_reason in [
        (
            SelectionStrategy.PRIORITY,
            "Selected provider 'p_first' (healthy) by configured priority",
        ),
        (
            SelectionStrategy.LOCAL_FIRST,
            "Selected provider 'p_first' (healthy) by priority under local-first strategy (unclassified locality)",
        ),
        (
            SelectionStrategy.CLOUD_FIRST,
            "Selected provider 'p_first' (healthy) by priority under cloud-first strategy (unclassified locality)",
        ),
        (
            SelectionStrategy.COST_FIRST,
            "Selected provider 'p_first' (healthy) by priority under cost-first strategy (unclassified cost)",
        ),
    ]:
        decision = selector.select(preferences, snapshots, descriptors=None, strategy=strategy)
        assert decision.selected_provider == "p_first"
        assert [c.provider_name for c in decision.ranked_candidates] == ["p_first", "p_second", "p_third"]
        assert decision.reason == expected_reason


def test_unreferenced_descriptors_safely_ignored() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["gemini", "cloudflare"]
    snapshots = [
        _make_health("gemini", InferenceHealthStatus.HEALTHY),
        _make_health("cloudflare", InferenceHealthStatus.HEALTHY),
    ]
    descriptors = [
        ProviderDescriptor("gemini", ProviderLocality.CLOUD, CostTier.STANDARD),
        ProviderDescriptor("cloudflare", ProviderLocality.CLOUD, CostTier.LOW),
        ProviderDescriptor("unreferenced_extra", ProviderLocality.LOCAL, CostTier.FREE),
    ]

    decision = selector.select(preferences, snapshots, descriptors=descriptors, strategy=SelectionStrategy.PRIORITY)
    assert decision.selected_provider == "gemini"
    assert len(decision.ranked_candidates) == 2
    assert "unreferenced_extra" not in [c.provider_name for c in decision.ranked_candidates]
    assert "unreferenced_extra" not in [e.provider_name for e in decision.excluded_candidates]


def test_concise_and_truthful_reason_generation() -> None:
    selector = DeterministicProviderSelector()

    # Case A: Degraded under local-first (no healthy available)
    dec_deg = selector.select(
        ["p_loc"],
        [_make_health("p_loc", InferenceHealthStatus.DEGRADED)],
        descriptors=[ProviderDescriptor("p_loc", ProviderLocality.LOCAL)],
        strategy=SelectionStrategy.LOCAL_FIRST,
    )
    assert dec_deg.reason == "Selected provider 'p_loc' (degraded; no healthy providers available) by local-first strategy"

    # Case B: Local-first when only cloud provider is available
    dec_cloud_fallback = selector.select(
        ["p_cld"],
        [_make_health("p_cld", InferenceHealthStatus.HEALTHY)],
        descriptors=[ProviderDescriptor("p_cld", ProviderLocality.CLOUD)],
        strategy=SelectionStrategy.LOCAL_FIRST,
    )
    assert dec_cloud_fallback.reason == "Selected provider 'p_cld' (healthy) by priority under local-first strategy (no local candidate in selected health tier)"

    # Case C: Cloud-first when only local provider is available
    dec_local_fallback = selector.select(
        ["p_loc"],
        [_make_health("p_loc", InferenceHealthStatus.HEALTHY)],
        descriptors=[ProviderDescriptor("p_loc", ProviderLocality.LOCAL)],
        strategy=SelectionStrategy.CLOUD_FIRST,
    )
    assert dec_local_fallback.reason == "Selected provider 'p_loc' (healthy) by priority under cloud-first strategy (no cloud candidate in selected health tier)"


@pytest.mark.parametrize(
    (
        "strategy",
        "degraded_name",
        "degraded_locality",
        "healthy_name",
        "healthy_locality",
        "expected_reason",
    ),
    [
        (
            SelectionStrategy.LOCAL_FIRST,
            "p_local",
            ProviderLocality.LOCAL,
            "p_cloud",
            ProviderLocality.CLOUD,
            "Selected provider 'p_cloud' (healthy) by priority under local-first strategy (no local candidate in selected health tier)",
        ),
        (
            SelectionStrategy.CLOUD_FIRST,
            "p_cloud",
            ProviderLocality.CLOUD,
            "p_local",
            ProviderLocality.LOCAL,
            "Selected provider 'p_local' (healthy) by priority under cloud-first strategy (no cloud candidate in selected health tier)",
        ),
    ],
)
def test_reason_truthful_when_preferred_locality_is_degraded(
    strategy: SelectionStrategy,
    degraded_name: str,
    degraded_locality: ProviderLocality,
    healthy_name: str,
    healthy_locality: ProviderLocality,
    expected_reason: str,
) -> None:
    selector = DeterministicProviderSelector()
    preferences = [degraded_name, healthy_name]
    snapshots = [
        _make_health(degraded_name, InferenceHealthStatus.DEGRADED),
        _make_health(healthy_name, InferenceHealthStatus.HEALTHY),
    ]
    descriptors = [
        ProviderDescriptor(degraded_name, degraded_locality),
        ProviderDescriptor(healthy_name, healthy_locality),
    ]

    decision = selector.select(
        preferences,
        snapshots,
        descriptors=descriptors,
        strategy=strategy,
    )

    assert decision.selected_provider == healthy_name
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert decision.reason == expected_reason
    assert "no local providers available" not in decision.reason
    assert "no cloud providers available" not in decision.reason


@pytest.mark.parametrize(
    ("descriptor_factory", "expected_err", "match"),
    [
        (lambda: ProviderDescriptor("", ProviderLocality.LOCAL), ValueError, "provider_name must be a non-empty string"),
        (lambda: ProviderDescriptor("p1", "local"), TypeError, "locality must be a ProviderLocality"),  # type: ignore
        (lambda: ProviderDescriptor("p1", ProviderLocality.LOCAL, "free"), TypeError, "cost_tier must be a CostTier"),  # type: ignore
    ],
)
def test_invalid_descriptor_construction_rejected(
    descriptor_factory: object,
    expected_err: type[Exception],
    match: str,
) -> None:
    with pytest.raises(expected_err, match=match):
        descriptor_factory()  # type: ignore


def test_invalid_descriptor_inputs_rejected() -> None:
    selector = DeterministicProviderSelector()

    # Invalid strategy type
    with pytest.raises(TypeError, match="strategy must be a SelectionStrategy"):
        selector.select(["p1"], [_make_health("p1", InferenceHealthStatus.HEALTHY)], strategy="local_first")  # type: ignore

    # Descriptors must be sequence
    with pytest.raises(TypeError, match="descriptors must be a sequence of ProviderDescriptor"):
        selector.select(
            ["p1"],
            [_make_health("p1", InferenceHealthStatus.HEALTHY)],
            descriptors="not-a-list",  # type: ignore
        )

    # Descriptor items must be ProviderDescriptor
    with pytest.raises(TypeError, match="descriptors items must be ProviderDescriptor instances"):
        selector.select(
            ["p1"],
            [_make_health("p1", InferenceHealthStatus.HEALTHY)],
            descriptors=[{"provider_name": "p1"}],  # type: ignore
        )

    # Duplicate descriptors for same provider name
    with pytest.raises(ValueError, match="Duplicate provider descriptor provided for 'p1'"):
        selector.select(
            ["p1"],
            [_make_health("p1", InferenceHealthStatus.HEALTHY)],
            descriptors=[
                ProviderDescriptor("p1", ProviderLocality.LOCAL),
                ProviderDescriptor("P1", ProviderLocality.CLOUD),
            ],
        )


def test_supports_arbitrary_provider_names() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["delta", "gamma", "beta", "alpha"]
    snapshots = [
        _make_health("delta", InferenceHealthStatus.HEALTHY),
        _make_health("gamma", InferenceHealthStatus.HEALTHY),
        _make_health("beta", InferenceHealthStatus.HEALTHY),
        _make_health("alpha", InferenceHealthStatus.HEALTHY),
    ]
    descriptors = [
        ProviderDescriptor("delta", ProviderLocality.CLOUD, CostTier.STANDARD),
        ProviderDescriptor("gamma", ProviderLocality.CLOUD, CostTier.LOW),
        ProviderDescriptor("beta", ProviderLocality.LOCAL, CostTier.STANDARD),
        ProviderDescriptor("alpha", ProviderLocality.LOCAL, CostTier.FREE),
    ]

    # Under LOCAL_FIRST: beta (LOCAL, priority 2) vs alpha (LOCAL, priority 3) -> beta wins on priority
    dec_local = selector.select(preferences, snapshots, descriptors=descriptors, strategy=SelectionStrategy.LOCAL_FIRST)
    assert dec_local.selected_provider == "beta"

    # Under COST_FIRST: alpha (FREE) beats gamma (LOW) beats delta/beta (STANDARD)
    dec_cost = selector.select(preferences, snapshots, descriptors=descriptors, strategy=SelectionStrategy.COST_FIRST)
    assert dec_cost.selected_provider == "alpha"
    assert [c.provider_name for c in dec_cost.ranked_candidates] == ["alpha", "gamma", "delta", "beta"]

    # Under CLOUD_FIRST: delta (CLOUD, priority 0) beats gamma (CLOUD, priority 1) on priority
    dec_cloud = selector.select(preferences, snapshots, descriptors=descriptors, strategy=SelectionStrategy.CLOUD_FIRST)
    assert dec_cloud.selected_provider == "delta"
