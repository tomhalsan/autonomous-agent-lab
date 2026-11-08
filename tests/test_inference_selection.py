"""Unit tests for pure deterministic inference provider selection policy."""

from dataclasses import FrozenInstanceError

import pytest
from agent_contracts import InferenceHealthStatus, InferenceProviderHealth
from agent_policy import (
    DeterministicProviderSelector,
    ExcludedProviderCandidate,
    ProviderSelectionDecision,
    RankedProviderCandidate,
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
    assert "Selected provider 'gemini' (healthy)" in decision.reason
    assert "preferred over higher-priority degraded candidates" in decision.reason
    assert "matching highest configured priority" not in decision.reason
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
    assert "no healthy providers available" in decision.reason


def test_degraded_selected_when_higher_priority_unavailable() -> None:
    selector = DeterministicProviderSelector()
    snapshots = [
        _make_health("p1", InferenceHealthStatus.UNAVAILABLE, reason="endpoint offline"),
        _make_health("p2", InferenceHealthStatus.DEGRADED),
    ]

    decision = selector.select(["p1", "p2"], snapshots)
    assert decision.selected_provider == "p2"
    assert decision.status == InferenceHealthStatus.DEGRADED
    assert "higher-priority providers are ineligible" in decision.reason
    assert len(decision.ranked_candidates) == 1
    assert decision.ranked_candidates[0].provider_name == "p2"
    assert len(decision.excluded_candidates) == 1
    assert decision.excluded_candidates[0].provider_name == "p1"
    assert decision.excluded_candidates[0].health_status == InferenceHealthStatus.UNAVAILABLE


# --- 1b. Transparent Reason Generation Tests ---


def test_reason_highest_priority_healthy_selected() -> None:
    selector = DeterministicProviderSelector()
    preferences = ["gemini", "cloudflare"]
    snapshots = [
        _make_health("gemini", InferenceHealthStatus.HEALTHY),
        _make_health("cloudflare", InferenceHealthStatus.HEALTHY),
    ]

    decision = selector.select(preferences, snapshots)

    assert decision.selected_provider == "gemini"
    assert decision.status == InferenceHealthStatus.HEALTHY
    assert decision.reason == "Selected provider 'gemini' (healthy) matching highest configured priority"


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
    assert "Selected provider 'gemini' (healthy)" in decision.reason
    assert "preferred over higher-priority degraded candidates" in decision.reason
    assert "matching highest configured priority" not in decision.reason


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
    assert "Selected provider 'gemini' (healthy)" in decision.reason
    assert "higher-priority providers are ineligible" in decision.reason
    assert "matching highest configured priority" not in decision.reason


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
        == "Selected provider 'cloudflare' (degraded) as highest configured priority; no healthy providers available"
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
        == "Selected provider 'gemini' (degraded); higher-priority providers are ineligible and no healthy providers available"
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
