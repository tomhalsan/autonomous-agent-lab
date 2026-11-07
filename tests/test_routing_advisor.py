"""Tests for RoutingAdvisor protocol, AdvisorRecommendation contract, and hybrid routing policy."""

from pathlib import Path

import pytest
from agent_contracts import RoutingProvenance, TaskRequested
from orchestrator.application.ports import (
    AssessingCapabilityRouter,
    CapabilityRouter,
    RoutingAdvisor,
    TaskDispatcher,
)
from orchestrator.application.use_cases import SubmitTask
from orchestrator.domain import (
    AdvisorError,
    AdvisorExecutionError,
    AdvisorRecommendation,
    AdvisorTimeoutError,
    AdvisorUnavailableError,
    AgentCapability,
    CapabilityUnavailableError,
    DeterministicCapabilityRouter,
    RoutingAssessment,
    RoutingClarity,
    RoutingSignal,
    RoutingSignalType,
    SignalStrength,
)

from tests.test_application import FakeTaskQueue
from tests.test_routing_decision import FakeRoutingRecorder
from tests.test_runtime_dispatch import FakeCandidateFinder, make_candidate


class RecordingAdvisor(RoutingAdvisor):
    """Advisor test double recording all invocations and returning configured recommendation."""

    def __init__(
        self,
        recommendation: AdvisorRecommendation | None = None,
        error_to_raise: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, RoutingAssessment]] = []
        self._recommendation = recommendation
        self._error_to_raise = error_to_raise

    async def advise(
        self, prompt: str, assessment: RoutingAssessment
    ) -> AdvisorRecommendation:
        self.calls.append((prompt, assessment))
        if self._error_to_raise is not None:
            raise self._error_to_raise
        if self._recommendation is not None:
            return self._recommendation
        return AdvisorRecommendation(
            recommended_capability=assessment.provisional_capability,
            reason="Advisor confirms provisional capability",
        )


class StubAssessingRouter(AssessingCapabilityRouter):
    """Router test double allowing exact control over assessment."""

    def __init__(self, assessment: RoutingAssessment) -> None:
        self._assessment = assessment

    def assess(self, prompt: str) -> RoutingAssessment:
        return self._assessment

    def route(self, prompt: str) -> AgentCapability:
        return self._assessment.provisional_capability


class LegacyRouteOnlyRouter(CapabilityRouter):
    """Router test double implementing only route(), not assess()."""

    def __init__(self, capability: AgentCapability) -> None:
        self._capability = capability

    def route(self, prompt: str) -> AgentCapability:
        return self._capability


# ==============================================================================
# Contract & Validation Tests
# ==============================================================================


def test_advisor_recommendation_contract_validation() -> None:
    rec = AdvisorRecommendation(
        recommended_capability=AgentCapability.RESEARCH,
        reason="Requires real-time market data retrieval from external sources",
    )
    assert rec.recommended_capability == AgentCapability.RESEARCH
    assert "market data" in rec.reason

    with pytest.raises(TypeError, match="recommended_capability must be an AgentCapability"):
        AdvisorRecommendation(
            recommended_capability="research",  # type: ignore[arg-type]
            reason="Valid reason",
        )

    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        AdvisorRecommendation(
            recommended_capability=AgentCapability.GENERAL,
            reason="   ",
        )


def test_advisor_recommendation_max_length_validation() -> None:
    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        AdvisorRecommendation(
            recommended_capability=AgentCapability.GENERAL,
            reason="x" * 4097,
        )


@pytest.mark.parametrize(
    "legitimate_phrase",
    [
        "Dispatch to research is warranted because fresh web data is needed",
        "Route to research agent recommended due to external documentation lookup",
        "The worker agent capabilities are insufficient for real-time CVE search",
        "Target execution requires specialized tools",
        "We should dispatch to general queue as all inputs are inline",
    ],
)
def test_advisor_recommendation_accepts_explanatory_prose_with_routing_concepts(
    legitimate_phrase: str,
) -> None:
    rec = AdvisorRecommendation(
        recommended_capability=AgentCapability.RESEARCH,
        reason=legitimate_phrase,
    )
    assert rec.reason == legitimate_phrase


def test_routing_provenance_allowed_values_and_type_validation() -> None:
    # Valid instance
    prov = RoutingProvenance(
        is_explicit_override=False,
        assessment_clarity="ambiguous",
        provisional_capability="general",
        advisor_consulted=True,
        advisor_verdict="changed",
        fallback_used=False,
        final_selection_reason="Advisor clarified external info need",
        deterministic_assessment_reason="Weak verb alone",
        deterministic_signal_types=("weak_research_verb",),
        advisor_reason="Requires real-time pricing",
    )
    assert prov.assessment_clarity == "ambiguous"
    assert prov.advisor_verdict == "changed"

    # Invalid clarity
    with pytest.raises(ValueError, match="Invalid assessment_clarity"):
        RoutingProvenance(
            final_selection_reason="reason",
            assessment_clarity="unsure",  # type: ignore[arg-type]
        )

    # Invalid provisional capability
    with pytest.raises(ValueError, match="Invalid provisional_capability"):
        RoutingProvenance(
            final_selection_reason="reason",
            provisional_capability="specialized",  # type: ignore[arg-type]
        )

    # Invalid advisor verdict
    with pytest.raises(ValueError, match="Invalid advisor_verdict"):
        RoutingProvenance(
            final_selection_reason="reason",
            advisor_consulted=True,
            advisor_verdict="vetoed",  # type: ignore[arg-type]
        )

    # Non-boolean bool fields
    for field in ("is_explicit_override", "advisor_consulted", "fallback_used"):
        with pytest.raises(TypeError, match="must be a boolean"):
            RoutingProvenance(
                final_selection_reason="reason",
                **{field: "false"},  # type: ignore[arg-type]
            )

    # Empty final_selection_reason
    with pytest.raises(ValueError, match="final_selection_reason must be a non-empty string"):
        RoutingProvenance(final_selection_reason="   ")


def test_routing_provenance_invariant_violations() -> None:
    # 1. advisor_verdict requires advisor_consulted=True
    with pytest.raises(ValueError, match="advisor_verdict requires advisor_consulted=True"):
        RoutingProvenance(
            final_selection_reason="reason",
            advisor_consulted=False,
            advisor_verdict="agreed",
        )

    # 2. "agreed" or "changed" requires fallback_used=False
    with pytest.raises(ValueError, match="requires fallback_used=False"):
        RoutingProvenance(
            final_selection_reason="reason",
            advisor_consulted=True,
            advisor_verdict="agreed",
            fallback_used=True,
        )

    with pytest.raises(ValueError, match="requires fallback_used=False"):
        RoutingProvenance(
            final_selection_reason="reason",
            advisor_consulted=True,
            advisor_verdict="changed",
            fallback_used=True,
        )

    # 3. "unavailable", "timeout", "failed" requires fallback_used=True
    for v in ("unavailable", "timeout", "failed"):
        with pytest.raises(ValueError, match="requires fallback_used=True"):
            RoutingProvenance(
                final_selection_reason="reason",
                advisor_consulted=True,
                advisor_verdict=v,
                fallback_used=False,
            )

    # 4. fallback_used=True requires advisor_consulted=True
    with pytest.raises(ValueError, match="fallback_used=True requires advisor_consulted=True"):
        RoutingProvenance(
            final_selection_reason="reason",
            advisor_consulted=False,
            fallback_used=True,
        )

    # 5. explicit override requires advisor_consulted=False, fallback_used=False, advisor_verdict=None
    with pytest.raises(ValueError, match="explicit override requires advisor_consulted=False"):
        RoutingProvenance(
            final_selection_reason="reason",
            is_explicit_override=True,
            advisor_consulted=True,
        )

    with pytest.raises(ValueError, match="explicit override requires fallback_used=False"):
        RoutingProvenance(
            final_selection_reason="reason",
            is_explicit_override=True,
            fallback_used=True,
        )

    with pytest.raises(ValueError, match="explicit override requires advisor_verdict=None"):
        RoutingProvenance(
            final_selection_reason="reason",
            is_explicit_override=True,
            advisor_verdict="agreed",
        )


def test_routing_provenance_from_dict_strict_booleans() -> None:
    # "false" string must not coerce to True
    with pytest.raises(TypeError, match="must be a boolean, got str"):
        RoutingProvenance.from_dict({
            "is_explicit_override": "false",
            "final_selection_reason": "ok",
        })

    with pytest.raises(TypeError, match="must be a boolean, got str"):
        RoutingProvenance.from_dict({
            "advisor_consulted": "true",
            "final_selection_reason": "ok",
        })

    with pytest.raises(TypeError, match="must be a boolean, got int"):
        RoutingProvenance.from_dict({
            "fallback_used": 1,
            "final_selection_reason": "ok",
        })


def test_routing_provenance_round_trip_serialization() -> None:
    prov = RoutingProvenance(
        is_explicit_override=False,
        assessment_clarity="ambiguous",
        provisional_capability="research",
        advisor_consulted=True,
        advisor_verdict="agreed",
        fallback_used=False,
        final_selection_reason="Advisor agreed with external search need",
        deterministic_assessment_reason="Borderline search intent",
        deterministic_signal_types=("weak_research_verb", "changing_external_info"),
        advisor_reason="Confirmed pricing research required",
    )
    d = prov.to_dict()
    assert isinstance(d["deterministic_signal_types"], list)
    restored = RoutingProvenance.from_dict(d)
    assert restored == prov


# ==============================================================================
# Authority & Policy Tests (Scenarios 1-19)
# ==============================================================================


@pytest.mark.anyio
async def test_1_explicit_override_never_calls_advisor() -> None:
    advisor = RecordingAdvisor()
    queue = FakeTaskQueue()
    use_case = SubmitTask(queue=queue, advisor=advisor)

    # Even on ambiguous prompt, explicit override must bypass router and advisor
    await use_case.execute(
        "Tell me about quantum computing advancements",
        capability=AgentCapability.GENERAL,
    )
    assert len(advisor.calls) == 0


@pytest.mark.anyio
async def test_2_clear_general_never_calls_advisor() -> None:
    advisor = RecordingAdvisor()
    queue = FakeTaskQueue()
    use_case = SubmitTask(queue=queue, advisor=advisor)

    await use_case.execute("Calculate the sum of [1, 2, 3, 4, 5]")
    assert len(advisor.calls) == 0


@pytest.mark.anyio
async def test_3_clear_research_never_calls_advisor() -> None:
    advisor = RecordingAdvisor()
    queue = FakeTaskQueue()
    use_case = SubmitTask(queue=queue, advisor=advisor)

    await use_case.execute("Search for the latest news and updates on Python 3.13 release")
    assert len(advisor.calls) == 0


@pytest.mark.anyio
async def test_4_ambiguous_assessment_calls_advisor_once_when_configured() -> None:
    advisor = RecordingAdvisor()
    queue = FakeTaskQueue()
    use_case = SubmitTask(queue=queue, advisor=advisor)

    prompt = "Tell me about quantum computing advancements"
    await use_case.execute(prompt)
    assert len(advisor.calls) == 1
    called_prompt, assessment = advisor.calls[0]
    assert called_prompt == prompt
    assert assessment.clarity == RoutingClarity.AMBIGUOUS


@pytest.mark.anyio
async def test_5_no_configured_advisor_preserves_provisional_capability() -> None:
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()
    use_case = SubmitTask(queue=queue, routing_recorder=recorder, advisor=None)

    prompt = "Tell me about quantum computing advancements"
    await use_case.execute(prompt)

    assert len(recorder.recorded) == 1
    decision = recorder.recorded[0]
    prov = RoutingProvenance.from_dict(decision.provenance)
    assert prov.advisor_consulted is False
    assert prov.provisional_capability == AgentCapability.RESEARCH.value
    assert decision.required_capabilities == [AgentCapability.RESEARCH.value]


@pytest.mark.anyio
async def test_6_advisor_recommendation_can_change_general_to_research() -> None:
    ambiguous_general_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.GENERAL,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(
            RoutingSignal(
                signal_type=RoutingSignalType.WEAK_RESEARCH_VERB,
                target_capability=AgentCapability.GENERAL,
                strength=SignalStrength.WEAK,
                evidence="investigate",
            ),
        ),
        reason="Ambiguous conceptual investigation",
    )
    router = StubAssessingRouter(ambiguous_general_assessment)
    advisor = RecordingAdvisor(
        recommendation=AdvisorRecommendation(
            recommended_capability=AgentCapability.RESEARCH,
            reason="Clarified that prompt requires recent external sources",
        )
    )
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()
    use_case = SubmitTask(
        router=router, queue=queue, routing_recorder=recorder, advisor=advisor
    )

    await use_case.execute("Investigate the architecture trade-offs")

    assert len(recorder.recorded) == 1
    decision = recorder.recorded[0]
    prov = RoutingProvenance.from_dict(decision.provenance)
    assert prov.advisor_consulted is True
    assert prov.advisor_verdict == "changed"
    assert prov.provisional_capability == AgentCapability.GENERAL.value
    assert decision.required_capabilities == [AgentCapability.RESEARCH.value]


@pytest.mark.anyio
async def test_7_advisor_recommendation_can_change_research_to_general() -> None:
    ambiguous_research_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.RESEARCH,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(
            RoutingSignal(
                signal_type=RoutingSignalType.FRESHNESS_OR_CURRENT_INFO,
                target_capability=AgentCapability.RESEARCH,
                strength=SignalStrength.WEAK,
                evidence="recent",
            ),
        ),
        reason="Ambiguous recent wording",
    )
    router = StubAssessingRouter(ambiguous_research_assessment)
    advisor = RecordingAdvisor(
        recommendation=AdvisorRecommendation(
            recommended_capability=AgentCapability.GENERAL,
            reason="Prompt can be fully satisfied by general reasoning",
        )
    )
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()
    use_case = SubmitTask(
        router=router, queue=queue, routing_recorder=recorder, advisor=advisor
    )

    await use_case.execute("Summarize recent thoughts on clean architecture")

    assert len(recorder.recorded) == 1
    decision = recorder.recorded[0]
    prov = RoutingProvenance.from_dict(decision.provenance)
    assert prov.advisor_consulted is True
    assert prov.advisor_verdict == "changed"
    assert prov.provisional_capability == AgentCapability.RESEARCH.value
    assert decision.required_capabilities == [AgentCapability.GENERAL.value]


@pytest.mark.anyio
async def test_8_advisor_agreement_preserves_capability() -> None:
    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.RESEARCH,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Ambiguous borderline prompt",
    )
    router = StubAssessingRouter(ambiguous_assessment)
    advisor = RecordingAdvisor(
        recommendation=AdvisorRecommendation(
            recommended_capability=AgentCapability.RESEARCH,
            reason="Advisor agrees external research is needed",
        )
    )
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()
    use_case = SubmitTask(
        router=router, queue=queue, routing_recorder=recorder, advisor=advisor
    )

    await use_case.execute("Some borderline research inquiry")

    decision = recorder.recorded[0]
    prov = RoutingProvenance.from_dict(decision.provenance)
    assert prov.advisor_consulted is True
    assert prov.advisor_verdict == "agreed"
    assert prov.fallback_used is False
    assert decision.required_capabilities == [AgentCapability.RESEARCH.value]


@pytest.mark.anyio
async def test_9_advisor_unavailable_falls_back_to_provisional() -> None:
    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.RESEARCH,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Borderline prompt",
    )
    router = StubAssessingRouter(ambiguous_assessment)
    advisor = RecordingAdvisor(
        error_to_raise=AdvisorUnavailableError("Advisor endpoint connection refused")
    )
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()
    use_case = SubmitTask(
        router=router, queue=queue, routing_recorder=recorder, advisor=advisor
    )

    await use_case.execute("Some borderline inquiry")

    decision = recorder.recorded[0]
    prov = RoutingProvenance.from_dict(decision.provenance)
    assert prov.advisor_consulted is True
    assert prov.advisor_verdict == "unavailable"
    assert prov.fallback_used is True
    assert decision.required_capabilities == [AgentCapability.RESEARCH.value]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error, expected_verdict",
    [
        (AdvisorTimeoutError("Advisor request deadline exceeded"), "timeout"),
        (TimeoutError("Standard library asyncio timeout"), "timeout"),
        (AdvisorExecutionError("Advisor model internal error"), "failed"),
        (AdvisorError("Generic advisor error"), "failed"),
    ],
)
async def test_10_advisor_timeout_and_failure_fall_back_to_provisional(
    error: Exception, expected_verdict: str
) -> None:
    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.RESEARCH,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Borderline prompt",
    )
    router = StubAssessingRouter(ambiguous_assessment)
    advisor = RecordingAdvisor(error_to_raise=error)
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()
    use_case = SubmitTask(
        router=router, queue=queue, routing_recorder=recorder, advisor=advisor
    )

    await use_case.execute("Some borderline inquiry")

    decision = recorder.recorded[0]
    prov = RoutingProvenance.from_dict(decision.provenance)
    assert prov.advisor_consulted is True
    assert prov.advisor_verdict == expected_verdict
    assert prov.fallback_used is True
    assert decision.required_capabilities == [AgentCapability.RESEARCH.value]


@pytest.mark.anyio
async def test_11_invalid_recommendation_raises_type_error() -> None:
    class BrokenAdvisor(RoutingAdvisor):
        async def advise(
            self, prompt: str, assessment: RoutingAssessment
        ) -> AdvisorRecommendation:
            return {"recommended_capability": "research"}  # type: ignore[return-value]

    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.RESEARCH,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Borderline prompt",
    )
    router = StubAssessingRouter(ambiguous_assessment)
    use_case = SubmitTask(
        router=router, queue=FakeTaskQueue(), advisor=BrokenAdvisor()
    )

    with pytest.raises(TypeError, match="must return AdvisorRecommendation"):
        await use_case.execute("Some borderline inquiry")


@pytest.mark.anyio
async def test_12_candidate_discovery_receives_final_resolved_capability() -> None:
    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.GENERAL,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Ambiguous prompt",
    )
    router = StubAssessingRouter(ambiguous_assessment)
    advisor = RecordingAdvisor(
        recommendation=AdvisorRecommendation(
            recommended_capability=AgentCapability.RESEARCH,
            reason="Advisor determines external research is required",
        )
    )
    candidate = make_candidate("research-1", capability="research", dispatch_target="agent.tasks.research")
    finder = FakeCandidateFinder(
        candidates_by_capability={AgentCapability.RESEARCH.value: [candidate]}
    )
    queue = FakeTaskQueue()
    use_case = SubmitTask(
        router=router,
        candidate_finder=finder,
        queue=queue,
        advisor=advisor,
    )

    await use_case.execute("Ambiguous prompt requiring research")
    assert finder.calls == [AgentCapability.RESEARCH.value]


@pytest.mark.anyio
async def test_13_availability_check_uses_final_resolved_capability() -> None:
    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.GENERAL,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Ambiguous prompt",
    )
    router = StubAssessingRouter(ambiguous_assessment)
    advisor = RecordingAdvisor(
        recommendation=AdvisorRecommendation(
            recommended_capability=AgentCapability.RESEARCH,
            reason="Advisor recommends research",
        )
    )
    # Only GENERAL candidates are registered
    general_candidate = make_candidate("worker-1", capability="general", dispatch_target="agent.tasks.general")
    finder = FakeCandidateFinder(
        candidates_by_capability={AgentCapability.GENERAL.value: [general_candidate]}
    )
    recorder = FakeRoutingRecorder()
    use_case = SubmitTask(
        router=router,
        candidate_finder=finder,
        routing_recorder=recorder,
        advisor=advisor,
    )

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await use_case.execute("Ambiguous prompt requiring research")

    assert exc_info.value.capability == AgentCapability.RESEARCH
    assert len(recorder.recorded) == 1
    decision = recorder.recorded[0]
    assert decision.required_capabilities == [AgentCapability.RESEARCH.value]
    prov = RoutingProvenance.from_dict(decision.provenance)
    assert prov.advisor_verdict == "changed"


@pytest.mark.anyio
async def test_14_dispatch_behavior_remains_unchanged() -> None:
    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.GENERAL,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Ambiguous prompt",
    )
    router = StubAssessingRouter(ambiguous_assessment)
    advisor = RecordingAdvisor(
        recommendation=AdvisorRecommendation(
            recommended_capability=AgentCapability.RESEARCH,
            reason="Advisor resolves to research",
        )
    )
    candidate = make_candidate(
        "res-1", capability="research", dispatch_target="agent.tasks.research"
    )
    finder = FakeCandidateFinder(
        candidates_by_capability={AgentCapability.RESEARCH.value: [candidate]}
    )
    queue = FakeTaskQueue()
    use_case = SubmitTask(
        router=router,
        candidate_finder=finder,
        queue=queue,
        advisor=advisor,
    )

    task = await use_case.execute("Ambiguous prompt")
    assert len(queue.enqueued) == 1
    dispatched_task, cap = queue.enqueued[0]
    assert dispatched_task.task_id == task.task_id
    assert cap == AgentCapability.RESEARCH
    assert len(queue.dispatches) == 1
    _, target = queue.dispatches[0]
    assert target == "agent.tasks.research"


@pytest.mark.anyio
async def test_15_explicit_override_remains_higher_authority_than_advisor() -> None:
    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.RESEARCH,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Ambiguous prompt",
    )
    router = StubAssessingRouter(ambiguous_assessment)
    advisor = RecordingAdvisor(
        recommendation=AdvisorRecommendation(
            recommended_capability=AgentCapability.RESEARCH,
            reason="Advisor says research",
        )
    )
    general_candidate = make_candidate("worker-1", capability="general", dispatch_target="agent.tasks.general")
    finder = FakeCandidateFinder(
        candidates_by_capability={AgentCapability.GENERAL.value: [general_candidate]}
    )
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()
    use_case = SubmitTask(
        router=router,
        candidate_finder=finder,
        queue=queue,
        routing_recorder=recorder,
        advisor=advisor,
    )

    # Caller explicitly forces GENERAL
    await use_case.execute(
        "Ambiguous prompt", capability=AgentCapability.GENERAL
    )

    assert len(advisor.calls) == 0
    assert finder.calls == [AgentCapability.GENERAL.value]
    decision = recorder.recorded[0]
    prov = RoutingProvenance.from_dict(decision.provenance)
    assert prov.is_explicit_override is True
    assert prov.advisor_consulted is False


@pytest.mark.anyio
async def test_16_route_only_legacy_test_doubles_remain_compatible() -> None:
    legacy_router = LegacyRouteOnlyRouter(AgentCapability.GENERAL)
    advisor = RecordingAdvisor()
    queue = FakeTaskQueue()
    recorder = FakeRoutingRecorder()
    use_case = SubmitTask(
        router=legacy_router,
        queue=queue,
        routing_recorder=recorder,
        advisor=advisor,
    )

    await use_case.execute("Any prompt")

    assert len(advisor.calls) == 0
    decision = recorder.recorded[0]
    prov = RoutingProvenance.from_dict(decision.provenance)
    assert prov.advisor_consulted is False
    assert prov.assessment_clarity is None
    assert prov.provisional_capability == AgentCapability.GENERAL.value


@pytest.mark.anyio
async def test_17_advisor_never_receives_clear_assessments() -> None:
    class StrictAdvisor(RoutingAdvisor):
        def __init__(self) -> None:
            self.invocations: int = 0

        async def advise(
            self, prompt: str, assessment: RoutingAssessment
        ) -> AdvisorRecommendation:
            if assessment.clarity == RoutingClarity.CLEAR:
                raise AssertionError("Strict invariant violated: CLEAR assessment sent to advisor")
            self.invocations += 1
            return AdvisorRecommendation(
                recommended_capability=assessment.provisional_capability,
                reason="Advising ambiguous assessment",
            )

    advisor = StrictAdvisor()
    router = DeterministicCapabilityRouter()
    queue = FakeTaskQueue()
    use_case = SubmitTask(router=router, queue=queue, advisor=advisor)

    clear_prompts = [
        "Calculate the sum of [1, 2, 3]",
        "Rewrite this function using list comprehensions: def foo(): ...",
        "Search for the latest CVE security advisories on OpenSSL",
        "What is the current version of Python released today?",
    ]
    for prompt in clear_prompts:
        await use_case.execute(prompt)

    assert advisor.invocations == 0


@pytest.mark.anyio
async def test_18_prompt_and_complete_routing_assessment_passed_unchanged() -> None:
    expected_signal = RoutingSignal(
        signal_type=RoutingSignalType.WEAK_RESEARCH_VERB,
        target_capability=AgentCapability.RESEARCH,
        strength=SignalStrength.WEAK,
        evidence="study",
    )
    assessment = RoutingAssessment(
        provisional_capability=AgentCapability.RESEARCH,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(expected_signal,),
        reason="Borderline study verb",
    )
    router = StubAssessingRouter(assessment)
    advisor = RecordingAdvisor()
    queue = FakeTaskQueue()
    use_case = SubmitTask(router=router, queue=queue, advisor=advisor)

    test_prompt = "Study the behavior of sorting algorithms"
    await use_case.execute(test_prompt)

    assert len(advisor.calls) == 1
    prompt_received, assessment_received = advisor.calls[0]
    assert prompt_received == test_prompt
    assert assessment_received == assessment
    assert assessment_received.signals == (expected_signal,)
    assert assessment_received.clarity == RoutingClarity.AMBIGUOUS
    assert assessment_received.provisional_capability == AgentCapability.RESEARCH


@pytest.mark.anyio
async def test_19_provenance_distinguishes_deterministic_advised_fallback_and_override() -> None:
    router = DeterministicCapabilityRouter()
    recorder = FakeRoutingRecorder()

    # 1. Explicit override
    use_case_override = SubmitTask(
        router=router, queue=FakeTaskQueue(), routing_recorder=recorder
    )
    await use_case_override.execute(
        "Any prompt", capability=AgentCapability.RESEARCH
    )
    override_prov = RoutingProvenance.from_dict(recorder.recorded[-1].provenance)
    assert override_prov.is_explicit_override is True
    assert override_prov.advisor_consulted is False

    # 2. Clear deterministic
    use_case_clear = SubmitTask(
        router=router, queue=FakeTaskQueue(), routing_recorder=recorder
    )
    await use_case_clear.execute("Calculate 1 + 1")
    clear_prov = RoutingProvenance.from_dict(recorder.recorded[-1].provenance)
    assert clear_prov.is_explicit_override is False
    assert clear_prov.assessment_clarity == "clear"
    assert clear_prov.advisor_consulted is False
    assert clear_prov.fallback_used is False

    # 3. Advised changed
    expected_signal = RoutingSignal(
        signal_type=RoutingSignalType.WEAK_RESEARCH_VERB,
        target_capability=AgentCapability.RESEARCH,
        strength=SignalStrength.WEAK,
        evidence="explore",
    )
    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.GENERAL,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(expected_signal,),
        reason="Weak explore verb",
    )
    stub_router = StubAssessingRouter(ambiguous_assessment)
    advisor_changed = RecordingAdvisor(
        recommendation=AdvisorRecommendation(
            recommended_capability=AgentCapability.RESEARCH,
            reason="Advisor resolved to research",
        )
    )
    use_case_changed = SubmitTask(
        router=stub_router,
        queue=FakeTaskQueue(),
        routing_recorder=recorder,
        advisor=advisor_changed,
    )
    await use_case_changed.execute("Ambiguous query")
    changed_prov = RoutingProvenance.from_dict(recorder.recorded[-1].provenance)
    assert changed_prov.assessment_clarity == "ambiguous"
    assert changed_prov.advisor_consulted is True
    assert changed_prov.advisor_verdict == "changed"
    assert changed_prov.fallback_used is False
    assert changed_prov.deterministic_assessment_reason == "Weak explore verb"
    assert changed_prov.deterministic_signal_types == ("weak_research_verb",)
    assert changed_prov.advisor_reason == "Advisor resolved to research"

    # 4. Advised fallback
    advisor_fallback = RecordingAdvisor(
        error_to_raise=AdvisorTimeoutError("Timed out")
    )
    use_case_fallback = SubmitTask(
        router=stub_router,
        queue=FakeTaskQueue(),
        routing_recorder=recorder,
        advisor=advisor_fallback,
    )
    await use_case_fallback.execute("Ambiguous query")
    fallback_prov = RoutingProvenance.from_dict(recorder.recorded[-1].provenance)
    assert fallback_prov.advisor_consulted is True
    assert fallback_prov.advisor_verdict == "timeout"
    assert fallback_prov.fallback_used is True
    assert fallback_prov.deterministic_assessment_reason == "Weak explore verb"
    assert fallback_prov.deterministic_signal_types == ("weak_research_verb",)
    assert fallback_prov.advisor_reason == ""


@pytest.mark.anyio
async def test_20_no_candidate_rejection_preserves_provenance() -> None:
    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.GENERAL,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Ambiguous prompt",
    )
    stub_router = StubAssessingRouter(ambiguous_assessment)
    advisor = RecordingAdvisor(
        recommendation=AdvisorRecommendation(
            recommended_capability=AgentCapability.RESEARCH,
            reason="Requires research",
        )
    )
    finder = FakeCandidateFinder(candidates_by_capability={})
    recorder = FakeRoutingRecorder()
    use_case = SubmitTask(
        router=stub_router,
        candidate_finder=finder,
        routing_recorder=recorder,
        advisor=advisor,
    )

    with pytest.raises(CapabilityUnavailableError):
        await use_case.execute("Ambiguous query")

    assert len(recorder.recorded) == 1
    decision = recorder.recorded[0]
    prov = RoutingProvenance.from_dict(decision.provenance)
    assert prov.advisor_consulted is True
    assert prov.advisor_verdict == "changed"
    assert prov.advisor_reason == "Requires research"


@pytest.mark.anyio
async def test_21_dispatch_failure_preserves_provenance() -> None:
    class FailingDispatcher(TaskDispatcher):
        async def dispatch(self, task: TaskRequested, target: str) -> None:
            raise RuntimeError("Transport connection lost during dispatch")

    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.GENERAL,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Ambiguous prompt",
    )
    stub_router = StubAssessingRouter(ambiguous_assessment)
    advisor = RecordingAdvisor(
        recommendation=AdvisorRecommendation(
            recommended_capability=AgentCapability.RESEARCH,
            reason="Requires research",
        )
    )
    candidate = make_candidate("res-1", capability="research", dispatch_target="agent.tasks.research")
    finder = FakeCandidateFinder(
        candidates_by_capability={AgentCapability.RESEARCH.value: [candidate]}
    )
    recorder = FakeRoutingRecorder()
    use_case = SubmitTask(
        router=stub_router,
        candidate_finder=finder,
        dispatcher=FailingDispatcher(),
        routing_recorder=recorder,
        advisor=advisor,
    )

    with pytest.raises(RuntimeError, match="Transport connection lost during dispatch"):
        await use_case.execute("Ambiguous query")

    assert len(recorder.recorded) == 1
    decision = recorder.recorded[0]
    prov = RoutingProvenance.from_dict(decision.provenance)
    assert prov.advisor_consulted is True
    assert prov.advisor_verdict == "changed"
    assert prov.advisor_reason == "Requires research"


def test_22_simulated_advisor_plumbing_cycle_exercises_all_branches() -> None:
    from evaluation.routing.benchmark import SimulatedPlumbingAdvisor

    advisor = SimulatedPlumbingAdvisor(scenario="cycle")
    assessment = RoutingAssessment(
        provisional_capability=AgentCapability.GENERAL,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Ambiguous",
    )

    # 1. Agreement
    rec1 = advisor.advise("Prompt 1", assessment)
    assert rec1.recommended_capability == AgentCapability.GENERAL
    assert advisor.agreement_count == 1

    # 2. Override
    rec2 = advisor.advise("Prompt 2", assessment)
    assert rec2.recommended_capability == AgentCapability.RESEARCH
    assert advisor.override_count == 1

    # 3. Unavailable
    with pytest.raises(AdvisorUnavailableError) as exc_unavail:
        advisor.advise("Prompt 3", assessment)
    advisor.record_fallback(exc_unavail.value)
    assert advisor.unavailable_count == 1
    assert advisor.fallback_count == 1

    # 4. Timeout
    with pytest.raises(AdvisorTimeoutError) as exc_timeout:
        advisor.advise("Prompt 4", assessment)
    advisor.record_fallback(exc_timeout.value)
    assert advisor.timeout_count == 1
    assert advisor.fallback_count == 2

    # 5. Execution Error
    with pytest.raises(AdvisorExecutionError) as exc_error:
        advisor.advise("Prompt 5", assessment)
    advisor.record_fallback(exc_error.value)
    assert advisor.execution_error_count == 1
    assert advisor.fallback_count == 3

    assert advisor.consultation_count == 5
    assert advisor.agreement_count == 1
    assert advisor.override_count == 1
    assert advisor.fallback_count == 3


def test_23_documentation_contains_no_file_urls() -> None:
    docs_dir = Path(__file__).parent.parent / "docs"
    assert docs_dir.exists()

    violating_files: list[str] = []
    for doc_file in docs_dir.glob("**/*.md"):
        content = doc_file.read_text(encoding="utf-8")
        if "file:///" in content:
            violating_files.append(str(doc_file.relative_to(docs_dir.parent)))

    assert not violating_files, f"Found file:/// links in documentation files: {violating_files}"


@pytest.mark.anyio
async def test_24_async_advisor_awaited_directly_rejects_sync_callable() -> None:
    class SyncAdvisor:
        def advise(self, prompt: str, assessment: RoutingAssessment) -> AdvisorRecommendation:
            return AdvisorRecommendation(
                recommended_capability=assessment.provisional_capability,
                reason="Sync return",
            )

    ambiguous_assessment = RoutingAssessment(
        provisional_capability=AgentCapability.RESEARCH,
        clarity=RoutingClarity.AMBIGUOUS,
        signals=(),
        reason="Ambiguous",
    )
    stub_router = StubAssessingRouter(ambiguous_assessment)
    use_case = SubmitTask(
        router=stub_router,
        queue=FakeTaskQueue(),
        advisor=SyncAdvisor(),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="can't be used in 'await' expression|cannot be used in an 'await'"):
        await use_case.execute("Ambiguous prompt")
