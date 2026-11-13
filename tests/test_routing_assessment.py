"""Deterministic tests for Phase B RoutingAssessment, clarity policies, and signal taxonomy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from agent_contracts import TaskRequested
from orchestrator.application.ports import AssessingCapabilityRouter
from orchestrator.application.use_cases import SubmitTask
from orchestrator.domain.assessment import (
    RoutingAssessment,
    RoutingClarity,
    RoutingSignal,
    RoutingSignalType,
    SignalStrength,
)
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.router import DeterministicCapabilityRouter


class FakeTaskQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[TaskRequested, AgentCapability]] = []

    async def enqueue(
        self,
        task: TaskRequested,
        capability: AgentCapability = AgentCapability.GENERAL,
        target: str | None = None,
    ) -> None:
        self.enqueued.append((task, capability))


class TestRoutingDomainContract:
    """Validates structural invariants of RoutingAssessment and RoutingSignal."""

    def test_immutable_assessment_model(self) -> None:
        signal = RoutingSignal(
            signal_type=RoutingSignalType.EXPLICIT_SOURCES_OR_CITATIONS,
            target_capability=AgentCapability.RESEARCH,
            strength=SignalStrength.STRONG,
            evidence="cite sources",
        )
        assessment = RoutingAssessment(
            provisional_capability=AgentCapability.RESEARCH,
            clarity=RoutingClarity.CLEAR,
            signals=(signal,),
            reason="Explicit sources requested",
        )

        with pytest.raises(FrozenInstanceError):
            assessment.clarity = RoutingClarity.AMBIGUOUS  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            assessment.reason = "Modified"  # type: ignore[misc]

        assert isinstance(assessment.signals, tuple)
        assert len(assessment.signals) == 1
        assert assessment.signals[0] == signal

    def test_allowed_clarity_values(self) -> None:
        assert RoutingClarity.CLEAR == "clear"
        assert RoutingClarity.AMBIGUOUS == "ambiguous"
        assert set(RoutingClarity) == {RoutingClarity.CLEAR, RoutingClarity.AMBIGUOUS}

    def test_allowed_signal_strength_values(self) -> None:
        assert SignalStrength.STRONG == "strong"
        assert SignalStrength.WEAK == "weak"
        assert set(SignalStrength) == {SignalStrength.STRONG, SignalStrength.WEAK}

    def test_typed_signals_validation(self) -> None:
        sig = RoutingSignal(
            signal_type=RoutingSignalType.SUPPLIED_LOCAL_ARTIFACT,
            target_capability=AgentCapability.GENERAL,
            strength=SignalStrength.STRONG,
            evidence="def func(): pass",
        )
        assert sig.signal_type == RoutingSignalType.SUPPLIED_LOCAL_ARTIFACT
        assert sig.target_capability == AgentCapability.GENERAL
        assert sig.strength == SignalStrength.STRONG

        with pytest.raises(TypeError, match="signal_type must be RoutingSignalType"):
            RoutingSignal(
                signal_type="invalid",  # type: ignore[arg-type]
                target_capability=AgentCapability.GENERAL,
                strength=SignalStrength.STRONG,
            )

        with pytest.raises(TypeError, match="target_capability must be AgentCapability"):
            RoutingSignal(
                signal_type=RoutingSignalType.SUPPLIED_LOCAL_ARTIFACT,
                target_capability="general",  # type: ignore[arg-type]
                strength=SignalStrength.STRONG,
            )

    def test_invalid_or_empty_assessment_state_rejected(self) -> None:
        with pytest.raises(ValueError, match="reason must be a non-empty string"):
            RoutingAssessment(
                provisional_capability=AgentCapability.GENERAL,
                clarity=RoutingClarity.CLEAR,
                signals=(),
                reason="   ",
            )

        with pytest.raises(TypeError, match="provisional_capability must be AgentCapability"):
            RoutingAssessment(
                provisional_capability="general",  # type: ignore[arg-type]
                clarity=RoutingClarity.CLEAR,
                signals=(),
                reason="Valid reason",
            )

        with pytest.raises(TypeError, match="signals must be a tuple"):
            RoutingAssessment(
                provisional_capability=AgentCapability.GENERAL,
                clarity=RoutingClarity.CLEAR,
                signals=[],  # type: ignore[arg-type]
                reason="Valid reason",
            )


class TestDeterministicAssessmentPolicy:
    """Validates deterministic signal evaluation and clarity policies against required patterns."""

    @pytest.fixture
    def router(self) -> DeterministicCapabilityRouter:
        return DeterministicCapabilityRouter()

    def test_weak_verbs_alone_do_not_determine_research(
        self, router: DeterministicCapabilityRouter
    ) -> None:
        prompts = [
            "Research",
            "Compare",
            "Investigate",
            "Research consensus algorithms",
            "Investigate distributed state synchronization",
            "Analyze data portability across clouds",
            "Compare local and cloud LLMs",
            "Evaluate caching strategies",
            "Benchmark redis throughput",
            "Survey available vector databases",
            "Explore fault recovery models",
            "Assess latency tradeoffs",
            "Perform a detailed study of network partitions",
        ]
        for prompt in prompts:
            assessment = router.assess(prompt)
            assert assessment.provisional_capability == AgentCapability.GENERAL
            assert assessment.clarity == RoutingClarity.AMBIGUOUS
            assert router.route(prompt) == AgentCapability.GENERAL
            assert any(
                s.signal_type == RoutingSignalType.WEAK_RESEARCH_VERB
                for s in assessment.signals
            )
            assert "Weak research verb" in assessment.reason

    @pytest.mark.parametrize(
        "prompt",
        [
            "Research trends in EDM music",
            "Compare PostgreSQL and CockroachDB for multi-region writes",
            "Conduct a literature review of retrieval-augmented generation",
            "Deep dive into EV battery recycling technologies",
        ],
    )
    def test_contextual_research_synthesis_routes_to_research(
        self, router: DeterministicCapabilityRouter, prompt: str
    ) -> None:
        assessment = router.assess(prompt)

        assert assessment.provisional_capability == AgentCapability.RESEARCH
        assert assessment.clarity == RoutingClarity.CLEAR
        assert any(
            signal.signal_type == RoutingSignalType.RESEARCH_SYNTHESIS
            and signal.strength == SignalStrength.STRONG
            for signal in assessment.signals
        )

    @pytest.mark.parametrize(
        ("prompt", "evidence"),
        [
            ("Investigate this traceback in my Python script: KeyError", "traceback"),
            ("Investigate this exception", "this exception"),
            ("Analyze this stack trace", "this stack trace"),
            ("Compare these two local Python functions", "these two local Python functions"),
            ("Benchmark this Python function", "this Python function"),
        ],
    )
    def test_explicit_local_targets_remain_general(
        self,
        router: DeterministicCapabilityRouter,
        prompt: str,
        evidence: str,
    ) -> None:
        assessment = router.assess(prompt)

        assert assessment.provisional_capability == AgentCapability.GENERAL
        assert assessment.clarity == RoutingClarity.CLEAR
        assert any(
            signal.signal_type == RoutingSignalType.LOCAL_CODE_OR_DEBUGGING
            and signal.strength == SignalStrength.STRONG
            and evidence.lower() in signal.evidence.lower()
            for signal in assessment.signals
        )
        assert any(
            signal.signal_type == RoutingSignalType.WEAK_RESEARCH_VERB
            for signal in assessment.signals
        )
        assert not any(
            signal.signal_type == RoutingSignalType.RESEARCH_SYNTHESIS
            for signal in assessment.signals
        )

    def test_compare_assess_on_supplied_local_artifacts(
        self, router: DeterministicCapabilityRouter
    ) -> None:
        prompt_json = (
            "Assess whether this JSON payload is valid:\n"
            '{"user_id": 101, "role": "admin", "active": true}'
        )
        assessment_json = router.assess(prompt_json)
        assert assessment_json.provisional_capability == AgentCapability.GENERAL
        assert assessment_json.clarity == RoutingClarity.CLEAR
        assert any(
            s.signal_type == RoutingSignalType.SUPPLIED_LOCAL_ARTIFACT
            for s in assessment_json.signals
        )
        assert "Strong general signals detected" in assessment_json.reason

        prompt_config = (
            "Compare the two configuration files below:\n"
            "[server]\nport = 8080\nhost = localhost\n\n"
            "[server]\nport = 9090\nhost = 0.0.0.0"
        )
        assessment_config = router.assess(prompt_config)
        assert assessment_config.provisional_capability == AgentCapability.GENERAL
        assert assessment_config.clarity == RoutingClarity.CLEAR
        assert any(
            s.signal_type == RoutingSignalType.SUPPLIED_LOCAL_ARTIFACT
            for s in assessment_config.signals
        )

    def test_currently_latest_supported_version_questions(
        self, router: DeterministicCapabilityRouter
    ) -> None:
        prompt = "What version of Redis is currently supported by the latest Ubuntu LTS release?"
        assessment = router.assess(prompt)
        assert assessment.provisional_capability == AgentCapability.RESEARCH
        assert assessment.clarity == RoutingClarity.CLEAR
        assert any(
            s.signal_type == RoutingSignalType.FRESHNESS_OR_CURRENT_INFO
            for s in assessment.signals
        )

    def test_todays_changing_pricing(self, router: DeterministicCapabilityRouter) -> None:
        prompt = "Review today's AWS Lambda pricing changes and calculate monthly cost differences."
        assessment = router.assess(prompt)
        assert assessment.provisional_capability == AgentCapability.RESEARCH
        assert assessment.clarity == RoutingClarity.CLEAR
        assert any(
            s.signal_type == RoutingSignalType.FRESHNESS_OR_CURRENT_INFO
            for s in assessment.signals
        )

    def test_current_external_guidance(self, router: DeterministicCapabilityRouter) -> None:
        prompt = (
            "Summarize the incident report and compare its findings with current NIST guidance."
        )
        assessment = router.assess(prompt)
        assert assessment.provisional_capability == AgentCapability.RESEARCH
        assert assessment.clarity == RoutingClarity.CLEAR
        assert any(
            s.signal_type == RoutingSignalType.FRESHNESS_OR_CURRENT_INFO
            for s in assessment.signals
        )
        assert any(
            s.signal_type == RoutingSignalType.OUTSIDE_AUTHORITY_OR_PROVIDER
            for s in assessment.signals
        )

    def test_current_local_variable_state_remains_general(
        self, router: DeterministicCapabilityRouter
    ) -> None:
        prompts = [
            "Check current_user permissions in this active session and verify token expiry.",
            "Inspect current_node in the syntax tree.",
            "Why does currentUser get set to null on logout?",
            "What is the current value of the loop counter in this function?",
        ]
        for p in prompts:
            assessment = router.assess(p)
            assert assessment.provisional_capability == AgentCapability.GENERAL
            assert assessment.clarity == RoutingClarity.CLEAR
            assert not any(
                s.signal_type == RoutingSignalType.FRESHNESS_OR_CURRENT_INFO
                for s in assessment.signals
            )

    def test_electrical_current_remains_general(
        self, router: DeterministicCapabilityRouter
    ) -> None:
        prompts = [
            "Calculate the voltage drop given a current of 10A across a 5 ohm resistor.",
            "Explain the difference between direct current and alternating current.",
            "Calculate the electrical current flowing through a 100 ohm resistor.",
        ]
        for p in prompts:
            assessment = router.assess(p)
            assert assessment.provisional_capability == AgentCapability.GENERAL
            assert assessment.clarity == RoutingClarity.CLEAR
            assert not any(
                s.signal_type == RoutingSignalType.FRESHNESS_OR_CURRENT_INFO
                for s in assessment.signals
            )

    def test_supplied_configuration_comparison(
        self, router: DeterministicCapabilityRouter
    ) -> None:
        prompt = (
            "Compare the two configuration files below:\n"
            "[database]\nhost = 127.0.0.1\nport = 5432\n\n"
            "[database]\nhost = 10.0.0.1\nport = 5432"
        )
        assessment = router.assess(prompt)
        assert assessment.provisional_capability == AgentCapability.GENERAL
        assert assessment.clarity == RoutingClarity.CLEAR
        assert any(
            s.signal_type == RoutingSignalType.SUPPLIED_LOCAL_ARTIFACT
            for s in assessment.signals
        )

    def test_actual_artifact_required_before_supplied_local_artifact_emitted(
        self, router: DeterministicCapabilityRouter
    ) -> None:
        # Prompt only mentions "below" or "this" without actual structure/content
        prompt = "Compare the two configuration files below"
        assessment = router.assess(prompt)
        assert not any(
            s.signal_type == RoutingSignalType.SUPPLIED_LOCAL_ARTIFACT
            for s in assessment.signals
        )

    def test_reasons_agree_with_signal_sets(
        self, router: DeterministicCapabilityRouter
    ) -> None:
        test_prompts = [
            "Search for the latest OpenSSL CVEs and write a checking script.",
            "Find peer-reviewed studies and cite sources.",
            "Write a Python script that benchmarks quicksort and mergesort.",
            "What version of Redis is currently supported by the latest Ubuntu LTS release?",
            "Research consensus algorithms",
            "Hello agent",
        ]
        all_signal_type_values = {st.value for st in RoutingSignalType}

        for p in test_prompts:
            assessment = router.assess(p)
            assert assessment.reason.strip(), f"Empty reason for {p!r}"
            # Verify no reason claims a signal type that is absent from assessment.signals
            present_signal_values = {s.signal_type.value for s in assessment.signals}
            for st_val in all_signal_type_values:
                if st_val in assessment.reason:
                    assert st_val in present_signal_values, (
                        f"Reason claims signal {st_val!r} which is absent from signals: {present_signal_values}"
                    )
            if not assessment.signals:
                assert "No research signals detected" in assessment.reason

    def test_legacy_v2_exists_only_in_evaluation_code(self) -> None:
        import orchestrator.domain
        import orchestrator.domain.router

        assert not hasattr(orchestrator.domain, "LegacyRoutingV2Router")
        assert not hasattr(orchestrator.domain.router, "LegacyRoutingV2Router")

        from evaluation.routing.legacy_v2 import LegacyRoutingV2Router

        router = LegacyRoutingV2Router()
        assert router.route("Benchmark redis throughput") == AgentCapability.RESEARCH

    def test_route_prompt_backward_compatibility(
        self, router: DeterministicCapabilityRouter
    ) -> None:
        prompts = [
            "What is 15% of 850?",
            "Who is the current Prime Minister of the United Kingdom?",
            "Write a binary search algorithm in TypeScript.",
            "Find peer-reviewed studies and cite sources.",
            "Research consensus algorithms",
        ]
        for p in prompts:
            assessment = router.assess(p)
            capability = router.route(p)
            assert capability == assessment.provisional_capability

    def test_assessment_signals_and_reason_are_deterministic(
        self, router: DeterministicCapabilityRouter
    ) -> None:
        prompt = "Compare these two Python dictionary merging methods: {**a, **b} vs a | b."
        a1 = router.assess(prompt)
        a2 = router.assess(prompt)

        assert a1.provisional_capability == a2.provisional_capability
        assert a1.clarity == a2.clarity
        assert a1.signals == a2.signals
        assert a1.reason == a2.reason


class RouteOnlyTestDouble:
    """Test double implementing only CapabilityRouter (route), not AssessingCapabilityRouter."""

    def route(self, prompt: str) -> AgentCapability:
        if "research" in prompt.lower():
            return AgentCapability.RESEARCH
        return AgentCapability.GENERAL


class TestSubmitTaskAssessingProtocol:
    """Verifies SubmitTask's interaction with AssessingCapabilityRouter and route-only routers."""

    @pytest.mark.anyio
    async def test_assessing_capability_router_protocol_usage(self) -> None:
        queue = FakeTaskQueue()
        router = DeterministicCapabilityRouter()
        assert isinstance(router, AssessingCapabilityRouter)

        submit = SubmitTask(queue=queue, router=router)
        await submit.execute("What version of Redis is currently supported by the latest Ubuntu LTS release?")
        assert len(queue.enqueued) == 1
        _, cap = queue.enqueued[0]
        assert cap == AgentCapability.RESEARCH

    @pytest.mark.anyio
    async def test_route_only_router_compatibility(self) -> None:
        queue = FakeTaskQueue()
        router = RouteOnlyTestDouble()
        assert not isinstance(router, AssessingCapabilityRouter)

        submit = SubmitTask(queue=queue, router=router)  # type: ignore[arg-type]
        await submit.execute("please research this")
        assert len(queue.enqueued) == 1
        _, cap = queue.enqueued[0]
        assert cap == AgentCapability.RESEARCH

    @pytest.mark.anyio
    async def test_assessing_router_invalid_return_type_raises_type_error(self) -> None:
        class BrokenAssessingRouter:
            def route(self, prompt: str) -> AgentCapability:
                return AgentCapability.GENERAL

            def assess(self, prompt: str) -> object:
                return "not_a_routing_assessment"

        queue = FakeTaskQueue()
        submit = SubmitTask(queue=queue, router=BrokenAssessingRouter())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must return RoutingAssessment"):
            await submit.execute("some prompt")

    @pytest.mark.anyio
    async def test_persisted_classification_alignment_with_actual_routing(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from tests.test_runtime_dispatch import make_candidate

        candidate_general = make_candidate("worker-1", capability="general", dispatch_target="agent.tasks.general")
        candidate_research = make_candidate("research-1", capability="research", dispatch_target="agent.tasks.research")
        finder = MagicMock()
        finder.execute = AsyncMock(
            side_effect=lambda cap: [candidate_general] if cap == AgentCapability.GENERAL else [candidate_research]
        )

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock()
        recorder = MagicMock()
        recorder.record_routing_decision = AsyncMock()

        router = DeterministicCapabilityRouter()
        submit = SubmitTask(
            router=router,
            candidate_finder=finder,
            dispatcher=dispatcher,
            routing_recorder=recorder,
        )

        # 1. "Research consensus algorithms" has weak verb alone -> provisionally GENERAL / AMBIGUOUS
        assessment = router.assess("Research consensus algorithms")
        assert assessment.provisional_capability == AgentCapability.GENERAL
        assert assessment.clarity == RoutingClarity.AMBIGUOUS

        await submit.execute("Research consensus algorithms")
        assert recorder.record_routing_decision.call_count == 1
        decision_general = recorder.record_routing_decision.call_args[0][0]
        # Persisted prompt classification must NOT claim a Research route!
        assert decision_general.prompt_classification == "general"
        assert decision_general.required_capabilities == ["general"]

        # 2. Strong research prompt -> RESEARCH / CLEAR
        recorder.record_routing_decision.reset_mock()
        await submit.execute("Research the latest quantum key distribution breakthroughs")
        assert recorder.record_routing_decision.call_count == 1
        decision_research = recorder.record_routing_decision.call_args[0][0]
        # Must persist classification consistent with RESEARCH
        assert decision_research.prompt_classification in (
            "research",
            "research_and_investigation",
            "freshness_and_current_info",
        )
        assert decision_research.required_capabilities == ["research"]


class TestSubmitTaskExplicitOverride:
    """Verifies that caller-supplied explicit overrides bypass inference while preserving execution flow."""

    @pytest.mark.anyio
    async def test_explicit_override_bypasses_router_inference(self) -> None:
        queue = FakeTaskQueue()
        submit_task = SubmitTask(queue=queue)

        # Clear research prompt explicitly forced to GENERAL
        task = await submit_task.execute(
            "Find peer-reviewed studies and cite sources.",
            capability=AgentCapability.GENERAL,
        )

        assert len(queue.enqueued) == 1
        enqueued_task, capability = queue.enqueued[0]
        assert enqueued_task == task
        assert capability == AgentCapability.GENERAL

    @pytest.mark.anyio
    async def test_explicit_override_preserves_candidate_evaluation(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from tests.test_runtime_dispatch import make_candidate

        candidate = make_candidate(
            "research-agent-1",
            capability="research",
            dispatch_target="agent.tasks.research",
        )
        finder = MagicMock()
        finder.execute = AsyncMock(return_value=[candidate])

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock()

        recorder = MagicMock()
        recorder.record_routing_decision = AsyncMock()

        submit_task = SubmitTask(
            candidate_finder=finder,
            dispatcher=dispatcher,
            routing_recorder=recorder,
        )

        await submit_task.execute(
            "Write a python script",  # naturally general
            capability=AgentCapability.RESEARCH,  # explicit override to research
        )

        # Candidate finder was called with the explicitly specified capability
        finder.execute.assert_called_once_with(AgentCapability.RESEARCH)

        # Routing decision records explicit override provenance
        assert recorder.record_routing_decision.call_count == 1
        decision = recorder.record_routing_decision.call_args[0][0]
        assert decision.required_capabilities == ["research"]
        assert "(explicit capability override)" in decision.rationale
