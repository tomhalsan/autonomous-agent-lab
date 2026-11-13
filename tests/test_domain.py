import pytest
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.router import DeterministicCapabilityRouter, ResearchIntentCategory
from orchestrator.domain.task import Task


def test_task_rejects_blank_prompt() -> None:
    with pytest.raises(ValueError):
        Task("   ")


def test_task_accepts_prompt() -> None:
    assert Task("inspect system").prompt == "inspect system"


@pytest.mark.parametrize(
    "prompt",
    [
        "Research the latest consensus algorithms",
        "Investigate recent distributed state synchronization developments",
        "Find sources about data portability across clouds",
        "Compare current features and pricing of local and cloud LLMs",
        "Research trends in EDM music",
        "Compare PostgreSQL and CockroachDB for multi-region writes",
        "Conduct a literature review of retrieval-augmented generation",
        "Deep dive into EV battery recycling technologies",
        "Find the published benchmarks comparing redis throughput",
        "Survey available vector databases as of today",
        "Find recent CVEs for fault recovery models",
        "Provide citations for latency tradeoffs",
        "Perform a detailed study of network partitions and cite papers",
    ],
)
def test_deterministic_router_routes_research_prompts(prompt: str) -> None:
    router = DeterministicCapabilityRouter()
    assert router.route(prompt) == AgentCapability.RESEARCH


@pytest.mark.parametrize(
    "prompt",
    [
        "Research consensus algorithms",
        "Investigate distributed state synchronization",
        "Analyze data portability across clouds",
        "Research",
        "Compare",
        "Investigate",
        "Compare local and cloud LLMs",
        "Evaluate caching strategies",
        "Benchmark redis throughput",
        "Survey available vector databases",
        "Explore fault recovery models",
        "Assess latency tradeoffs",
        "Perform a detailed study of network partitions",
    ],
)
def test_deterministic_router_weak_verbs_alone_route_to_general(prompt: str) -> None:
    router = DeterministicCapabilityRouter()
    assert router.route(prompt) == AgentCapability.GENERAL


@pytest.mark.parametrize(
    "prompt",
    [
        "build agent",
        "test task prompt",
        "inspect system",
        "deploy application to cluster",
        "fix bug in auth handler",
        "restart worker container",
        "hello agent",
    ],
)
def test_deterministic_router_routes_general_prompts(prompt: str) -> None:
    router = DeterministicCapabilityRouter()
    assert router.route(prompt) == AgentCapability.GENERAL


def test_deterministic_router_routes_case_insensitively() -> None:
    router = DeterministicCapabilityRouter()
    assert router.route("rEsEaRcH the latest developments") == AgentCapability.RESEARCH
    assert router.route("INVESTIGATE recent CVEs") == AgentCapability.RESEARCH
    assert router.route("FIND SOURCES for security audit") == AgentCapability.RESEARCH


@pytest.mark.parametrize(
    "prompt",
    [
        "Investigate this traceback in my Python script: KeyError",
        "Investigate this exception",
        "Analyze this stack trace",
        "Compare these two local Python functions",
        "Benchmark this Python function",
    ],
)
def test_deterministic_router_keeps_local_code_tasks_general(prompt: str) -> None:
    router = DeterministicCapabilityRouter()
    assert router.route(prompt) == AgentCapability.GENERAL


def test_deterministic_router_respects_word_boundaries() -> None:
    router = DeterministicCapabilityRouter()
    # "company" contains "comp" but is not "compare"
    assert router.route("Search company directory") == AgentCapability.GENERAL
    # "restudy" contains "study" but is not a standalone word
    assert router.route("Please restudy this later") == AgentCapability.GENERAL


def test_deterministic_router_custom_keywords() -> None:
    router = DeterministicCapabilityRouter(keywords={"custom_probe"})
    assert router.route("Run custom_probe now") == AgentCapability.RESEARCH
    # Default keywords are not used when custom keywords are explicitly provided
    assert router.route("Research consensus") == AgentCapability.GENERAL


# --- Routing v2 Intent-Aware Capability Router Tests ---


@pytest.mark.parametrize(
    ("prompt", "expected_capability"),
    [
        ("What is quantum computing?", AgentCapability.GENERAL),
        (
            "Research the latest developments in quantum computing.",
            AgentCapability.RESEARCH,
        ),
        (
            "Explain the difference between containers and virtual machines.",
            AgentCapability.GENERAL,
        ),
        (
            "Find current information about container adoption.",
            AgentCapability.RESEARCH,
        ),
        ("Find sources about container security.", AgentCapability.RESEARCH),
        (
            "What are the latest developments in battery technology?",
            AgentCapability.RESEARCH,
        ),
        ("Explain container adoption.", AgentCapability.GENERAL),
    ],
)
def test_routing_v2_required_intent_prompts(
    prompt: str, expected_capability: AgentCapability
) -> None:
    router = DeterministicCapabilityRouter()
    assert router.route(prompt) == expected_capability


def test_routing_v2_intent_detection_categorization() -> None:
    router = DeterministicCapabilityRouter()

    # Explicit research/investigation
    assert (
        router.detect_intent("Research consensus algorithms")
        == ResearchIntentCategory.RESEARCH_AND_INVESTIGATION
    )
    assert (
        router.detect_intent("Investigate distributed state synchronization")
        == ResearchIntentCategory.RESEARCH_AND_INVESTIGATION
    )
    assert (
        router.detect_intent("Benchmark redis throughput")
        == ResearchIntentCategory.RESEARCH_AND_INVESTIGATION
    )

    # Freshness / current information
    assert (
        router.detect_intent("What are the latest developments in battery technology?")
        == ResearchIntentCategory.FRESHNESS_AND_CURRENT_INFO
    )
    assert (
        router.detect_intent("Current status of Python 3.14 release")
        == ResearchIntentCategory.FRESHNESS_AND_CURRENT_INFO
    )
    assert (
        router.detect_intent("What is the latest on fusion energy?")
        == ResearchIntentCategory.FRESHNESS_AND_CURRENT_INFO
    )

    # Source / evidence finding
    assert (
        router.detect_intent("Find sources about container security.")
        == ResearchIntentCategory.SOURCES_AND_EVIDENCE
    )
    assert (
        router.detect_intent("Find evidence of water on Mars")
        == ResearchIntentCategory.SOURCES_AND_EVIDENCE
    )
    assert (
        router.detect_intent("Provide citations for quantum supremacy")
        == ResearchIntentCategory.SOURCES_AND_EVIDENCE
    )

    # General explanatory / knowledge requests have no research intent
    assert router.detect_intent("What is quantum computing?") is None
    assert (
        router.detect_intent("Explain the difference between containers and virtual machines.")
        is None
    )
    assert router.detect_intent("Explain container adoption.") is None
    assert router.detect_intent("How does garbage collection work in Python?") is None


def test_routing_v2_distinguishes_source_code_from_research_sources() -> None:
    router = DeterministicCapabilityRouter()
    # Source code requests are general developer tasks, not research source requests
    assert router.route("Find source code for web crawler") == AgentCapability.GENERAL
    # Requests for citations / sources about topics are research
    assert router.route("Find sources about web crawler architectures") == AgentCapability.RESEARCH


