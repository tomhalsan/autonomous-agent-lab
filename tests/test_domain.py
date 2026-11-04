import pytest
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.router import DeterministicCapabilityRouter
from orchestrator.domain.task import Task


def test_task_rejects_blank_prompt() -> None:
    with pytest.raises(ValueError):
        Task("   ")


def test_task_accepts_prompt() -> None:
    assert Task("inspect system").prompt == "inspect system"


@pytest.mark.parametrize(
    "prompt",
    [
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
    ],
)
def test_deterministic_router_routes_research_prompts(prompt: str) -> None:
    router = DeterministicCapabilityRouter()
    assert router.route(prompt) == AgentCapability.RESEARCH


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
    assert router.route("rEsEaRcH consensus") == AgentCapability.RESEARCH
    assert router.route("INVESTIGATE failure") == AgentCapability.RESEARCH
    assert router.route("AnAlYzE logs") == AgentCapability.RESEARCH


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

