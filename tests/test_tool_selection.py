"""Provider-neutral selection decisions and deterministic reference selection."""

import ast
from pathlib import Path

import pytest
from agent_contracts import (
    ToolDefinition,
    ToolInvocationProposal,
    ToolInvocationRequest,
    ToolInvocationResult,
    ToolRegistration,
    ToolSelectionDecision,
    ToolSelectionKind,
)
from agent_policy import DeterministicToolSelector


class _Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, object] | None = None,
    ) -> None:
        self._definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or {},
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        return ToolInvocationResult.create_success(
            call_id=request.call_id, tool_name=request.tool_name, output=None
        )


def _registration(
    name: str,
    description: str,
    category: str,
    parameters: dict[str, object] | None = None,
) -> ToolRegistration:
    return ToolRegistration(
        _Tool(name, description, parameters),
        category=category,
    )


def test_selection_decisions_have_explicit_exclusive_outcomes() -> None:
    no_tool = ToolSelectionDecision.no_tool_needed()
    proposed = ToolSelectionDecision.proposed_invocation("search", {"query": "weather"})

    assert no_tool.kind is ToolSelectionKind.NO_TOOL_NEEDED
    assert no_tool.invocation is None
    assert proposed.kind is ToolSelectionKind.PROPOSED_INVOCATION
    assert proposed.invocation is not None
    assert proposed.invocation.tool_name == "search"
    assert proposed.invocation.arguments == {"query": "weather"}

    with pytest.raises(ValueError, match="no-tool decisions must not contain"):
        ToolSelectionDecision(
            kind=ToolSelectionKind.NO_TOOL_NEEDED,
            invocation=ToolInvocationProposal("search", {}),
        )
    with pytest.raises(ValueError, match="require an invocation"):
        ToolSelectionDecision(kind=ToolSelectionKind.PROPOSED_INVOCATION)
    with pytest.raises(TypeError, match="kind must be"):
        ToolSelectionDecision(kind="no_tool_needed")  # type: ignore[arg-type]


def test_invocation_proposal_defensively_copies_json_arguments() -> None:
    original = {"query": "weather", "filters": {"regions": ["north"]}}
    proposal = ToolInvocationProposal("search", original)
    original["filters"]["regions"].append("south")
    returned = proposal.arguments
    returned["filters"]["regions"].append("west")

    assert proposal.arguments == {"query": "weather", "filters": {"regions": ["north"]}}
    with pytest.raises(TypeError, match="object keys must be strings"):
        ToolInvocationProposal("search", {1: "not JSON"})  # type: ignore[dict-item]


@pytest.mark.anyio
async def test_deterministic_selector_selects_obvious_tool_and_builds_query() -> None:
    search = _registration(
        "web_search",
        "Search the web for relevant pages",
        "research",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    decision = await DeterministicToolSelector().select(
        "Search the web for climate research", [search]
    )

    assert decision.kind is ToolSelectionKind.PROPOSED_INVOCATION
    assert decision.invocation is not None
    assert decision.invocation.tool_name == "web_search"
    assert decision.invocation.arguments == {"query": "Search the web for climate research"}


@pytest.mark.anyio
async def test_unrelated_or_ambiguous_tasks_return_no_tool() -> None:
    first = _registration("web_search", "Search the web", "research")
    second = _registration("site_search", "Search websites", "research")
    selector = DeterministicToolSelector()

    unrelated = await selector.select("add two integers", [first, second])
    ambiguous = await selector.select("search", [first, second])

    assert unrelated.kind is ToolSelectionKind.NO_TOOL_NEEDED
    assert ambiguous.kind is ToolSelectionKind.NO_TOOL_NEEDED


@pytest.mark.anyio
async def test_category_match_alone_is_not_enough_to_choose_a_tool() -> None:
    tool = _registration("web_search", "Search the web", "research")

    decision = await DeterministicToolSelector().select("research", [tool])

    assert decision.kind is ToolSelectionKind.NO_TOOL_NEEDED


@pytest.mark.anyio
async def test_unique_best_match_is_deterministic_with_multiple_tools() -> None:
    search = _registration("web_search", "Search the web", "research")
    calculator = _registration("calculator", "Add or multiply numbers", "compute")
    selector = DeterministicToolSelector()

    first = await selector.select("search web pages", [calculator, search])
    second = await selector.select("search web pages", [search, calculator])

    assert first == second
    assert first.invocation is not None
    assert first.invocation.tool_name == "web_search"


@pytest.mark.anyio
async def test_metadata_none_tool_is_a_candidate_with_explicit_registration_category() -> None:
    tool = _Tool("web_search", "Search the web")
    registration = ToolRegistration(tool, category="research")

    decision = await DeterministicToolSelector().select("search the web", [registration])

    assert registration.definition.metadata is None
    assert registration.category == "research"
    assert decision.invocation is not None
    assert decision.invocation.tool_name == registration.name
    assert registration.definition.metadata is None


def test_selection_contract_and_policy_do_not_import_provider_or_mcp_types() -> None:
    import agent_contracts.tool_selection as contract_module
    import agent_policy.tool_selection as policy_module

    for module in (contract_module, policy_module):
        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert not any(
            name == "mcp" or name.startswith(("mcp.", "agent_infrastructure", "research_agent"))
            for name in imports
        )
