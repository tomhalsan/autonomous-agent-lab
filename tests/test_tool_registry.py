from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from agent_contracts import (
    Tool,
    ToolDefinition,
    ToolInvocationRequest,
    ToolInvocationResult,
    ToolRegistration,
    ToolRegistry,
    ToolRegistryReader,
)
from agent_infrastructure import InMemoryToolRegistry


class DummyTool:
    """Minimal deterministic tool implementation for unit testing."""

    def __init__(
        self,
        name: str,
        description: str = "Test tool description",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self._definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters if parameters is not None else {},
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        return ToolInvocationResult.create_success(
            call_id=request.call_id,
            tool_name=request.tool_name,
            output={"result": "ok"},
        )


def test_tool_registry_protocol_conformance() -> None:
    registry = InMemoryToolRegistry()
    assert isinstance(registry, ToolRegistry)
    assert isinstance(registry, ToolRegistryReader)


def test_tool_registration_properties_and_immutability() -> None:
    tool = DummyTool(
        name="web_search",
        parameters={"required": ["query", "limit"]},
    )
    assert isinstance(tool, Tool)
    registration = ToolRegistration(tool=tool, category="search")

    assert registration.name == "web_search"
    assert registration.category == "search"
    assert registration.definition == tool.definition
    assert registration.required_inputs == frozenset({"query", "limit"})

    with pytest.raises(FrozenInstanceError):
        registration.category = "other"  # type: ignore[misc]


def test_tool_registration_rejects_non_tool() -> None:
    with pytest.raises(TypeError, match="must implement Tool protocol"):
        ToolRegistration(tool="not_a_tool", category="general")  # type: ignore[arg-type]


def test_tool_registration_requires_non_empty_category() -> None:
    tool = DummyTool(name="test_tool")
    for invalid_cat in ["", "   "]:
        with pytest.raises(ValueError, match="category must be a non-empty string"):
            ToolRegistration(tool=tool, category=invalid_cat)


def test_tool_registration_required_inputs_extraction() -> None:
    # 1. Missing required key defaults to empty frozenset
    tool_no_req = DummyTool(name="tool_1", parameters={"properties": {"x": {"type": "string"}}})
    reg_no_req = ToolRegistration(tool=tool_no_req, category="general")
    assert reg_no_req.required_inputs == frozenset()

    # 2. Empty collection defaults to empty frozenset
    tool_empty_req = DummyTool(name="tool_2", parameters={"required": []})
    reg_empty_req = ToolRegistration(tool=tool_empty_req, category="general")
    assert reg_empty_req.required_inputs == frozenset()

    # 3. Collection of unique non-empty strings extracted correctly
    tool_valid_req = DummyTool(name="tool_3", parameters={"required": ["a", "b"]})
    reg_valid_req = ToolRegistration(tool=tool_valid_req, category="general")
    assert reg_valid_req.required_inputs == frozenset({"a", "b"})


def test_tool_registration_rejects_malformed_required_inputs() -> None:
    tool = DummyTool(name="test_tool")

    # A string is not a valid collection of required inputs
    tool._definition = ToolDefinition(
        name="test_tool",
        description="desc",
        parameters={"required": "query"},
    )
    with pytest.raises(ValueError, match="parameters\\['required'\\] must be a collection"):
        ToolRegistration(tool=tool, category="general")

    # Non-string element in collection
    tool._definition = ToolDefinition(
        name="test_tool",
        description="desc",
        parameters={"required": ["query", 123]},  # type: ignore[list-item]
    )
    with pytest.raises(ValueError, match="must contain non-empty strings"):
        ToolRegistration(tool=tool, category="general")

    # Empty/whitespace string in collection
    tool._definition = ToolDefinition(
        name="test_tool",
        description="desc",
        parameters={"required": ["query", "   "]},
    )
    with pytest.raises(ValueError, match="must contain non-empty strings"):
        ToolRegistration(tool=tool, category="general")

    # Duplicate string elements
    tool._definition = ToolDefinition(
        name="test_tool",
        description="desc",
        parameters={"required": ["query", "query"]},
    )
    with pytest.raises(ValueError, match="must contain unique strings"):
        ToolRegistration(tool=tool, category="general")


def test_registry_register_and_unregister_lifecycle() -> None:
    registry = InMemoryToolRegistry()
    tool = DummyTool(name="echo")

    reg = registry.register(tool=tool, category="utility")
    assert reg.name == "echo"
    assert registry.get("echo") == reg
    assert registry.list_all() == [reg]

    unreg = registry.unregister("echo")
    assert unreg == reg
    assert registry.get("echo") is None
    assert registry.list_all() == []

    # Unregistering nonexistent tool returns None
    assert registry.unregister("echo") is None


def test_registry_duplicate_registration_rejected() -> None:
    registry = InMemoryToolRegistry()
    tool_1 = DummyTool(name="search")
    tool_2 = DummyTool(name="search", description="different description")

    registry.register(tool=tool_1, category="retrieval")

    with pytest.raises(ValueError, match="Tool with name 'search' is already registered"):
        registry.register(tool=tool_2, category="retrieval")


def test_registry_explicit_unregister_then_reregister() -> None:
    registry = InMemoryToolRegistry()
    tool_1 = DummyTool(name="search", description="first version")
    tool_2 = DummyTool(name="search", description="second version")

    reg_1 = registry.register(tool=tool_1, category="retrieval")
    assert registry.get("search") == reg_1

    assert registry.unregister("search") == reg_1
    assert registry.get("search") is None

    reg_2 = registry.register(tool=tool_2, category="retrieval_v2")
    assert registry.get("search") == reg_2
    assert reg_2.definition.description == "second version"


def test_registry_exact_name_lookup_semantics() -> None:
    registry = InMemoryToolRegistry()
    tool_lower = DummyTool(name="search")
    tool_upper = DummyTool(name="Search")

    registry.register(tool=tool_lower, category="retrieval")
    registry.register(tool=tool_upper, category="retrieval")

    # Lookup preserves exact casing and does not strip
    assert registry.get("search") == ToolRegistration(tool=tool_lower, category="retrieval")
    assert registry.get("Search") == ToolRegistration(tool=tool_upper, category="retrieval")
    assert registry.get("SEARCH") is None
    assert registry.get(" search ") is None


def test_registry_category_lookup_case_insensitivity() -> None:
    registry = InMemoryToolRegistry()
    t1 = DummyTool(name="tool1")
    t2 = DummyTool(name="tool2")
    t3 = DummyTool(name="tool3")

    reg1 = registry.register(tool=t1, category="Search")
    reg2 = registry.register(tool=t2, category="search")
    reg3 = registry.register(tool=t3, category="Math")

    # Case-insensitive category query matches reg1 and reg2
    assert registry.find_by_category("search") == [reg1, reg2]
    assert registry.find_by_category("SEARCH") == [reg1, reg2]
    assert registry.find_by_category("  search  ") == [reg1, reg2]

    # Category matching math
    assert registry.find_by_category("math") == [reg3]

    # Nonexistent category
    assert registry.find_by_category("database") == []

    # Invalid category input
    with pytest.raises(ValueError, match="category must be a non-empty string"):
        registry.find_by_category("")
    with pytest.raises(ValueError, match="category must be a non-empty string"):
        registry.find_by_category("   ")


def test_registry_find_by_required_inputs_superset_matching() -> None:
    registry = InMemoryToolRegistry()
    t_none = DummyTool(name="no_req", parameters={})
    t_q = DummyTool(name="req_q", parameters={"required": ["query"]})
    t_ql = DummyTool(name="req_ql", parameters={"required": ["query", "limit"]})
    t_u = DummyTool(name="req_u", parameters={"required": ["url"]})

    registry.register(tool=t_none, category="general")
    reg_q = registry.register(tool=t_q, category="search")
    reg_ql = registry.register(tool=t_ql, category="search")
    reg_u = registry.register(tool=t_u, category="fetch")

    # Tools that require at least 'query' -> reg_q and reg_ql
    assert registry.find_by_required_inputs(["query"]) == [reg_q, reg_ql]
    assert registry.find_by_required_inputs({"query"}) == [reg_q, reg_ql]

    # Tools that require at least 'query' and 'limit' -> only reg_ql
    assert registry.find_by_required_inputs(["query", "limit"]) == [reg_ql]

    # Tools that require at least 'url' -> only reg_u
    assert registry.find_by_required_inputs(["url"]) == [reg_u]

    # Empty requested inputs -> raises ValueError
    with pytest.raises(ValueError, match="required_inputs must contain at least one input"):
        registry.find_by_required_inputs([])
    with pytest.raises(ValueError, match="required_inputs must contain at least one input"):
        registry.find_by_required_inputs(set())

    # Disjoint requested inputs -> matches none
    assert registry.find_by_required_inputs(["query", "url"]) == []

    # Input validation for query
    with pytest.raises(TypeError, match="required_inputs must be a collection of strings"):
        registry.find_by_required_inputs("query")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="required_inputs must contain non-empty strings"):
        registry.find_by_required_inputs(["query", ""])


def test_registry_deterministic_registration_order() -> None:
    registry = InMemoryToolRegistry()
    t1 = DummyTool(name="alpha", parameters={"required": ["id"]})
    t2 = DummyTool(name="beta", parameters={"required": ["id"]})
    t3 = DummyTool(name="gamma", parameters={"required": ["id"]})

    reg1 = registry.register(tool=t1, category="cat1")
    reg2 = registry.register(tool=t2, category="cat1")
    reg3 = registry.register(tool=t3, category="cat1")

    # list_all preserves registration order
    assert registry.list_all() == [reg1, reg2, reg3]

    # find_by_category preserves registration order
    assert registry.find_by_category("cat1") == [reg1, reg2, reg3]

    # find_by_required_inputs preserves registration order
    assert registry.find_by_required_inputs(["id"]) == [reg1, reg2, reg3]


def test_registry_returned_snapshots_are_isolated() -> None:
    registry = InMemoryToolRegistry()
    tool = DummyTool(name="tool", parameters={"required": ["id"]})
    reg = registry.register(tool=tool, category="cat")

    list_snapshot = registry.list_all()
    list_snapshot.clear()
    assert registry.list_all() == [reg]

    cat_snapshot = registry.find_by_category("cat")
    cat_snapshot.pop()
    assert registry.find_by_category("cat") == [reg]

    req_snapshot = registry.find_by_required_inputs(["id"])
    req_snapshot.append("mutation")  # type: ignore[arg-type]
    assert registry.find_by_required_inputs(["id"]) == [reg]
