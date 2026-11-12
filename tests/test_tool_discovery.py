"""Tests for dynamic category-scoped tool catalog discovery."""

import ast
from pathlib import Path

import pytest
from agent_contracts import (
    ToolDefinition,
    ToolInvocationRequest,
    ToolInvocationResult,
)
from agent_infrastructure import InMemoryToolRegistry
from agent_policy import ToolDiscoveryScope, ToolDiscoveryService


class _TestTool:
    def __init__(self, name: str) -> None:
        self._definition = ToolDefinition(name=name, description=f"{name} test tool")

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        return ToolInvocationResult.create_success(
            call_id=request.call_id,
            tool_name=request.tool_name,
            output=None,
        )


def test_discovery_sees_registration_added_after_service_creation() -> None:
    registry = InMemoryToolRegistry()
    discovery = ToolDiscoveryService(registry)
    scope = ToolDiscoveryScope(categories={"research"})

    first = registry.register(_TestTool("first_research"), category="research")
    assert discovery.discover(scope) == [first]

    second = registry.register(_TestTool("second_research"), category="research")
    assert discovery.discover(scope) == [first, second]


def test_discovery_filters_categories_and_preserves_registration_order() -> None:
    registry = InMemoryToolRegistry()
    research_one = registry.register(_TestTool("research_one"), category="research")
    registry.register(_TestTool("compute"), category="compute")
    research_two = registry.register(_TestTool("research_two"), category="Research")
    discovery = ToolDiscoveryService(registry)

    assert discovery.discover(ToolDiscoveryScope(categories={"RESEARCH"})) == [
        research_one,
        research_two,
    ]


def test_empty_scope_discovers_all_current_registrations() -> None:
    registry = InMemoryToolRegistry()
    first = registry.register(_TestTool("first"), category="utility")
    second = registry.register(_TestTool("second"), category="research")
    discovery = ToolDiscoveryService(registry)

    assert discovery.discover(ToolDiscoveryScope()) == [first, second]


def test_empty_registry_and_unmatched_categories_return_empty_catalogs() -> None:
    registry = InMemoryToolRegistry()
    discovery = ToolDiscoveryService(registry)

    assert discovery.discover(ToolDiscoveryScope()) == []

    registry.register(_TestTool("compute"), category="compute")
    assert discovery.discover(ToolDiscoveryScope(categories={"research"})) == []


def test_unclassified_tool_is_discovered_by_explicit_registration_category() -> None:
    registry = InMemoryToolRegistry()
    tool = _TestTool("mcp_shaped_tool")
    registration = registry.register(tool, category="research")
    discovery = ToolDiscoveryService(registry)

    assert registration.definition.metadata is None
    assert discovery.discover(ToolDiscoveryScope(categories={"research"})) == [registration]
    assert registration.definition.metadata is None


def test_discovery_scope_is_immutable_and_validates_categories() -> None:
    categories = {" Research "}
    scope = ToolDiscoveryScope(categories=categories)
    categories.add("compute")
    assert scope.categories == frozenset({"research"})

    with pytest.raises(AttributeError):
        scope.categories = frozenset({"compute"})  # type: ignore[misc]
    with pytest.raises(TypeError, match="categories must be an iterable"):
        ToolDiscoveryScope(categories="research")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="categories must contain non-empty strings"):
        ToolDiscoveryScope(categories={" "})
    with pytest.raises(ValueError, match="categories must contain non-empty strings"):
        ToolDiscoveryScope(categories={"research", 1})  # type: ignore[arg-type]


def test_discovery_contract_does_not_import_infrastructure_or_mcp() -> None:
    import agent_policy.tool_discovery as discovery_module

    source_path = Path(discovery_module.__file__)
    tree = ast.parse(source_path.read_text())
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

    assert "agent_contracts" in imports
    assert not any(name.startswith("agent_infrastructure") for name in imports)
    assert not any(name == "mcp" or name.startswith("mcp.") for name in imports)


def test_discovery_does_not_require_mcp_sdk_types() -> None:
    from agent_policy.tool_discovery import ToolDiscoveryService as Service

    assert Service is ToolDiscoveryService
    assert ToolDiscoveryService.__module__ == "agent_policy.tool_discovery"
