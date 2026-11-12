"""Dynamic category-scoped tool catalog discovery.

Discovery eligibility describes catalog visibility only. It is not authorization to
invoke a tool, and it does not select or execute tools. The reader is queried for
every request, so registrations added to a shared live registry are visible without
recreating this service. The current in-memory registry is process-local.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from agent_contracts import ToolRegistration, ToolRegistryReader


@dataclass(frozen=True)
class ToolDiscoveryScope:
    """Categories of tools the caller wants to discover.

    An empty category set means all currently registered tools, consistent with
    ``ToolRegistryReader.list_all``. Categories are case-insensitive, matching
    ``ToolRegistration.category`` lookup semantics. Categories describe discovery
    interest, not permission grants.
    """

    categories: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.categories, Iterable) or isinstance(self.categories, str):
            raise TypeError("categories must be an iterable of non-empty strings")

        normalized: set[str] = set()
        for category in self.categories:
            if not isinstance(category, str) or not category.strip():
                raise ValueError("categories must contain non-empty strings")
            normalized.add(category.strip().lower())

        object.__setattr__(self, "categories", frozenset(normalized))


class ToolDiscoveryService:
    """Query the live tool registry for the catalog in a discovery scope.

    Discovery does not authorize invocation, rank or select tools, or execute them.
    Agent types and advertised agent capabilities are not permission grants and are
    not used as filters; the current contracts define no truthful mapping from those
    fields to tool eligibility.
    """

    def __init__(self, registry: ToolRegistryReader) -> None:
        self._registry = registry

    def discover(self, scope: ToolDiscoveryScope) -> list[ToolRegistration]:
        """Return registrations matching ``scope`` in registry order.

        The reader is queried on every call. An empty scope returns the current
        complete catalog; an unmatched non-empty scope returns an empty list.
        """
        if not isinstance(scope, ToolDiscoveryScope):
            raise TypeError("scope must be a ToolDiscoveryScope")

        registrations = self._registry.list_all()
        if not scope.categories:
            return registrations

        return [
            registration
            for registration in registrations
            if registration.category is not None
            and registration.category.strip().lower() in scope.categories
        ]
