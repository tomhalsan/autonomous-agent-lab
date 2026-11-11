from agent_contracts import (
    Tool,
    ToolRegistration,
)


class InMemoryToolRegistry:
    """In-memory implementation of the ToolRegistry protocol."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolRegistration] = {}

    def register(self, tool: Tool, category: str) -> ToolRegistration:
        """Register a tool with an associated capability category.

        Raises:
            TypeError: If tool does not implement Tool protocol.
            ValueError: If tool name is already registered, if category is empty,
                or if tool parameters['required'] is malformed.
        """
        registration = ToolRegistration(tool=tool, category=category)
        if registration.name in self._tools:
            raise ValueError(f"Tool with name '{registration.name}' is already registered")
        self._tools[registration.name] = registration
        return registration

    def unregister(self, name: str) -> ToolRegistration | None:
        """Remove a tool by exact name.

        Returns the removed ToolRegistration, or None if not found.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        return self._tools.pop(name, None)

    def get(self, name: str) -> ToolRegistration | None:
        """Retrieve a tool registration by exact name, or None if not found."""
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        return self._tools.get(name)

    def find_by_category(self, category: str) -> list[ToolRegistration]:
        """Find all registered tools in the specified category (case-insensitive).

        Returns a new list snapshot in registration order.
        """
        if not isinstance(category, str) or not category.strip():
            raise ValueError("category must be a non-empty string")
        target_category = category.strip().lower()
        return [
            reg for reg in self._tools.values() if reg.category.strip().lower() == target_category
        ]

    def find_by_required_inputs(
        self, required_inputs: set[str] | list[str] | tuple[str, ...] | frozenset[str]
    ) -> list[ToolRegistration]:
        """Find registered tools whose required inputs contain all requested inputs.

        Returns a new list snapshot in registration order.
        """
        if not isinstance(required_inputs, (set, list, tuple, frozenset)):
            raise TypeError("required_inputs must be a collection of strings")
        if not required_inputs:
            raise ValueError("required_inputs must contain at least one input")
        req_set: set[str] = set()
        for item in required_inputs:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("required_inputs must contain non-empty strings")
            req_set.add(item)
        return [reg for reg in self._tools.values() if reg.required_inputs.issuperset(req_set)]

    def list_all(self) -> list[ToolRegistration]:
        """Return all registered tools as a new list snapshot in registration order."""
        return list(self._tools.values())
