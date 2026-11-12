"""Provider-neutral deterministic reference implementation of tool selection."""

import re
from collections.abc import Sequence

from agent_contracts import (
    ToolRegistration,
    ToolSelectionDecision,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_COMMON_TASK_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "can",
        "could",
        "do",
        "for",
        "get",
        "i",
        "me",
        "my",
        "of",
        "on",
        "please",
        "the",
        "to",
        "want",
        "with",
        "you",
    }
)


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token for token in _TOKEN_PATTERN.findall(value.lower()) if token not in _COMMON_TASK_WORDS
    }


class DeterministicToolSelector:
    """Select a uniquely best lexical match, or explicitly return no-tool."""

    async def select(self, goal: str, catalog: Sequence[ToolRegistration]) -> ToolSelectionDecision:
        if not isinstance(goal, str):
            raise TypeError("goal must be a string")
        if not isinstance(catalog, Sequence) or isinstance(catalog, (str, bytes)):
            raise TypeError("catalog must be a sequence of ToolRegistration values")
        if any(not isinstance(registration, ToolRegistration) for registration in catalog):
            raise TypeError("catalog items must be ToolRegistration values")

        goal_tokens = _meaningful_tokens(goal)
        if not goal_tokens:
            return ToolSelectionDecision.no_tool_needed()

        scored: list[tuple[int, ToolRegistration]] = []
        for registration in catalog:
            definition = registration.definition
            descriptive_tokens = _meaningful_tokens(f"{definition.name} {definition.description}")
            matching_descriptors = goal_tokens & descriptive_tokens
            if not matching_descriptors:
                continue
            category_tokens = _meaningful_tokens(registration.category or "")
            score = len(matching_descriptors) + len(goal_tokens & category_tokens)
            if score:
                scored.append((score, registration))

        if not scored:
            return ToolSelectionDecision.no_tool_needed()

        best_score = max(score for score, _ in scored)
        best_matches = [registration for score, registration in scored if score == best_score]
        if len(best_matches) != 1:
            return ToolSelectionDecision.no_tool_needed()

        selected = best_matches[0]
        arguments = self._construct_arguments(goal, selected)
        return ToolSelectionDecision.proposed_invocation(selected.name, arguments)

    @staticmethod
    def _construct_arguments(goal: str, registration: ToolRegistration) -> dict[str, str]:
        """Use the goal only when one required string input makes that mapping clear."""
        parameters = registration.definition.parameters
        required = parameters.get("required", [])
        properties = parameters.get("properties", {})
        if not isinstance(required, list) or not isinstance(properties, dict):
            return {}
        if len(required) != 1 or not isinstance(required[0], str):
            return {}

        property_schema = properties.get(required[0])
        if not isinstance(property_schema, dict) or property_schema.get("type") != "string":
            return {}
        if required[0].lower() not in {"prompt", "query", "question", "text"}:
            return {}
        if not goal.strip():
            return {}
        return {required[0]: goal.strip()}
