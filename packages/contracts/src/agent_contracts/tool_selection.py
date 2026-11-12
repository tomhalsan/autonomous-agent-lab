"""Provider-neutral contracts for selecting tools from a discovered catalog."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .tools import ToolRegistration

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]


class ToolSelectionKind(str, Enum):
    """The two valid outcomes of tool selection."""

    NO_TOOL_NEEDED = "no_tool_needed"
    PROPOSED_INVOCATION = "proposed_invocation"


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} object keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}")


@dataclass(frozen=True, init=False)
class ToolInvocationProposal:
    """Selected tool identity and defensively copied candidate JSON arguments."""

    tool_name: str
    _arguments_json: str = field(repr=False)

    def __init__(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise TypeError("arguments must be a dict")
        _validate_json_value(arguments, "arguments")
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(
            self,
            "_arguments_json",
            json.dumps(arguments, allow_nan=False, separators=(",", ":"), sort_keys=True),
        )

    @property
    def arguments(self) -> dict[str, JSONValue]:
        """Return a fresh JSON-compatible copy of candidate arguments."""
        return json.loads(self._arguments_json)


@dataclass(frozen=True)
class ToolSelectionDecision:
    """Explicit no-tool or proposed-invocation selection outcome."""

    kind: ToolSelectionKind
    invocation: ToolInvocationProposal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ToolSelectionKind):
            raise TypeError("kind must be a ToolSelectionKind")
        if self.kind is ToolSelectionKind.NO_TOOL_NEEDED and self.invocation is not None:
            raise ValueError("no-tool decisions must not contain an invocation")
        if self.kind is ToolSelectionKind.PROPOSED_INVOCATION and not isinstance(
            self.invocation, ToolInvocationProposal
        ):
            raise ValueError("proposed-invocation decisions require an invocation")

    @classmethod
    def no_tool_needed(cls) -> "ToolSelectionDecision":
        return cls(kind=ToolSelectionKind.NO_TOOL_NEEDED)

    @classmethod
    def proposed_invocation(
        cls, tool_name: str, arguments: dict[str, Any]
    ) -> "ToolSelectionDecision":
        return cls(
            kind=ToolSelectionKind.PROPOSED_INVOCATION,
            invocation=ToolInvocationProposal(tool_name=tool_name, arguments=arguments),
        )


@runtime_checkable
class ToolSelector(Protocol):
    """Asynchronous, provider-neutral selection boundary."""

    async def select(
        self, goal: str, catalog: Sequence[ToolRegistration]
    ) -> ToolSelectionDecision: ...
