"""Bounded tool argument validation tests."""

import pytest
from agent_contracts import (
    ToolDefinition,
    ToolInvocationProposal,
    ToolInvocationRequest,
    ToolInvocationResult,
    ToolRegistration,
)
from agent_policy import (
    DeterministicToolArgumentCorrector,
    ToolArgumentValidationResult,
    ToolArgumentValidator,
)


class _ValidationTool:
    def __init__(self, parameters: dict[str, object]) -> None:
        self._definition = ToolDefinition(
            name="validation_tool",
            description="Tool used to test argument validation",
            parameters=parameters,
        )
        self.requests: list[ToolInvocationRequest] = []

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        self.requests.append(request)
        return ToolInvocationResult.create_success(
            call_id=request.call_id, tool_name=request.tool_name, output=None
        )


def _registration(parameters: dict[str, object]) -> tuple[ToolRegistration, _ValidationTool]:
    tool = _ValidationTool(parameters)
    return ToolRegistration(tool, category="test"), tool


def test_validator_accepts_valid_required_arguments_and_json_values() -> None:
    registration, tool = _registration(
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "filters": {"type": "object"},
                "labels": {"type": "array"},
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        }
    )

    result = ToolArgumentValidator().validate(
        registration,
        {
            "query": "weather",
            "limit": 4,
            "filters": {"region": "north"},
            "labels": ["current"],
        },
    )

    assert result.valid
    assert result.errors == ()
    assert tool.requests == []


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "Missing required argument 'query'"),
        ({"query": "x", "extra": 1}, "Unexpected argument 'extra'"),
        ({"query": 3}, "argument 'query' must be of type string"),
        (["not", "an", "object"], "arguments must be a dictionary/object"),
    ],
)
def test_validator_rejects_missing_extra_wrong_type_and_non_object_arguments(
    arguments: object, message: str
) -> None:
    registration, _ = _registration(
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }
    )

    result = ToolArgumentValidator().validate(registration, arguments)

    assert not result.valid
    assert any(message in error for error in result.errors)


def test_validator_rejects_unsupported_declared_property_types() -> None:
    registration, _ = _registration(
        {
            "type": "object",
            "properties": {"value": {"type": "custom"}},
        }
    )

    result = ToolArgumentValidator().validate(registration, {})

    assert not result.valid
    assert any("unsupported type" in error for error in result.errors)


def test_validator_does_not_treat_boolean_as_integer() -> None:
    registration, _ = _registration(
        {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
            "required": ["limit"],
        }
    )

    result = ToolArgumentValidator().validate(registration, {"limit": True})

    assert not result.valid
    assert "argument 'limit' must be of type integer" in result.errors


@pytest.mark.anyio
async def test_corrector_removes_undeclared_arguments_when_they_are_the_only_errors() -> None:
    registration, _ = _registration(
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }
    )
    proposal = ToolInvocationProposal(
        "validation_tool", {"query": "weather", "unexpected": "discard me"}
    )
    validation = ToolArgumentValidationResult(
        valid=False, errors=("Unexpected argument 'unexpected'",)
    )

    corrected = await DeterministicToolArgumentCorrector().correct(
        registration, proposal, validation
    )

    assert corrected is not None
    assert corrected.arguments == {"query": "weather"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("required", "arguments", "errors"),
    [
        (
            ["query"],
            {"query": 3, "unexpected": "discard me"},
            (
                "Unexpected argument 'unexpected'",
                "argument 'query' must be of type string",
            ),
        ),
        (
            ["query", "limit"],
            {"query": "weather", "unexpected": "discard me"},
            (
                "Unexpected argument 'unexpected'",
                "Missing required argument 'limit'",
            ),
        ),
    ],
)
async def test_corrector_refuses_extra_arguments_with_independent_errors(
    required: list[str],
    arguments: dict[str, object],
    errors: tuple[str, ...],
) -> None:
    registration, _ = _registration(
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": required,
            "additionalProperties": False,
        }
    )
    proposal = ToolInvocationProposal("validation_tool", arguments)
    validation = ToolArgumentValidationResult(valid=False, errors=errors)

    corrected = await DeterministicToolArgumentCorrector().correct(
        registration, proposal, validation
    )

    assert corrected is None
