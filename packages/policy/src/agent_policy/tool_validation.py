"""Bounded validation and correction for proposed tool arguments."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent_contracts import ToolInvocationProposal, ToolRegistration

_SUPPORTED_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})


@dataclass(frozen=True)
class ToolArgumentValidationResult:
    """Outcome of validating candidate arguments against a tool's bounded schema."""

    valid: bool
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.valid and self.errors:
            raise ValueError("valid results must not contain errors")
        if not self.valid and not self.errors:
            raise ValueError("invalid results must contain at least one error")
        if not isinstance(self.errors, tuple) or any(
            not isinstance(error, str) or not error.strip() for error in self.errors
        ):
            raise TypeError("errors must be a tuple of non-empty strings")


class ToolArgumentValidator:
    """Validate object arguments using only the project's bounded tool-schema profile."""

    def validate(
        self, registration: ToolRegistration, arguments: Any
    ) -> ToolArgumentValidationResult:
        if not isinstance(registration, ToolRegistration):
            raise TypeError("registration must be a ToolRegistration")
        errors: list[str] = []
        if not isinstance(arguments, dict):
            return ToolArgumentValidationResult(
                valid=False, errors=("arguments must be a dictionary/object",)
            )

        schema = registration.definition.parameters
        if not isinstance(schema, dict):
            return ToolArgumentValidationResult(
                valid=False, errors=("tool parameters schema must be an object",)
            )
        if schema.get("type", "object") != "object":
            errors.append("tool parameters schema type must be 'object'")

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            errors.append("tool properties schema must be an object")
            properties = {}
        else:
            for name, property_schema in properties.items():
                if not isinstance(name, str) or not name.strip():
                    errors.append("tool property names must be non-empty strings")
                elif not isinstance(property_schema, dict):
                    errors.append(f"schema for argument {name!r} must be an object")
                else:
                    self._validate_schema_type(name, property_schema, errors)

        required = schema.get("required", [])
        if not isinstance(required, list) or any(
            not isinstance(name, str) or not name.strip() for name in required
        ):
            errors.append("tool required schema must be a list of non-empty strings")
            required_names: set[str] = set()
        else:
            required_names = set(required)
            if len(required_names) != len(required):
                errors.append("tool required schema must not contain duplicates")

        for name in required_names - arguments.keys():
            errors.append(f"Missing required argument {name!r}")

        additional_properties = schema.get("additionalProperties", True)
        if not isinstance(additional_properties, (bool, dict)):
            errors.append("additionalProperties must be a boolean or schema object")
            additional_properties = True
        elif isinstance(additional_properties, dict):
            self._validate_schema_type("additional properties", additional_properties, errors)

        for name, value in arguments.items():
            if not isinstance(name, str):
                errors.append("argument names must be strings")
                continue
            property_schema = properties.get(name)
            if property_schema is None:
                if additional_properties is False:
                    errors.append(f"Unexpected argument {name!r}")
                elif isinstance(additional_properties, dict):
                    self._validate_declared_type(name, value, additional_properties, errors)
                self._validate_json_value(name, value, errors)
                continue
            if not isinstance(property_schema, dict):
                errors.append(f"schema for argument {name!r} must be an object")
                continue
            self._validate_declared_type(name, value, property_schema, errors)
            self._validate_json_value(name, value, errors)

        return ToolArgumentValidationResult(
            valid=not errors,
            errors=tuple(errors),
        )

    @staticmethod
    def _validate_declared_type(
        name: str, value: Any, schema: Mapping[str, Any], errors: list[str]
    ) -> None:
        if "type" not in schema:
            return
        declared_type = schema["type"]
        if not ToolArgumentValidator._validate_schema_type(name, schema, errors):
            return

        matches = {
            "array": isinstance(value, list),
            "boolean": isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "null": value is None,
            "number": (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and (not isinstance(value, float) or math.isfinite(value))
            ),
            "object": isinstance(value, dict),
            "string": isinstance(value, str),
        }[declared_type]
        if not matches:
            errors.append(f"argument {name!r} must be of type {declared_type}")

    @staticmethod
    def _validate_schema_type(name: str, schema: Mapping[str, Any], errors: list[str]) -> bool:
        if "type" not in schema:
            return True
        declared_type = schema["type"]
        if isinstance(declared_type, str) and declared_type in _SUPPORTED_TYPES:
            return True
        errors.append(f"schema for argument {name!r} declares an unsupported type")
        return False

    @classmethod
    def _validate_json_value(cls, name: str, value: Any, errors: list[str]) -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                errors.append(f"argument {name!r} must not contain non-finite numbers")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                cls._validate_json_value(f"{name}[{index}]", item, errors)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    errors.append(f"argument {name!r} object keys must be strings")
                    continue
                cls._validate_json_value(f"{name}.{key}", item, errors)
            return
        errors.append(f"argument {name!r} contains a non-JSON value of type {type(value).__name__}")


@runtime_checkable
class ToolArgumentCorrector(Protocol):
    """Bounded seam for correcting an invalid proposal before execution."""

    async def correct(
        self,
        registration: ToolRegistration,
        proposal: ToolInvocationProposal,
        validation: ToolArgumentValidationResult,
    ) -> ToolInvocationProposal | None: ...


class DeterministicToolArgumentCorrector:
    """Remove only undeclared optional values forbidden by additionalProperties=false."""

    async def correct(
        self,
        registration: ToolRegistration,
        proposal: ToolInvocationProposal,
        validation: ToolArgumentValidationResult,
    ) -> ToolInvocationProposal | None:
        if validation.valid:
            return None

        schema = registration.definition.parameters
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        arguments = proposal.arguments

        if (
            schema.get("additionalProperties") is not False
            or not isinstance(properties, dict)
            or not isinstance(required, list)
            or any(not isinstance(name, str) or not name.strip() for name in required)
        ):
            return None

        if not set(required).issubset(arguments):
            return None
        unexpected = {name for name in arguments if name not in properties}
        if not unexpected:
            return None
        unexpected_errors = {f"Unexpected argument {name!r}" for name in unexpected}
        if (
            len(validation.errors) != len(unexpected_errors)
            or set(validation.errors) != unexpected_errors
        ):
            return None

        corrected = {name: value for name, value in arguments.items() if name in properties}
        return ToolInvocationProposal(proposal.tool_name, corrected)
