import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ToolErrorCode(str, Enum):
    """Semantic error categories for tool execution failures."""

    INVALID_ARGUMENTS = "invalid_arguments"
    UNAVAILABLE = "unavailable"
    EXECUTION_FAILURE = "execution_failure"
    TIMEOUT = "timeout"


class ToolOperationType(str, Enum):
    """Execution side-effect nature of a tool operation."""

    READ_ONLY = "read_only"
    STATE_MUTATING = "state_mutating"


class ToolRiskLevel(str, Enum):
    """Descriptive risk classification for tool execution."""

    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"


class ToolIdempotency(str, Enum):
    """Idempotency guarantees under repeated invocations with identical arguments."""

    IDEMPOTENT = "idempotent"
    NON_IDEMPOTENT = "non_idempotent"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


def _parse_enum(value: Any, enum_cls: type[Any], field_name: str) -> Any:
    """Parse an enum member or case-insensitive string, rejecting invalid types and values."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, Enum) or not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a {enum_cls.__name__} or string, got {type(value).__name__}"
        )
    normalized = value.strip().lower()
    try:
        return enum_cls(normalized)
    except ValueError as exc:
        valid = [m.value for m in enum_cls]
        raise ValueError(
            f"Unknown {field_name} {value!r}. Expected one of {valid}"
        ) from exc


@dataclass(frozen=True)
class ToolCapabilityMetadata:
    """Descriptive capability metadata declaring a tool's operational characteristics."""

    category: str
    operation_type: ToolOperationType
    risk_level: ToolRiskLevel
    idempotency: ToolIdempotency

    def __post_init__(self) -> None:
        if not isinstance(self.category, str):
            raise TypeError(f"category must be a string, got {type(self.category).__name__}")
        normalized_category = self.category.strip()
        if not normalized_category:
            raise ValueError("category must be a non-empty string")
        object.__setattr__(self, "category", normalized_category)

        parsed_op = _parse_enum(self.operation_type, ToolOperationType, "operation_type")
        object.__setattr__(self, "operation_type", parsed_op)

        parsed_risk = _parse_enum(self.risk_level, ToolRiskLevel, "risk_level")
        object.__setattr__(self, "risk_level", parsed_risk)

        parsed_idem = _parse_enum(self.idempotency, ToolIdempotency, "idempotency")
        object.__setattr__(self, "idempotency", parsed_idem)

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "operation_type": self.operation_type.value,
            "risk_level": self.risk_level.value,
            "idempotency": self.idempotency.value,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCapabilityMetadata":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(
            category=data["category"],
            operation_type=data["operation_type"],
            risk_level=data["risk_level"],
            idempotency=data["idempotency"],
        )

    @classmethod
    def from_json(cls, value: str) -> "ToolCapabilityMetadata":
        return cls.from_dict(json.loads(value))


def _validate_tool_schema(schema: dict[str, Any], schema_name: str) -> None:
    """Validate a schema dictionary against the Tiny Agent Lab tool-schema profile."""
    if not isinstance(schema, dict):
        if schema_name == "output_schema":
            raise TypeError("output_schema must be a dict if provided")
        raise TypeError(f"{schema_name} must be a dict")

    if not schema:
        return

    if "type" in schema:
        raw_type = schema["type"]
        if not isinstance(raw_type, str):
            raise ValueError(f"{schema_name}['type'] must be a string")
        if schema_name == "parameters":
            if raw_type != "object":
                raise ValueError(f"{schema_name}['type'] must be 'object' for tool invocation")
        else:
            valid_types = {"object", "array", "string", "number", "integer", "boolean", "null"}
            if raw_type not in valid_types:
                raise ValueError(
                    f"{schema_name}['type'] must be one of {sorted(valid_types)}"
                )

    if "properties" in schema:
        props = schema["properties"]
        if not isinstance(props, dict):
            raise ValueError(f"{schema_name}['properties'] must be a dict")
        for prop_name, prop_def in props.items():
            if not isinstance(prop_name, str) or not prop_name.strip():
                raise ValueError(f"{schema_name}['properties'] keys must be non-empty strings")
            if not isinstance(prop_def, dict):
                raise TypeError(
                    f"{schema_name}['properties'][{prop_name!r}] must be a dict"
                )

    if "required" in schema:
        raw_required = schema["required"]
        if not isinstance(raw_required, list):
            raise TypeError(
                f"{schema_name}['required'] must be a list of unique, non-empty strings"
            )
        seen: set[str] = set()
        for item in raw_required:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"{schema_name}['required'] must contain non-empty strings"
                )
            if item in seen:
                raise ValueError(
                    f"{schema_name}['required'] must contain unique strings"
                )
            seen.add(item)

    if "additionalProperties" in schema:
        add_props = schema["additionalProperties"]
        if not isinstance(add_props, (bool, dict)):
            raise ValueError(f"{schema_name}['additionalProperties'] must be a bool or dict")


@dataclass(frozen=True)
class ToolError:
    """Standardized failure description for tool operations."""

    code: ToolErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("message must not be empty")

        if isinstance(self.code, str) and not isinstance(self.code, ToolErrorCode):
            try:
                object.__setattr__(self, "code", ToolErrorCode(self.code.lower()))
            except ValueError as exc:
                valid = [c.value for c in ToolErrorCode]
                raise ValueError(
                    f"Unknown tool error code {self.code!r}. Expected one of {valid}"
                ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolError":
        raw_code = data["code"]
        code = raw_code if isinstance(raw_code, ToolErrorCode) else ToolErrorCode(str(raw_code))
        return cls(
            code=code,
            message=str(data["message"]),
            details=dict(data.get("details", {})),
        )

    @classmethod
    def from_json(cls, value: str) -> "ToolError":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class ToolDefinition:
    """Machine-readable metadata and schemas defining a tool's capability."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    metadata: ToolCapabilityMetadata | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tool name must not be empty")
        if not self.description.strip():
            raise ValueError("Tool description must not be empty")
        _validate_tool_schema(self.parameters, "parameters")
        if self.output_schema is not None:
            _validate_tool_schema(self.output_schema, "output_schema")
        if self.metadata is not None and not isinstance(self.metadata, ToolCapabilityMetadata):
            raise TypeError("metadata must be a ToolCapabilityMetadata instance if provided")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }
        if self.output_schema is not None:
            data["output_schema"] = dict(self.output_schema)
        if self.metadata is not None:
            data["metadata"] = self.metadata.to_dict()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolDefinition":
        raw_metadata = data.get("metadata")
        metadata = (
            ToolCapabilityMetadata.from_dict(raw_metadata)
            if raw_metadata is not None
            else None
        )
        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            parameters=dict(data.get("parameters", {})),
            output_schema=(
                dict(data["output_schema"])
                if data.get("output_schema") is not None
                else None
            ),
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, value: str) -> "ToolDefinition":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class ToolInvocationRequest:
    """Request payload sent to invoke a tool."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")
        if not self.call_id.strip():
            raise ValueError("call_id must not be empty")
        if not isinstance(self.arguments, dict):
            raise TypeError("arguments must be a dict")

    @classmethod
    def create(
        cls,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> "ToolInvocationRequest":
        return cls(
            tool_name=tool_name,
            arguments=arguments if arguments is not None else {},
            call_id=call_id if call_id is not None else uuid.uuid4().hex,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolInvocationRequest":
        return cls(
            tool_name=str(data["tool_name"]),
            arguments=dict(data.get("arguments", {})),
            call_id=str(data.get("call_id", uuid.uuid4().hex)),
        )

    @classmethod
    def from_json(cls, value: str) -> "ToolInvocationRequest":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class ToolInvocationResult:
    """Result payload returned by a tool execution."""

    call_id: str
    tool_name: str
    success: bool
    output: Any | None = None
    error: ToolError | None = None

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("call_id must not be empty")
        if not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")
        if self.success and self.error is not None:
            raise ValueError("error must be None when success is True")
        if not self.success and self.error is None:
            raise ValueError("error must be provided when success is False")

    @classmethod
    def create_success(
        cls,
        call_id: str,
        tool_name: str,
        output: Any,
    ) -> "ToolInvocationResult":
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            success=True,
            output=output,
            error=None,
        )

    @classmethod
    def create_failure(
        cls,
        call_id: str,
        tool_name: str,
        error: ToolError | str,
        code: ToolErrorCode = ToolErrorCode.EXECUTION_FAILURE,
        details: dict[str, Any] | None = None,
    ) -> "ToolInvocationResult":
        tool_error = (
            error
            if isinstance(error, ToolError)
            else ToolError(code=code, message=str(error), details=details or {})
        )
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            success=False,
            output=None,
            error=tool_error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error.to_dict() if self.error is not None else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolInvocationResult":
        raw_error = data.get("error")
        error = ToolError.from_dict(raw_error) if raw_error is not None else None
        return cls(
            call_id=str(data["call_id"]),
            tool_name=str(data["tool_name"]),
            success=bool(data["success"]),
            output=data.get("output"),
            error=error,
        )

    @classmethod
    def from_json(cls, value: str) -> "ToolInvocationResult":
        return cls.from_dict(json.loads(value))


@runtime_checkable
class Tool(Protocol):
    """Abstract capability interface for invokable tools."""

    @property
    def definition(self) -> ToolDefinition: ...

    async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult: ...


@dataclass(frozen=True)
class ToolRegistration:
    """Immutable registration of a tool within a capability category."""

    tool: Tool
    category: str | None = None
    _required_inputs: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tool, Tool):
            raise TypeError("tool must implement Tool protocol")

        meta = self.tool.definition.metadata
        if meta is not None:
            if self.category is not None:
                if not isinstance(self.category, str) or not self.category.strip():
                    raise ValueError("category must be a non-empty string")
                if self.category.strip().lower() != meta.category.strip().lower():
                    raise ValueError(
                        f"Registration category '{self.category}' conflicts with tool capability category '{meta.category}'"
                    )
            object.__setattr__(self, "category", meta.category)
        else:
            if self.category is None:
                raise ValueError("category must be provided when tool metadata is unclassified")
            if not isinstance(self.category, str) or not self.category.strip():
                raise ValueError("category must be a non-empty string")
            object.__setattr__(self, "category", self.category.strip())

        params = self.tool.definition.parameters
        if "required" in params:
            object.__setattr__(self, "_required_inputs", frozenset(params["required"]))
        else:
            object.__setattr__(self, "_required_inputs", frozenset())

    @property
    def name(self) -> str:
        return self.tool.definition.name

    @property
    def definition(self) -> ToolDefinition:
        return self.tool.definition

    @property
    def required_inputs(self) -> frozenset[str]:
        return self._required_inputs


@runtime_checkable
class ToolRegistryReader(Protocol):
    """Read-only query interface for inspecting registered tools."""

    def get(self, name: str) -> ToolRegistration | None: ...

    def find_by_category(self, category: str) -> list[ToolRegistration]: ...

    def find_by_required_inputs(
        self, required_inputs: set[str] | list[str] | tuple[str, ...] | frozenset[str]
    ) -> list[ToolRegistration]: ...

    def list_all(self) -> list[ToolRegistration]: ...


@runtime_checkable
class ToolRegistry(ToolRegistryReader, Protocol):
    """Runtime registry interface for managing tool lifecycle."""

    def register(self, tool: Tool, category: str | None = None) -> ToolRegistration: ...

    def unregister(self, name: str) -> ToolRegistration | None: ...
