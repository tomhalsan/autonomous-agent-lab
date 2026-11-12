from dataclasses import FrozenInstanceError

import pytest
from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    FallbackTransition,
    InferenceExecutionTelemetry,
    InferenceResponse,
    TaskRequested,
    TaskResult,
    Tool,
    ToolCapabilityMetadata,
    ToolDefinition,
    ToolError,
    ToolErrorCode,
    ToolExecutionTelemetry,
    ToolIdempotency,
    ToolInvocationRequest,
    ToolInvocationResult,
    ToolOperationType,
    ToolRiskLevel,
)


def test_task_message_round_trip() -> None:
    original = TaskRequested.create("hello")
    restored = TaskRequested.from_json(original.to_json())
    assert restored == original


def test_agent_event_round_trip() -> None:
    original = AgentEvent.create("worker-1", "working", "Processing prompt", task_id="task-123")
    restored = AgentEvent.from_json(original.to_json())
    assert restored == original
    assert restored.task_id == "task-123"


def test_agent_event_to_dict() -> None:
    event = AgentEvent.create("worker-1", "idle", "Waiting")
    data = event.to_dict()
    assert data["agent_id"] == "worker-1"
    assert data["state"] == "idle"
    assert data["task_id"] is None


def test_tool_execution_telemetry_event_round_trip_without_state() -> None:
    telemetry = ToolExecutionTelemetry(
        tool_name="web_search",
        agent_id="worker-1",
        call_id="call-123",
        duration_ms=12.5,
        argument_size_bytes=19,
        result_size_bytes=32,
        outcome="failed",
        task_id="task-123",
        started_at="2026-11-11T12:00:00+00:00",
        error_code=ToolErrorCode.EXECUTION_FAILURE,
    )
    event = AgentEvent.create(
        agent_id="worker-1",
        state=None,
        detail="Tool execution telemetry",
        task_id="task-123",
        tool_execution_telemetry=telemetry,
    )

    data = event.to_dict()
    assert "state" not in data
    assert data["tool_execution_telemetry"] == telemetry.to_dict()
    assert AgentEvent.from_json(event.to_json()) == event

    with pytest.raises(ValueError, match="events without state require tool execution telemetry"):
        AgentEvent.create("worker-1", None, "State-less event")


def test_agent_capability_metadata_round_trip() -> None:
    original = AgentCapabilityMetadata(
        agent_id="research-agent-1",
        capabilities=["research"],
        agent_type="research-agent",
        providers={"research_executor": "tavily", "inference_provider": "cloudflare"},
    )
    restored = AgentCapabilityMetadata.from_json(original.to_json())
    assert restored == original
    assert restored.agent_id == "research-agent-1"
    assert restored.capabilities == ["research"]
    assert restored.agent_type == "research-agent"
    assert restored.providers["research_executor"] == "tavily"
    assert restored.providers["inference_provider"] == "cloudflare"


def test_agent_capability_metadata_rejects_empty_agent_id() -> None:
    with pytest.raises(ValueError, match="agent_id must not be empty"):
        AgentCapabilityMetadata(agent_id="   ", capabilities=["general"])


@pytest.mark.parametrize(
    "bad_key",
    ["api_key", "tavily_api_key", "token", "auth_token", "account_id", "secret", "user_password", "credential"],
)
def test_agent_capability_metadata_rejects_sensitive_keys(bad_key: str) -> None:
    with pytest.raises(ValueError, match="must not be exposed in provider metadata"):
        AgentCapabilityMetadata(
            agent_id="agent-1",
            capabilities=["research"],
            providers={bad_key: "confidential_value"},
        )


def test_task_result_round_trip() -> None:
    original = TaskResult(
        task_id="task-general-1",
        output="Completed general task execution",
        status="completed",
        acknowledged=True,
    )
    restored = TaskResult.from_json(original.to_json())
    assert restored == original
    assert restored.task_id == "task-general-1"
    assert restored.output == "Completed general task execution"
    assert restored.status == "completed"
    assert restored.acknowledged is True


def test_task_result_rejects_empty_task_id() -> None:
    with pytest.raises(ValueError, match="task_id must not be empty"):
        TaskResult(task_id="   ", output="some output")


def test_task_result_rejects_empty_status() -> None:
    with pytest.raises(ValueError, match="status must not be empty"):
        TaskResult(task_id="task-123", output="some output", status="   ")


# --- Fallback Transition Contracts Tests (Issue #16 Increment 3) ---


def test_fallback_transition_attributes_and_normalization() -> None:
    transition = FallbackTransition(
        failed_provider="  CLOUDFLARE  ",
        target_provider="  GEMINI  ",
        reason="  rate_limited  ",
        attempt_index=0,
    )
    assert transition.failed_provider == "cloudflare"
    assert transition.target_provider == "gemini"
    assert transition.reason == "rate_limited"
    assert transition.attempt_index == 0


def test_fallback_transition_validation() -> None:
    with pytest.raises(ValueError, match="failed_provider"):
        FallbackTransition(failed_provider="  ", target_provider="gemini", reason="timeout", attempt_index=0)

    with pytest.raises(ValueError, match="target_provider"):
        FallbackTransition(failed_provider="cloudflare", target_provider="", reason="timeout", attempt_index=0)

    with pytest.raises(ValueError, match="reason"):
        FallbackTransition(failed_provider="cloudflare", target_provider="gemini", reason="  ", attempt_index=0)

    with pytest.raises(ValueError, match="attempt_index"):
        FallbackTransition(failed_provider="cloudflare", target_provider="gemini", reason="timeout", attempt_index=-1)

    with pytest.raises(ValueError, match="attempt_index"):
        FallbackTransition(failed_provider="cloudflare", target_provider="gemini", reason="timeout", attempt_index=True)  # type: ignore[arg-type]


def test_fallback_transition_to_dict_and_from_dict() -> None:
    original = FallbackTransition(
        failed_provider="cloudflare",
        target_provider="gemini",
        reason="timeout",
        attempt_index=1,
    )
    data = original.to_dict()
    assert data == {
        "failed_provider": "cloudflare",
        "target_provider": "gemini",
        "reason": "timeout",
        "attempt_index": 1,
    }
    restored = FallbackTransition.from_dict(data)
    assert restored == original


def test_agent_event_carries_fallback_transition() -> None:
    transition = FallbackTransition(
        failed_provider="cloudflare",
        target_provider="gemini",
        reason="rate_limited",
        attempt_index=0,
    )
    event = AgentEvent.create(
        agent_id="worker-1",
        state="fallback",
        detail="Inference fallback: cloudflare -> gemini (rate_limited)",
        task_id="task-fb-1",
        fallback_transition=transition,
    )
    assert event.fallback_transition == transition
    assert event.state == "fallback"
    assert event.task_id == "task-fb-1"

    data = event.to_dict()
    assert "fallback_transition" in data
    assert data["fallback_transition"] == transition.to_dict()


def test_agent_event_serialization_round_trip_with_fallback_transition() -> None:
    transition = FallbackTransition(
        failed_provider="cloudflare",
        target_provider="gemini",
        reason="timeout",
        attempt_index=1,
    )
    event = AgentEvent.create(
        agent_id="worker-1",
        state="fallback",
        detail="Inference fallback: cloudflare -> gemini (timeout)",
        task_id="task-fb-2",
        fallback_transition=transition,
    )
    json_str = event.to_json()
    restored = AgentEvent.from_json(json_str)
    assert restored == event
    assert isinstance(restored.fallback_transition, FallbackTransition)
    assert restored.fallback_transition.failed_provider == "cloudflare"
    assert restored.fallback_transition.target_provider == "gemini"
    assert restored.fallback_transition.reason == "timeout"
    assert restored.fallback_transition.attempt_index == 1


def test_agent_event_omits_fallback_transition_when_none() -> None:
    event = AgentEvent.create(
        agent_id="worker-1",
        state="idle",
        detail="Waiting for work",
    )
    assert event.fallback_transition is None
    data = event.to_dict()
    assert "fallback_transition" not in data


def test_agent_event_rejects_invalid_fallback_transition_type() -> None:
    with pytest.raises(TypeError, match="fallback_transition must be FallbackTransition"):
        AgentEvent(
            agent_id="worker-1",
            state="fallback",
            detail="detail",
            occurred_at="2026-11-09T00:00:00Z",
            fallback_transition="invalid-transition",  # type: ignore[arg-type]
        )


# --- Tool Capability Contracts Tests ---


def test_tool_definition_valid_round_trip() -> None:
    param_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }
    output_schema = {
        "type": "object",
        "properties": {"results": {"type": "array"}},
    }
    tool_def = ToolDefinition(
        name="web_search",
        description="Search public web for relevant citations",
        parameters=param_schema,
        output_schema=output_schema,
    )

    restored = ToolDefinition.from_json(tool_def.to_json())
    assert restored == tool_def
    assert restored.name == "web_search"
    assert restored.description == "Search public web for relevant citations"
    assert restored.parameters == param_schema
    assert restored.output_schema == output_schema


def test_tool_definition_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="Tool name must not be empty"):
        ToolDefinition(name="   ", description="Executes calculation")


def test_tool_definition_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="Tool description must not be empty"):
        ToolDefinition(name="calc", description="   ")


def test_tool_definition_rejects_invalid_schema_types() -> None:
    with pytest.raises(TypeError, match="parameters must be a dict"):
        ToolDefinition(name="calc", description="Valid", parameters="not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="output_schema must be a dict if provided"):
        ToolDefinition(
            name="calc",
            description="Valid",
            parameters={},
            output_schema=["not", "a", "dict"],  # type: ignore[arg-type]
        )


def test_tool_invocation_request_round_trip() -> None:
    request = ToolInvocationRequest(
        tool_name="web_search",
        arguments={"query": "quantum computing", "max_results": 3},
        call_id="call-42",
    )

    restored = ToolInvocationRequest.from_json(request.to_json())
    assert restored == request
    assert restored.call_id == "call-42"
    assert restored.tool_name == "web_search"
    assert restored.arguments == {"query": "quantum computing", "max_results": 3}


def test_tool_invocation_request_create_auto_call_id() -> None:
    req = ToolInvocationRequest.create("weather_lookup", {"city": "Seattle"})
    assert req.tool_name == "weather_lookup"
    assert req.arguments == {"city": "Seattle"}
    assert isinstance(req.call_id, str)
    assert len(req.call_id) > 0


def test_tool_invocation_request_rejects_blank_identity() -> None:
    with pytest.raises(ValueError, match="tool_name must not be empty"):
        ToolInvocationRequest(tool_name="   ", arguments={})

    with pytest.raises(ValueError, match="call_id must not be empty"):
        ToolInvocationRequest(tool_name="lookup", call_id="   ", arguments={})

    with pytest.raises(TypeError, match="arguments must be a dict"):
        ToolInvocationRequest(tool_name="lookup", arguments="bad-args")  # type: ignore[arg-type]


def test_tool_invocation_result_success_round_trip() -> None:
    structured_output = {
        "status": "ok",
        "temperature_c": 18.5,
        "conditions": "partly cloudy",
    }
    result = ToolInvocationResult.create_success(
        call_id="call-101",
        tool_name="weather_lookup",
        output=structured_output,
    )

    assert result.success is True
    assert result.error is None
    assert result.output == structured_output

    restored = ToolInvocationResult.from_json(result.to_json())
    assert restored == result
    assert restored.call_id == "call-101"
    assert restored.tool_name == "weather_lookup"
    assert restored.success is True
    assert restored.output == structured_output
    assert restored.error is None


def test_tool_invocation_result_failure_with_error_object() -> None:
    err = ToolError(
        code=ToolErrorCode.TIMEOUT,
        message="Tool execution timed out after 15s",
        details={"timeout_seconds": 15},
    )
    result = ToolInvocationResult.create_failure(
        call_id="call-102",
        tool_name="remote_runner",
        error=err,
    )

    assert result.success is False
    assert result.output is None
    assert result.error == err

    restored = ToolInvocationResult.from_json(result.to_json())
    assert restored == result
    assert restored.call_id == "call-102"
    assert restored.tool_name == "remote_runner"
    assert restored.success is False
    assert restored.error is not None
    assert restored.error.code == ToolErrorCode.TIMEOUT
    assert restored.error.message == "Tool execution timed out after 15s"
    assert restored.error.details == {"timeout_seconds": 15}


def test_tool_invocation_result_failure_from_string_and_code() -> None:
    result = ToolInvocationResult.create_failure(
        call_id="call-103",
        tool_name="database_query",
        error="Connection refused by remote host",
        code=ToolErrorCode.UNAVAILABLE,
        details={"host": "db.internal", "port": 5432},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == ToolErrorCode.UNAVAILABLE
    assert result.error.message == "Connection refused by remote host"
    assert result.error.details == {"host": "db.internal", "port": 5432}


@pytest.mark.parametrize(
    ("code", "expected_value"),
    [
        (ToolErrorCode.INVALID_ARGUMENTS, "invalid_arguments"),
        (ToolErrorCode.UNAVAILABLE, "unavailable"),
        (ToolErrorCode.EXECUTION_FAILURE, "execution_failure"),
        (ToolErrorCode.TIMEOUT, "timeout"),
    ],
)
def test_tool_error_standard_taxonomy_coverage(
    code: ToolErrorCode, expected_value: str
) -> None:
    assert code.value == expected_value
    err = ToolError(code=code, message=f"Error code {expected_value}")
    assert err.code == code
    restored = ToolError.from_json(err.to_json())
    assert restored == err
    assert restored.code == code


def test_tool_error_rejects_blank_message() -> None:
    with pytest.raises(ValueError, match="message must not be empty"):
        ToolError(code=ToolErrorCode.INVALID_ARGUMENTS, message="   ")


def test_tool_error_string_code_coercion() -> None:
    err = ToolError(code="invalid_arguments", message="Missing required param: id")  # type: ignore[arg-type]
    assert err.code == ToolErrorCode.INVALID_ARGUMENTS

    with pytest.raises(ValueError, match="Unknown tool error code 'nonexistent'"):
        ToolError(code="nonexistent", message="bad code")  # type: ignore[arg-type]


def test_tool_invocation_result_validation() -> None:
    err = ToolError(code=ToolErrorCode.EXECUTION_FAILURE, message="Failure")

    with pytest.raises(ValueError, match="error must be None when success is True"):
        ToolInvocationResult(
            call_id="c1",
            tool_name="t1",
            success=True,
            output="ok",
            error=err,
        )

    with pytest.raises(ValueError, match="error must be provided when success is False"):
        ToolInvocationResult(
            call_id="c1",
            tool_name="t1",
            success=False,
            output=None,
            error=None,
        )


def test_tool_protocol_implementation() -> None:
    class EchoTool:
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="echo",
                description="Echo input text",
                parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            )

        async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
            return ToolInvocationResult.create_success(
                call_id=request.call_id,
                tool_name=request.tool_name,
                output={"echo": request.arguments.get("text", "")},
            )

    tool = EchoTool()
    assert isinstance(tool, Tool)
    assert tool.definition.name == "echo"


def test_tool_contracts_have_no_external_dependencies() -> None:
    import agent_contracts.tools as tools_module

    # Must only use Python standard library modules and declare no external dependencies
    module_imports = {
        name
        for name, val in tools_module.__dict__.items()
        if hasattr(val, "__module__") and not name.startswith("_")
    }

    # Ensure no external provider/transport references in module scope
    disallowed_terms = {"httpx", "mcp", "tavily", "cloudflare", "redis"}
    for term in disallowed_terms:
        assert term not in module_imports
        assert term not in tools_module.__dict__


def test_tool_operation_type_enumeration() -> None:
    assert ToolOperationType.READ_ONLY == "read_only"
    assert ToolOperationType.STATE_MUTATING == "state_mutating"

    # Direct enum member
    meta_enum = ToolCapabilityMetadata(
        category="search",
        operation_type=ToolOperationType.READ_ONLY,
        risk_level=ToolRiskLevel.LOW,
        idempotency=ToolIdempotency.IDEMPOTENT,
    )
    assert meta_enum.operation_type == ToolOperationType.READ_ONLY

    # Case-insensitive string
    meta_str = ToolCapabilityMetadata(
        category="search",
        operation_type="READ_ONLY",
        risk_level=ToolRiskLevel.LOW,
        idempotency=ToolIdempotency.IDEMPOTENT,
    )
    assert meta_str.operation_type == ToolOperationType.READ_ONLY

    # Rejection of invalid types with TypeError
    for bad_val in [123, True, False, object(), ToolRiskLevel.LOW]:
        with pytest.raises(TypeError, match="operation_type must be a ToolOperationType or string"):
            ToolCapabilityMetadata(
                category="search",
                operation_type=bad_val,  # type: ignore[arg-type]
                risk_level=ToolRiskLevel.LOW,
                idempotency=ToolIdempotency.IDEMPOTENT,
            )

    # Rejection of unknown string with ValueError
    with pytest.raises(ValueError, match="Unknown operation_type 'invalid_op'"):
        ToolCapabilityMetadata(
            category="search",
            operation_type="invalid_op",
            risk_level=ToolRiskLevel.LOW,
            idempotency=ToolIdempotency.IDEMPOTENT,
        )


def test_tool_risk_level_enumeration() -> None:
    assert ToolRiskLevel.LOW == "low"
    assert ToolRiskLevel.ELEVATED == "elevated"
    assert ToolRiskLevel.HIGH == "high"

    # Direct enum member and case-insensitive string
    meta_enum = ToolCapabilityMetadata(
        category="search",
        operation_type=ToolOperationType.READ_ONLY,
        risk_level=ToolRiskLevel.ELEVATED,
        idempotency=ToolIdempotency.IDEMPOTENT,
    )
    assert meta_enum.risk_level == ToolRiskLevel.ELEVATED

    meta_str = ToolCapabilityMetadata(
        category="search",
        operation_type=ToolOperationType.READ_ONLY,
        risk_level="elevated",
        idempotency=ToolIdempotency.IDEMPOTENT,
    )
    assert meta_str.risk_level == ToolRiskLevel.ELEVATED

    # Rejection of invalid types with TypeError
    for bad_val in [123, True, False, object(), ToolOperationType.READ_ONLY]:
        with pytest.raises(TypeError, match="risk_level must be a ToolRiskLevel or string"):
            ToolCapabilityMetadata(
                category="search",
                operation_type=ToolOperationType.READ_ONLY,
                risk_level=bad_val,  # type: ignore[arg-type]
                idempotency=ToolIdempotency.IDEMPOTENT,
            )

    # Rejection of unknown string with ValueError
    with pytest.raises(ValueError, match="Unknown risk_level 'critical'"):
        ToolCapabilityMetadata(
            category="search",
            operation_type=ToolOperationType.READ_ONLY,
            risk_level="critical",
            idempotency=ToolIdempotency.IDEMPOTENT,
        )


def test_tool_idempotency_enumeration() -> None:
    assert ToolIdempotency.IDEMPOTENT == "idempotent"
    assert ToolIdempotency.NON_IDEMPOTENT == "non_idempotent"
    assert ToolIdempotency.CONDITIONAL == "conditional"
    assert ToolIdempotency.UNKNOWN == "unknown"

    # Distinguish CONDITIONAL from UNKNOWN
    assert ToolIdempotency.CONDITIONAL != ToolIdempotency.UNKNOWN

    # Direct enum and case-insensitive string
    meta_enum = ToolCapabilityMetadata(
        category="search",
        operation_type=ToolOperationType.READ_ONLY,
        risk_level=ToolRiskLevel.LOW,
        idempotency=ToolIdempotency.CONDITIONAL,
    )
    assert meta_enum.idempotency == ToolIdempotency.CONDITIONAL

    meta_str = ToolCapabilityMetadata(
        category="search",
        operation_type=ToolOperationType.READ_ONLY,
        risk_level=ToolRiskLevel.LOW,
        idempotency="Conditional",
    )
    assert meta_str.idempotency == ToolIdempotency.CONDITIONAL

    # Rejection of invalid types with TypeError
    for bad_val in [123, True, False, object(), ToolRiskLevel.LOW]:
        with pytest.raises(TypeError, match="idempotency must be a ToolIdempotency or string"):
            ToolCapabilityMetadata(
                category="search",
                operation_type=ToolOperationType.READ_ONLY,
                risk_level=ToolRiskLevel.LOW,
                idempotency=bad_val,  # type: ignore[arg-type]
            )

    # Rejection of unknown string with ValueError
    with pytest.raises(ValueError, match="Unknown idempotency 'always'"):
        ToolCapabilityMetadata(
            category="search",
            operation_type=ToolOperationType.READ_ONLY,
            risk_level=ToolRiskLevel.LOW,
            idempotency="always",
        )


def test_tool_capability_metadata_category_normalization() -> None:
    # Leading/trailing whitespace stripped while preserving capitalization
    meta = ToolCapabilityMetadata(
        category=" Search Service ",
        operation_type=ToolOperationType.READ_ONLY,
        risk_level=ToolRiskLevel.LOW,
        idempotency=ToolIdempotency.IDEMPOTENT,
    )
    assert meta.category == "Search Service"
    assert meta.to_dict()["category"] == "Search Service"
    assert '"category": "Search Service"' in meta.to_json()

    # Rejects non-string category with TypeError
    for bad_cat in [123, True, None, object()]:
        with pytest.raises(TypeError, match="category must be a string"):
            ToolCapabilityMetadata(
                category=bad_cat,  # type: ignore[arg-type]
                operation_type=ToolOperationType.READ_ONLY,
                risk_level=ToolRiskLevel.LOW,
                idempotency=ToolIdempotency.IDEMPOTENT,
            )

    # Rejects empty or whitespace-only category with ValueError
    for empty_cat in ["", "   "]:
        with pytest.raises(ValueError, match="category must be a non-empty string"):
            ToolCapabilityMetadata(
                category=empty_cat,
                operation_type=ToolOperationType.READ_ONLY,
                risk_level=ToolRiskLevel.LOW,
                idempotency=ToolIdempotency.IDEMPOTENT,
            )


def test_tool_capability_metadata_from_dict_rules() -> None:
    # Valid dict with strings
    data = {
        "category": " Search ",
        "operation_type": "READ_ONLY",
        "risk_level": "LOW",
        "idempotency": "IDEMPOTENT",
    }
    meta = ToolCapabilityMetadata.from_dict(data)
    assert meta.category == "Search"
    assert meta.operation_type == ToolOperationType.READ_ONLY
    assert meta.risk_level == ToolRiskLevel.LOW
    assert meta.idempotency == ToolIdempotency.IDEMPOTENT

    # from_dict rejects non-string category with TypeError
    bad_data_cat = dict(data, category=123)
    with pytest.raises(TypeError, match="category must be a string"):
        ToolCapabilityMetadata.from_dict(bad_data_cat)

    # from_dict rejects non-string / non-enum operation_type with TypeError
    bad_data_op = dict(data, operation_type=999)
    with pytest.raises(TypeError, match="operation_type must be a ToolOperationType or string"):
        ToolCapabilityMetadata.from_dict(bad_data_op)

    # from_dict rejects unknown enum string with ValueError
    bad_data_unknown = dict(data, operation_type="not_an_op")
    with pytest.raises(ValueError, match="Unknown operation_type 'not_an_op'"):
        ToolCapabilityMetadata.from_dict(bad_data_unknown)


def test_tool_capability_metadata_immutability_and_roundtrip() -> None:
    original = ToolCapabilityMetadata(
        category="retrieval",
        operation_type=ToolOperationType.READ_ONLY,
        risk_level=ToolRiskLevel.LOW,
        idempotency=ToolIdempotency.IDEMPOTENT,
    )

    with pytest.raises(FrozenInstanceError):
        original.category = "other"  # type: ignore[misc]

    # JSON roundtrip
    serialized = original.to_json()
    restored = ToolCapabilityMetadata.from_json(serialized)
    assert restored == original
    assert restored.category == "retrieval"
    assert restored.operation_type == ToolOperationType.READ_ONLY
    assert restored.risk_level == ToolRiskLevel.LOW
    assert restored.idempotency == ToolIdempotency.IDEMPOTENT

    # Dict roundtrip
    dict_data = original.to_dict()
    assert dict_data == {
        "category": "retrieval",
        "operation_type": "read_only",
        "risk_level": "low",
        "idempotency": "idempotent",
    }
    assert ToolCapabilityMetadata.from_dict(dict_data) == original


def test_tool_capability_metadata_rejects_empty_category() -> None:
    for bad_cat in ["", "   "]:
        with pytest.raises(ValueError, match="category must be a non-empty string"):
            ToolCapabilityMetadata(
                category=bad_cat,
                operation_type=ToolOperationType.READ_ONLY,
                risk_level=ToolRiskLevel.LOW,
                idempotency=ToolIdempotency.IDEMPOTENT,
            )


def test_orthogonal_operation_and_risk() -> None:
    # Read-only operation with HIGH risk (e.g. confidential security scan)
    high_risk_reader = ToolCapabilityMetadata(
        category="security",
        operation_type=ToolOperationType.READ_ONLY,
        risk_level=ToolRiskLevel.HIGH,
        idempotency=ToolIdempotency.IDEMPOTENT,
    )
    assert high_risk_reader.operation_type == ToolOperationType.READ_ONLY
    assert high_risk_reader.risk_level == ToolRiskLevel.HIGH

    # Mutating operation with LOW risk (e.g. temporary scratchpad write)
    low_risk_mutator = ToolCapabilityMetadata(
        category="scratchpad",
        operation_type=ToolOperationType.STATE_MUTATING,
        risk_level=ToolRiskLevel.LOW,
        idempotency=ToolIdempotency.NON_IDEMPOTENT,
    )
    assert low_risk_mutator.operation_type == ToolOperationType.STATE_MUTATING
    assert low_risk_mutator.risk_level == ToolRiskLevel.LOW


def test_tool_definition_schema_profile_validation() -> None:
    # Valid parameter and output schemas
    valid_def = ToolDefinition(
        name="valid_tool",
        description="A tool with valid schemas",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"items": {"type": "array"}},
        },
    )
    assert valid_def.name == "valid_tool"

    # parameters['type'] must be 'object' for tool invocation if specified
    with pytest.raises(ValueError, match="parameters\\['type'\\] must be 'object'"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"type": "string"},
        )

    # non-string 'type'
    with pytest.raises(ValueError, match="parameters\\['type'\\] must be a string"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"type": 123},  # type: ignore[dict-item]
        )

    # invalid output_schema type
    with pytest.raises(ValueError, match="output_schema\\['type'\\] must be one of"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={},
            output_schema={"type": "invalid_type"},
        )

    # properties must be a dict
    with pytest.raises(ValueError, match="parameters\\['properties'\\] must be a dict"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"properties": "not-a-dict"},  # type: ignore[dict-item]
        )

    # property value must be a dict
    with pytest.raises(TypeError, match="parameters\\['properties'\\]\\['field'\\] must be a dict"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"properties": {"field": "not-a-dict"}},  # type: ignore[dict-item]
        )

    # property key must not be empty string
    with pytest.raises(ValueError, match="parameters\\['properties'\\] keys must be non-empty strings"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"properties": {"": {"type": "string"}}},
        )

    # required must be a list of unique, non-empty strings
    with pytest.raises(TypeError, match="parameters\\['required'\\] must be a list"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"required": "not-a-list"},  # type: ignore[dict-item]
        )

    # Rejects tuple, set, and frozenset
    with pytest.raises(TypeError, match="parameters\\['required'\\] must be a list"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"required": ("query",)},  # type: ignore[dict-item]
        )

    with pytest.raises(TypeError, match="parameters\\['required'\\] must be a list"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"required": {"query"}},  # type: ignore[dict-item]
        )

    with pytest.raises(TypeError, match="parameters\\['required'\\] must be a list"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"required": frozenset({"query"})},  # type: ignore[dict-item]
        )

    with pytest.raises(ValueError, match="parameters\\['required'\\] must contain non-empty strings"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"required": ["query", 123]},  # type: ignore[list-item]
        )

    with pytest.raises(ValueError, match="parameters\\['required'\\] must contain non-empty strings"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"required": ["query", "   "]},
        )

    with pytest.raises(ValueError, match="parameters\\['required'\\] must contain unique strings"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"required": ["query", "query"]},
        )

    # additionalProperties must be bool or dict
    with pytest.raises(ValueError, match="parameters\\['additionalProperties'\\] must be a bool or dict"):
        ToolDefinition(
            name="tool",
            description="desc",
            parameters={"additionalProperties": "not-bool-or-dict"},  # type: ignore[dict-item]
        )


def test_tool_capability_metadata_has_no_transport_or_credential_fields() -> None:
    # Structural assertion: ToolCapabilityMetadata contains zero transport or credential fields
    field_names = set(ToolCapabilityMetadata.__dataclass_fields__.keys())
    assert field_names == {"category", "operation_type", "risk_level", "idempotency"}

    disallowed_substrings = ["host", "url", "port", "token", "key", "secret", "password", "auth", "cred"]
    for field_name in field_names:
        for disallowed in disallowed_substrings:
            assert disallowed not in field_name.lower(), f"Unexpected field {field_name!r}"


def test_tool_definition_allows_legitimate_security_argument_names() -> None:
    # No keyword denylisting on argument property names: tools can accept credentials or account_ids as arguments
    legitimate_tool = ToolDefinition(
        name="credential_validator",
        description="Validates credential syntax without storing",
        parameters={
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "api_key_format": {"type": "string"},
                "token_expiry": {"type": "integer"},
            },
            "required": ["account_id"],
        },
    )
    assert "account_id" in legitimate_tool.parameters["properties"]
    assert "api_key_format" in legitimate_tool.parameters["properties"]


def test_tool_definition_with_composed_metadata_roundtrip() -> None:
    metadata = ToolCapabilityMetadata(
        category="search",
        operation_type=ToolOperationType.READ_ONLY,
        risk_level=ToolRiskLevel.LOW,
        idempotency=ToolIdempotency.IDEMPOTENT,
    )
    tool_def = ToolDefinition(
        name="web_search",
        description="Search web",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        output_schema={"type": "object"},
        metadata=metadata,
    )

    serialized = tool_def.to_json()
    restored = ToolDefinition.from_json(serialized)
    assert restored == tool_def
    assert restored.metadata == metadata
    assert restored.metadata.category == "search"
    assert restored.metadata.operation_type == ToolOperationType.READ_ONLY
    assert restored.metadata.risk_level == ToolRiskLevel.LOW
    assert restored.metadata.idempotency == ToolIdempotency.IDEMPOTENT

    # Backward compatibility: unclassified tool without metadata omits 'metadata' key
    unclassified_def = ToolDefinition(
        name="calc",
        description="Calculator",
    )
    unclassified_dict = unclassified_def.to_dict()
    assert "metadata" not in unclassified_dict
    restored_unclassified = ToolDefinition.from_json(unclassified_def.to_json())
    assert restored_unclassified.metadata is None


# --- InferenceResponse Contract Tests ---


def test_inference_response_round_trip() -> None:
    original = InferenceResponse(
        text="Sample output content",
        provider_name="Cloudflare",
        model_name=" @cf/meta/llama-3.1-8b-instruct ",
        finish_reason=" STOP ",
        input_tokens=12,
        output_tokens=34,
        total_tokens=46,
    )
    assert original.provider_name == "cloudflare"
    assert original.model_name == "@cf/meta/llama-3.1-8b-instruct"
    assert original.finish_reason == "stop"

    restored = InferenceResponse.from_json(original.to_json())
    assert restored == original
    assert restored.text == "Sample output content"
    assert restored.input_tokens == 12
    assert restored.output_tokens == 34
    assert restored.total_tokens == 46


def test_inference_response_preserves_text_exact_whitespace() -> None:
    exact_text = "  Leading and trailing whitespace preserved\n\n- item 1\n"
    resp = InferenceResponse(text=exact_text, provider_name="gemini")
    assert resp.text == exact_text


def test_inference_response_minimal_defaults() -> None:
    resp = InferenceResponse(text="Minimal", provider_name="stub")
    assert resp.text == "Minimal"
    assert resp.provider_name == "stub"
    assert resp.model_name is None
    assert resp.finish_reason is None
    assert resp.input_tokens is None
    assert resp.output_tokens is None
    assert resp.total_tokens is None


def test_inference_response_rejects_empty_provider_name() -> None:
    with pytest.raises(ValueError, match="provider_name must be a non-empty string"):
        InferenceResponse(text="output", provider_name="   ")


def test_inference_response_rejects_non_string_text() -> None:
    with pytest.raises(TypeError, match="text must be a str"):
        InferenceResponse(text=12345, provider_name="stub")  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["input_tokens", "output_tokens", "total_tokens"])
def test_inference_response_rejects_bool_as_token_count(field_name: str) -> None:
    kwargs = {"text": "hello", "provider_name": "stub", field_name: True}
    with pytest.raises(TypeError, match=f"{field_name} must be an int or None"):
        InferenceResponse(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["input_tokens", "output_tokens", "total_tokens"])
def test_inference_response_rejects_negative_token_count(field_name: str) -> None:
    kwargs = {"text": "hello", "provider_name": "stub", field_name: -1}
    with pytest.raises(ValueError, match=f"{field_name} must be a non-negative integer"):
        InferenceResponse(**kwargs)


def test_inference_response_rejects_empty_optional_strings() -> None:
    with pytest.raises(ValueError, match="model_name must be a non-empty string if specified"):
        InferenceResponse(text="hi", provider_name="stub", model_name="  ")

    with pytest.raises(ValueError, match="finish_reason must be a non-empty string if specified"):
        InferenceResponse(text="hi", provider_name="stub", finish_reason="  ")


# --- InferenceExecutionTelemetry Contract Tests ---


def test_inference_telemetry_succeeded_round_trip() -> None:
    original = InferenceExecutionTelemetry(
        provider_name=" Gemini ",
        attempt_index=0,
        latency_ms=123.45,
        status="succeeded",
        input_tokens=50,
        output_tokens=25,
        total_tokens=75,
        finish_reason="STOP",
        model_name=" gemini-2.5-flash ",
    )
    assert original.provider_name == "gemini"
    assert original.finish_reason == "stop"
    assert original.model_name == "gemini-2.5-flash"
    assert original.latency_ms == 123.45

    restored = InferenceExecutionTelemetry.from_json(original.to_json())
    assert restored == original
    assert restored.failure_reason is None
    assert restored.status_code is None


def test_inference_telemetry_failure_round_trip() -> None:
    original = InferenceExecutionTelemetry(
        provider_name="cloudflare",
        attempt_index=1,
        latency_ms=45,
        status="fallback_eligible_failure",
        failure_reason="rate_limited",
        status_code=429,
        model_name="@cf/meta/llama-3.1-8b-instruct",
    )
    assert original.latency_ms == 45.0
    restored = InferenceExecutionTelemetry.from_json(original.to_json())
    assert restored == original
    assert restored.failure_reason == "rate_limited"
    assert restored.status_code == 429


@pytest.mark.parametrize(
    "status",
    ["fallback_eligible_failure", "non_fallback_failure", "unexpected_failure"],
)
def test_inference_telemetry_failure_statuses_require_failure_reason(status: str) -> None:
    with pytest.raises(ValueError, match=f"{status} status requires a non-empty failure_reason"):
        InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=0,
            latency_ms=10.0,
            status=status,
            failure_reason=None,
        )


@pytest.mark.parametrize(
    "valid_slug",
    [
        "rate_limited",
        "timeout",
        "connection_error",
        "server_error_500",
        "http_429",
        "malformed_response",
        "blocked_safety",
        "unexpected_error",
        "fallback_eligible_error",
    ],
)
def test_inference_telemetry_accepts_valid_failure_reason_slugs(valid_slug: str) -> None:
    telemetry = InferenceExecutionTelemetry(
        provider_name="stub",
        attempt_index=0,
        latency_ms=10.0,
        status="fallback_eligible_failure",
        failure_reason=valid_slug,
    )
    assert telemetry.failure_reason == valid_slug


@pytest.mark.parametrize(
    ("invalid_reason", "description"),
    [
        ("HTTP 429 Rate limit exceeded", "prose"),
        ("rate_limited!", "punctuation"),
        ("error; drop table", "SQL/punctuation"),
        ("https://api.cloudflare.com/client/v4/accounts", "URL"),
        ("Bearer test-secret-token-12345", "authorization header"),
        ("Traceback (most recent call last):\n  File ...", "stack trace"),
        ("timeout\n", "newline"),
        ("timeout-error", "hyphen"),
    ],
)
def test_inference_telemetry_rejects_invalid_failure_reason(
    invalid_reason: str,
    description: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=0,
            latency_ms=10.0,
            status="fallback_eligible_failure",
            failure_reason=invalid_reason,
        )
    assert "failure_reason must contain only lowercase letters, digits, and underscores" in str(exc_info.value)
    # Ensure sensitive inputs are not retained or leaked in the error message
    assert invalid_reason not in str(exc_info.value)


def test_inference_telemetry_succeeded_rejects_failure_reason_or_status_code() -> None:
    with pytest.raises(ValueError, match="succeeded status requires failure_reason to be None"):
        InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=0,
            latency_ms=10.0,
            status="succeeded",
            failure_reason="rate_limited",
        )

    with pytest.raises(ValueError, match="succeeded status requires status_code to be None"):
        InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=0,
            latency_ms=10.0,
            status="succeeded",
            status_code=200,
        )


def test_inference_telemetry_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="Invalid status"):
        InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=0,
            latency_ms=10.0,
            status="unknown_status",
            failure_reason="error",
        )


def test_inference_telemetry_rejects_invalid_attempt_index() -> None:
    with pytest.raises(ValueError, match="attempt_index must be a non-negative integer"):
        InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=True,  # type: ignore[arg-type]
            latency_ms=10.0,
            status="succeeded",
        )

    with pytest.raises(ValueError, match="attempt_index must be a non-negative integer"):
        InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=-1,
            latency_ms=10.0,
            status="succeeded",
        )


def test_inference_telemetry_rejects_invalid_latency() -> None:
    with pytest.raises(ValueError, match="latency_ms must be a non-negative number"):
        InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=0,
            latency_ms=True,  # type: ignore[arg-type]
            status="succeeded",
        )

    with pytest.raises(ValueError, match="latency_ms must be a non-negative number"):
        InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=0,
            latency_ms=-5.0,
            status="succeeded",
        )


def test_inference_telemetry_rejects_invalid_status_code() -> None:
    with pytest.raises(TypeError, match="status_code must be an int or None"):
        InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=0,
            latency_ms=10.0,
            status="fallback_eligible_failure",
            failure_reason="timeout",
            status_code=True,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="status_code must be a valid HTTP status"):
        InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=0,
            latency_ms=10.0,
            status="fallback_eligible_failure",
            failure_reason="timeout",
            status_code=999,
        )


# --- AgentEvent Inference Telemetry Integration Tests ---


def test_agent_event_with_inference_telemetry_round_trip() -> None:
    telemetry = InferenceExecutionTelemetry(
        provider_name="cloudflare",
        attempt_index=0,
        latency_ms=145.2,
        status="succeeded",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        finish_reason="stop",
        model_name="@cf/meta/llama-3.1-8b-instruct",
    )
    event = AgentEvent.create(
        agent_id="worker-1",
        state="working",
        detail="Inference attempt 0 succeeded (cloudflare)",
        task_id="task-telem-1",
        inference_telemetry=telemetry,
    )
    assert event.inference_telemetry == telemetry

    data = event.to_dict()
    assert "inference_telemetry" in data
    assert data["inference_telemetry"] == telemetry.to_dict()

    restored = AgentEvent.from_json(event.to_json())
    assert restored == event
    assert isinstance(restored.inference_telemetry, InferenceExecutionTelemetry)
    assert restored.inference_telemetry.provider_name == "cloudflare"
    assert restored.inference_telemetry.input_tokens == 100
    assert restored.inference_telemetry.output_tokens == 50
    assert restored.inference_telemetry.total_tokens == 150


def test_agent_event_omits_inference_telemetry_when_none() -> None:
    event = AgentEvent.create(
        agent_id="worker-1",
        state="working",
        detail="Processing task",
        task_id="task-telem-2",
    )
    assert event.inference_telemetry is None
    data = event.to_dict()
    assert "inference_telemetry" not in data


def test_agent_event_rejects_invalid_inference_telemetry_type() -> None:
    with pytest.raises(TypeError, match="inference_telemetry must be InferenceExecutionTelemetry"):
        AgentEvent(
            agent_id="worker-1",
            state="working",
            detail="detail",
            occurred_at="2026-11-09T00:00:00Z",
            inference_telemetry="invalid_telemetry",  # type: ignore[arg-type]
        )
