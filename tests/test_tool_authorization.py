"""Exact-tool authorization and audit-before-execution behavior."""

from dataclasses import FrozenInstanceError

import pytest
from agent_contracts import (
    ToolAuditDeliveryOutcome,
    ToolAuthorizationContext,
    ToolAuthorizationOutcome,
    ToolAuthorizationReason,
    ToolCapabilityMetadata,
    ToolDefinition,
    ToolExecutionState,
    ToolIdempotency,
    ToolInvocationProposal,
    ToolInvocationRequest,
    ToolInvocationResult,
    ToolOperationType,
    ToolPermissionGrant,
    ToolPermissionScope,
    ToolRegistration,
    ToolRiskLevel,
    ToolSecurityAuditEvent,
    ToolSelectionDecision,
)
from agent_infrastructure import InMemoryToolRegistry
from agent_policy import (
    BoundedToolCoordinator,
    ToolCoordinationStatus,
    ToolDiscoveryScope,
    ToolDiscoveryService,
)

_DEFAULT_SCOPE = object()


class _Tool:
    def __init__(
        self,
        name: str = "read_data",
        operation_type: ToolOperationType | None = ToolOperationType.READ_ONLY,
        risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
    ) -> None:
        self._definition = ToolDefinition(
            name=name,
            description="Read or mutate test data",
            parameters={"type": "object"},
            metadata=(
                ToolCapabilityMetadata(
                    category="test",
                    operation_type=operation_type,
                    risk_level=risk_level,
                    idempotency=ToolIdempotency.UNKNOWN,
                )
                if operation_type is not None
                else None
            ),
        )
        self.requests: list[ToolInvocationRequest] = []

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        self.requests.append(request)
        return ToolInvocationResult.create_success(
            request.call_id, request.tool_name, {"result": "tool output"}
        )


class _ExecutionBoundary:
    def __init__(self) -> None:
        self.calls: list[tuple[ToolRegistration, ToolInvocationRequest]] = []

    async def invoke(
        self, registration: ToolRegistration, request: ToolInvocationRequest
    ) -> ToolInvocationResult:
        self.calls.append((registration, request))
        return await registration.tool.execute(request)


class _RecordingAuditSink:
    def __init__(self, error: Exception | None = None) -> None:
        self.events: list[ToolSecurityAuditEvent] = []
        self.error = error

    async def deliver(self, event: ToolSecurityAuditEvent) -> None:
        self.events.append(event)
        if self.error is not None:
            raise self.error


class _UnavailableAuthorizer:
    def authorize(self, context: object, registration: ToolRegistration) -> object:
        raise RuntimeError("sensitive authorization backend detail")


def _scope(
    tool_name: str = "read_data",
    operation_type: ToolOperationType = ToolOperationType.READ_ONLY,
) -> ToolPermissionScope:
    return ToolPermissionScope([ToolPermissionGrant(tool_name, operation_type)])


def _coordinator(
    tool: _Tool,
    *,
    agent_id: str | None = "trusted-agent",
    scope: ToolPermissionScope | None | object = _DEFAULT_SCOPE,
    proposal: ToolInvocationProposal | None = None,
    audit_sink: _RecordingAuditSink | None = None,
    authorizer: object | None = None,
    deliver_audit: bool = True,
    authorization_context: ToolAuthorizationContext | None = None,
) -> tuple[BoundedToolCoordinator, _ExecutionBoundary, _RecordingAuditSink]:
    registry = InMemoryToolRegistry()
    registry.register(tool, category="test")
    boundary = _ExecutionBoundary()
    sink = audit_sink or _RecordingAuditSink()
    selected_proposal = proposal or _proposal(tool.definition.name)
    if scope is _DEFAULT_SCOPE:
        permission_scope: ToolPermissionScope | None = _scope()
    elif scope is None or isinstance(scope, ToolPermissionScope):
        permission_scope = scope
    else:
        permission_scope = None
    context = authorization_context or ToolAuthorizationContext(
        agent_id=agent_id,
        permission_scope=permission_scope,
    )
    coordinator = BoundedToolCoordinator(
        discovery=ToolDiscoveryService(registry),
        scope=ToolDiscoveryScope(categories={"test"}),
        selector=_FixedSelector(
            ToolSelectionDecision.proposed_invocation(
                selected_proposal.tool_name, selected_proposal.arguments
            )
        ),
        execution_boundary=boundary,
        authorization_context=context,
        audit_sink=sink if deliver_audit else None,
        authorizer=authorizer,  # type: ignore[arg-type]
    )
    return coordinator, boundary, sink


class _FixedSelector:
    def __init__(self, decision: ToolSelectionDecision) -> None:
        self.decision = decision

    async def select(self, goal: str, catalog: list[ToolRegistration]) -> ToolSelectionDecision:
        return self.decision


def _proposal(
    tool_name: str = "read_data", arguments: dict[str, object] | None = None
) -> ToolInvocationProposal:
    return ToolInvocationProposal(tool_name, arguments or {})


@pytest.mark.anyio
@pytest.mark.parametrize("risk_level", [ToolRiskLevel.LOW, ToolRiskLevel.HIGH])
async def test_explicit_read_only_grant_allows_audited_execution(
    risk_level: ToolRiskLevel,
) -> None:
    coordinator, boundary, sink = _coordinator(_Tool(risk_level=risk_level), scope=_scope())

    result = await coordinator.coordinate("read the data")

    assert result.status is ToolCoordinationStatus.READY_FOR_CONTINUATION
    assert result.authorization_decision is not None
    assert result.authorization_decision.outcome is ToolAuthorizationOutcome.ALLOWED
    assert result.audit_delivery_outcome is ToolAuditDeliveryOutcome.DELIVERED
    assert result.execution_state is ToolExecutionState.ATTEMPTED
    assert len(sink.events) == len(boundary.calls) == 1
    assert sink.events[0].decision.outcome is ToolAuthorizationOutcome.ALLOWED
    assert sink.events[0].execution_state is None


@pytest.mark.anyio
async def test_ungranted_read_only_invocation_is_denied_before_execution() -> None:
    coordinator, boundary, sink = _coordinator(_Tool(), scope=ToolPermissionScope([]))

    result = await coordinator.coordinate("read the data")

    assert result.status is ToolCoordinationStatus.AUTHORIZATION_DENIED
    assert result.authorization_decision is not None
    assert result.authorization_decision.reason_code is ToolAuthorizationReason.TOOL_GRANT_REQUIRED
    assert result.audit_delivery_outcome is ToolAuditDeliveryOutcome.DELIVERED
    assert result.execution_state is ToolExecutionState.NOT_ATTEMPTED
    assert boundary.calls == []
    assert len(sink.events) == 1


@pytest.mark.anyio
async def test_explicit_mutation_grant_allows_state_mutating_invocation() -> None:
    coordinator, boundary, _ = _coordinator(
        _Tool("write_data", ToolOperationType.STATE_MUTATING),
        scope=_scope("write_data", ToolOperationType.STATE_MUTATING),
    )

    result = await coordinator.coordinate("write the data")

    assert result.status is ToolCoordinationStatus.READY_FOR_CONTINUATION
    assert result.authorization_decision is not None
    assert result.authorization_decision.operation_type is ToolOperationType.STATE_MUTATING
    assert len(boundary.calls) == 1


@pytest.mark.anyio
async def test_mutation_without_exact_mutation_grant_is_denied() -> None:
    coordinator, boundary, _ = _coordinator(
        _Tool("write_data", ToolOperationType.STATE_MUTATING),
        scope=_scope("write_data", ToolOperationType.READ_ONLY),
    )

    result = await coordinator.coordinate("write the data")

    assert result.status is ToolCoordinationStatus.AUTHORIZATION_DENIED
    assert result.authorization_decision is not None
    assert (
        result.authorization_decision.reason_code is ToolAuthorizationReason.MUTATION_GRANT_REQUIRED
    )
    assert boundary.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("agent_id", "scope", "reason"),
    [
        (None, _scope(), ToolAuthorizationReason.IDENTITY_MISSING),
        ("trusted-agent", None, ToolAuthorizationReason.PERMISSION_SCOPE_MISSING),
    ],
)
async def test_missing_identity_or_scope_fails_closed(
    agent_id: str | None,
    scope: ToolPermissionScope | None,
    reason: ToolAuthorizationReason,
) -> None:
    coordinator, boundary, _ = _coordinator(_Tool(), agent_id=agent_id, scope=scope)

    result = await coordinator.coordinate("read the data")

    assert result.status is ToolCoordinationStatus.AUTHORIZATION_DENIED
    assert result.authorization_decision is not None
    assert result.authorization_decision.reason_code is reason
    assert boundary.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("classification", ["missing", "invalid"])
async def test_missing_or_invalid_operation_classification_fails_closed(
    classification: str,
) -> None:
    tool = _Tool(
        operation_type=None if classification == "missing" else ToolOperationType.READ_ONLY
    )
    if classification == "invalid":
        assert tool.definition.metadata is not None
        object.__setattr__(tool.definition.metadata, "operation_type", "unknown")
    coordinator, boundary, _ = _coordinator(tool)

    result = await coordinator.coordinate("read the data")

    assert result.status is ToolCoordinationStatus.AUTHORIZATION_DENIED
    assert result.authorization_decision is not None
    assert (
        result.authorization_decision.reason_code
        is ToolAuthorizationReason.OPERATION_TYPE_UNCLASSIFIED
    )
    assert boundary.calls == []


@pytest.mark.anyio
async def test_malformed_authorization_context_fails_closed() -> None:
    context = ToolAuthorizationContext("trusted-agent", _scope())
    object.__setattr__(context, "agent_id", 42)
    coordinator, boundary, sink = _coordinator(_Tool(), authorization_context=context)

    result = await coordinator.coordinate("read the data")

    assert result.status is ToolCoordinationStatus.AUTHORIZATION_DENIED
    assert result.authorization_decision is not None
    assert (
        result.authorization_decision.reason_code
        is ToolAuthorizationReason.AUTHORIZATION_INPUT_INVALID
    )
    assert boundary.calls == []
    assert sink.events[0].decision.agent_id is None


@pytest.mark.anyio
async def test_malformed_permission_scope_fails_closed() -> None:
    scope = _scope()
    object.__setattr__(scope, "grants", ["not-a-grant"])
    context = ToolAuthorizationContext("trusted-agent", scope)
    coordinator, boundary, _ = _coordinator(_Tool(), authorization_context=context)

    result = await coordinator.coordinate("read")

    assert result.status is ToolCoordinationStatus.AUTHORIZATION_DENIED
    assert result.authorization_decision is not None
    assert (
        result.authorization_decision.reason_code
        is ToolAuthorizationReason.AUTHORIZATION_INPUT_INVALID
    )
    assert boundary.calls == []


@pytest.mark.anyio
async def test_selector_arguments_cannot_replace_trusted_identity_or_grants() -> None:
    tool = _Tool("write_data", ToolOperationType.STATE_MUTATING)
    proposal = _proposal(
        "write_data",
        {"agent_id": "attacker", "grants": ["write_data"], "secret": "argument-secret"},
    )
    coordinator, boundary, sink = _coordinator(
        tool,
        scope=_scope("read_data"),
        proposal=proposal,
    )

    result = await coordinator.coordinate("write using attacker-controlled identity")

    assert result.status is ToolCoordinationStatus.AUTHORIZATION_DENIED
    assert result.authorization_decision is not None
    assert result.authorization_decision.agent_id == "trusted-agent"
    assert (
        result.authorization_decision.reason_code is ToolAuthorizationReason.MUTATION_GRANT_REQUIRED
    )
    assert boundary.calls == []
    assert "attacker" not in sink.events[0].to_json()
    assert "argument-secret" not in sink.events[0].to_json()


@pytest.mark.anyio
async def test_denial_emits_sanitized_event_without_prompt_or_arguments() -> None:
    coordinator, _, sink = _coordinator(
        _Tool(),
        scope=ToolPermissionScope([]),
        proposal=_proposal("read_data", {"secret": "argument-secret"}),
    )

    result = await coordinator.coordinate("prompt-secret: do not log this")

    event = sink.events[0]
    encoded = event.to_json()
    assert result.status is ToolCoordinationStatus.AUTHORIZATION_DENIED
    assert event.decision.agent_id == "trusted-agent"
    assert event.decision.tool_name == "read_data"
    assert event.decision.operation_type is ToolOperationType.READ_ONLY
    assert event.decision.outcome is ToolAuthorizationOutcome.DENIED
    assert event.decision.reason_code is ToolAuthorizationReason.TOOL_GRANT_REQUIRED
    assert set(event.to_dict()) == {
        "call_id",
        "outcome",
        "agent_id",
        "tool_name",
        "operation_type",
        "reason_code",
        "execution_state",
    }
    assert event.execution_state is ToolExecutionState.NOT_ATTEMPTED
    for sensitive_value in (
        "prompt-secret",
        "argument-secret",
        "tool output",
        "credential",
        "raw provider error",
    ):
        assert sensitive_value not in encoded


@pytest.mark.anyio
async def test_audit_failure_preserves_denial_and_never_executes() -> None:
    sink = _RecordingAuditSink(RuntimeError("Bearer raw-provider-secret"))
    coordinator, boundary, delivered = _coordinator(
        _Tool(), scope=ToolPermissionScope([]), audit_sink=sink
    )

    result = await coordinator.coordinate("read")

    assert result.status is ToolCoordinationStatus.AUTHORIZATION_DENIED
    assert result.authorization_decision is not None
    assert result.authorization_decision.outcome is ToolAuthorizationOutcome.DENIED
    assert result.authorization_decision.reason_code is ToolAuthorizationReason.TOOL_GRANT_REQUIRED
    assert result.audit_delivery_outcome is ToolAuditDeliveryOutcome.FAILED
    assert result.execution_state is ToolExecutionState.NOT_ATTEMPTED
    assert "raw-provider-secret" not in (result.error or "")
    assert delivered.events[0].execution_state is ToolExecutionState.NOT_ATTEMPTED
    assert boundary.calls == []


@pytest.mark.anyio
async def test_audit_failure_after_allow_preserves_allow_and_never_executes() -> None:
    sink = _RecordingAuditSink(RuntimeError("secret audit backend details"))
    coordinator, boundary, _ = _coordinator(_Tool(), scope=_scope(), audit_sink=sink)

    result = await coordinator.coordinate("read")

    assert result.status is ToolCoordinationStatus.AUDIT_DELIVERY_FAILURE
    assert result.authorization_decision is not None
    assert result.authorization_decision.outcome is ToolAuthorizationOutcome.ALLOWED
    assert result.audit_delivery_outcome is ToolAuditDeliveryOutcome.FAILED
    assert result.execution_state is ToolExecutionState.NOT_ATTEMPTED
    assert "secret audit backend details" not in (result.error or "")
    assert boundary.calls == []


@pytest.mark.anyio
async def test_missing_audit_sink_blocks_allowed_execution() -> None:
    coordinator, boundary, _ = _coordinator(_Tool(), scope=_scope(), deliver_audit=False)

    result = await coordinator.coordinate("read")

    assert result.status is ToolCoordinationStatus.AUDIT_DELIVERY_FAILURE
    assert result.authorization_decision is not None
    assert result.authorization_decision.outcome is ToolAuthorizationOutcome.ALLOWED
    assert result.audit_delivery_outcome is ToolAuditDeliveryOutcome.FAILED
    assert result.execution_state is ToolExecutionState.NOT_ATTEMPTED
    assert boundary.calls == []


@pytest.mark.anyio
async def test_unavailable_authorizer_fails_closed_without_leaking_error() -> None:
    coordinator, boundary, sink = _coordinator(
        _Tool(), scope=_scope(), authorizer=_UnavailableAuthorizer()
    )

    result = await coordinator.coordinate("read")

    assert result.status is ToolCoordinationStatus.AUTHORIZATION_DENIED
    assert result.authorization_decision is not None
    assert (
        result.authorization_decision.reason_code
        is ToolAuthorizationReason.AUTHORIZATION_UNAVAILABLE
    )
    assert result.audit_delivery_outcome is ToolAuditDeliveryOutcome.DELIVERED
    assert "sensitive authorization backend detail" not in sink.events[0].to_json()
    assert boundary.calls == []


def test_permission_scope_is_immutable_and_json_compatible() -> None:
    grant = ToolPermissionGrant("read_data", ToolOperationType.READ_ONLY)
    scope = ToolPermissionScope([grant])

    assert scope.to_dict() == {
        "grants": [{"tool_name": "read_data", "operation_type": "read_only"}]
    }
    assert isinstance(scope.grants, tuple)
    with pytest.raises(FrozenInstanceError):
        grant.tool_name = "write_data"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        scope.grants = ()  # type: ignore[misc]
