"""One bounded discovery-to-invocation cycle with an explicit execution seam."""

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_contracts import (
    ToolAuditDeliveryOutcome,
    ToolAuthorizationContext,
    ToolAuthorizationDecision,
    ToolAuthorizationOutcome,
    ToolAuthorizationReason,
    ToolCapabilityMetadata,
    ToolErrorCode,
    ToolExecutionState,
    ToolExecutionTelemetry,
    ToolInvocationProposal,
    ToolInvocationRequest,
    ToolInvocationResult,
    ToolOperationType,
    ToolPermissionScope,
    ToolRegistration,
    ToolSecurityAuditEvent,
    ToolSelectionDecision,
    ToolSelectionKind,
    ToolSelector,
)

from .tool_authorization import ExactToolAuthorizer, ToolAuthorizer, ToolSecurityAuditSink
from .tool_discovery import ToolDiscoveryScope, ToolDiscoveryService
from .tool_validation import (
    DeterministicToolArgumentCorrector,
    ToolArgumentCorrector,
    ToolArgumentValidationResult,
    ToolArgumentValidator,
)

logger = logging.getLogger(__name__)


def _json_size_bytes(value: object) -> int | None:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    except Exception:  # noqa: BLE001 - serialization must not affect tool execution
        logger.debug("Tool execution telemetry size measurement unavailable")
        return None
    return len(encoded)


class ToolCoordinationStatus(str, Enum):
    """Observable outcomes from one bounded coordination request."""

    NO_TOOL_NEEDED = "no_tool_needed"
    READY_FOR_CONTINUATION = "ready_for_continuation"
    AUTHORIZATION_DENIED = "authorization_denied"
    AUDIT_DELIVERY_FAILURE = "audit_delivery_failure"
    PRE_EXECUTION_FAILURE = "pre_execution_failure"
    TOOL_INVOCATION_FAILURE = "tool_invocation_failure"


@dataclass(frozen=True)
class ToolCoordinationResult:
    """Lifecycle result kept distinct from the underlying invocation result."""

    status: ToolCoordinationStatus
    decision: ToolSelectionDecision | None = None
    invocation_result: ToolInvocationResult | None = None
    error: str | None = None
    effective_invocation: ToolInvocationProposal | None = None
    authorization_decision: ToolAuthorizationDecision | None = None
    audit_delivery_outcome: ToolAuditDeliveryOutcome = ToolAuditDeliveryOutcome.NOT_ATTEMPTED
    execution_state: ToolExecutionState = ToolExecutionState.NOT_ATTEMPTED

    def __post_init__(self) -> None:
        if not isinstance(self.status, ToolCoordinationStatus):
            raise TypeError("status must be a ToolCoordinationStatus")
        if self.decision is not None and not isinstance(self.decision, ToolSelectionDecision):
            raise TypeError("decision must be a ToolSelectionDecision")
        if self.effective_invocation is not None and not isinstance(
            self.effective_invocation, ToolInvocationProposal
        ):
            raise TypeError("effective_invocation must be a ToolInvocationProposal")
        if self.invocation_result is not None and not isinstance(
            self.invocation_result, ToolInvocationResult
        ):
            raise TypeError("invocation_result must be a ToolInvocationResult")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string when provided")
        if self.authorization_decision is not None and not isinstance(
            self.authorization_decision, ToolAuthorizationDecision
        ):
            raise TypeError("authorization_decision must be a ToolAuthorizationDecision")
        if not isinstance(self.audit_delivery_outcome, ToolAuditDeliveryOutcome):
            raise TypeError("audit_delivery_outcome must be a ToolAuditDeliveryOutcome")
        if not isinstance(self.execution_state, ToolExecutionState):
            raise TypeError("execution_state must be a ToolExecutionState")

        if self.status is ToolCoordinationStatus.NO_TOOL_NEEDED:
            if (
                self.decision is None
                or self.decision.kind is not ToolSelectionKind.NO_TOOL_NEEDED
                or self.invocation_result is not None
                or self.error is not None
                or self.effective_invocation is not None
                or self.authorization_decision is not None
                or self.audit_delivery_outcome is not ToolAuditDeliveryOutcome.NOT_ATTEMPTED
                or self.execution_state is not ToolExecutionState.NOT_ATTEMPTED
            ):
                raise ValueError("no-tool results require only a no-tool decision")
        elif self.status is ToolCoordinationStatus.READY_FOR_CONTINUATION:
            if (
                self.decision is None
                or self.decision.kind is not ToolSelectionKind.PROPOSED_INVOCATION
                or self.decision.invocation is None
                or self.effective_invocation is None
                or self.decision.invocation.tool_name != self.effective_invocation.tool_name
                or self.invocation_result is None
                or self.invocation_result.tool_name != self.effective_invocation.tool_name
                or not self.invocation_result.success
                or self.error is not None
                or self.authorization_decision is None
                or self.authorization_decision.outcome is not ToolAuthorizationOutcome.ALLOWED
                or self.audit_delivery_outcome is not ToolAuditDeliveryOutcome.DELIVERED
                or self.execution_state is not ToolExecutionState.ATTEMPTED
            ):
                raise ValueError("continuation results require a successful invocation")
        elif self.status is ToolCoordinationStatus.PRE_EXECUTION_FAILURE:
            if (
                not isinstance(self.error, str)
                or not self.error.strip()
                or (
                    self.decision is not None
                    and self.decision.kind is not ToolSelectionKind.PROPOSED_INVOCATION
                )
                or self.invocation_result is not None
                or self.effective_invocation is not None
                or self.authorization_decision is not None
                or self.audit_delivery_outcome is not ToolAuditDeliveryOutcome.NOT_ATTEMPTED
                or self.execution_state is not ToolExecutionState.NOT_ATTEMPTED
            ):
                raise ValueError("pre-execution failures require an error and no invocation result")
        elif self.status is ToolCoordinationStatus.AUTHORIZATION_DENIED:
            if (
                self.decision is None
                or self.decision.kind is not ToolSelectionKind.PROPOSED_INVOCATION
                or self.authorization_decision is None
                or self.authorization_decision.outcome is not ToolAuthorizationOutcome.DENIED
                or self.authorization_decision.reason_code is None
                or self.error != self.authorization_decision.reason_code.value
                or self.audit_delivery_outcome is ToolAuditDeliveryOutcome.NOT_ATTEMPTED
                or self.execution_state is not ToolExecutionState.NOT_ATTEMPTED
                or self.invocation_result is not None
                or self.effective_invocation is not None
            ):
                raise ValueError("authorization denials must remain observable and unexecuted")
        elif self.status is ToolCoordinationStatus.AUDIT_DELIVERY_FAILURE:
            if (
                self.decision is None
                or self.decision.kind is not ToolSelectionKind.PROPOSED_INVOCATION
                or self.authorization_decision is None
                or self.authorization_decision.outcome is not ToolAuthorizationOutcome.ALLOWED
                or self.audit_delivery_outcome is not ToolAuditDeliveryOutcome.FAILED
                or self.execution_state is not ToolExecutionState.NOT_ATTEMPTED
                or self.invocation_result is not None
                or self.effective_invocation is not None
                or not isinstance(self.error, str)
                or not self.error.strip()
            ):
                raise ValueError("audit failures must preserve authorization and block execution")
        elif self.status is ToolCoordinationStatus.TOOL_INVOCATION_FAILURE and (
            self.decision is None
            or self.decision.kind is not ToolSelectionKind.PROPOSED_INVOCATION
            or self.decision.invocation is None
            or self.effective_invocation is None
            or self.decision.invocation.tool_name != self.effective_invocation.tool_name
            or self.invocation_result is None
            or self.invocation_result.tool_name != self.effective_invocation.tool_name
            or self.invocation_result.success
            or not isinstance(self.error, str)
            or not self.error.strip()
            or self.authorization_decision is None
            or self.authorization_decision.outcome is not ToolAuthorizationOutcome.ALLOWED
            or self.audit_delivery_outcome is not ToolAuditDeliveryOutcome.DELIVERED
            or self.execution_state is not ToolExecutionState.ATTEMPTED
        ):
            raise ValueError("invocation failures require a failed tool result")


@runtime_checkable
class ToolExecutionBoundary(Protocol):
    """Seam between validated proposals and execution policy."""

    async def invoke(
        self, registration: ToolRegistration, request: ToolInvocationRequest
    ) -> ToolInvocationResult: ...


class TrustedCompositionToolInvoker:
    """Invoke a deliberately supplied tool after coordinator policy has authorized it.

    This adapter performs no authorization or audit delivery itself. Call it only
    through a coordinator that enforces those boundaries.
    """

    async def invoke(
        self, registration: ToolRegistration, request: ToolInvocationRequest
    ) -> ToolInvocationResult:
        return await registration.tool.execute(request)


class BoundedToolCoordinator:
    """Discover, select, validate, and optionally invoke at most one tool."""

    def __init__(
        self,
        discovery: ToolDiscoveryService,
        scope: ToolDiscoveryScope,
        selector: ToolSelector,
        execution_boundary: ToolExecutionBoundary,
        validator: ToolArgumentValidator | None = None,
        corrector: ToolArgumentCorrector | None = None,
        *,
        authorization_context: ToolAuthorizationContext | None = None,
        authorizer: ToolAuthorizer | None = None,
        audit_sink: ToolSecurityAuditSink | None = None,
        on_telemetry: Callable[[ToolExecutionTelemetry], Awaitable[None]] | None = None,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if on_telemetry is not None and not callable(on_telemetry):
            raise TypeError("on_telemetry must be callable or None")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._discovery = discovery
        self._scope = scope
        self._selector = selector
        self._execution_boundary = execution_boundary
        self._validator = validator or ToolArgumentValidator()
        self._corrector = corrector or DeterministicToolArgumentCorrector()
        self._authorization_context = authorization_context
        self._authorizer = authorizer if authorizer is not None else ExactToolAuthorizer()
        self._audit_sink = audit_sink
        self._on_telemetry = on_telemetry
        self._clock = clock

    async def coordinate(self, goal: str) -> ToolCoordinationResult:
        try:
            catalog = self._discovery.discover(self._scope)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._pre_execution_failure(f"Tool discovery failed: {exc}")

        if not isinstance(catalog, list) or any(
            not isinstance(registration, ToolRegistration) for registration in catalog
        ):
            return self._pre_execution_failure("Tool discovery returned a malformed catalog")
        names = [registration.name for registration in catalog]
        if len(names) != len(set(names)):
            return self._pre_execution_failure("Tool discovery returned duplicate tool identities")
        if not catalog:
            return self._no_tool_needed()

        try:
            decision = await self._selector.select(goal, catalog)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._pre_execution_failure(f"Tool selection failed: {exc}")
        if not isinstance(decision, ToolSelectionDecision):
            return self._pre_execution_failure(
                "Tool selector returned a malformed selection decision"
            )
        if decision.kind is ToolSelectionKind.NO_TOOL_NEEDED:
            return self._no_tool_needed(decision)

        proposal = decision.invocation
        if proposal is None or not isinstance(proposal, ToolInvocationProposal):
            return self._pre_execution_failure(
                "Tool selector returned a malformed invocation proposal", decision
            )
        matching = [
            registration for registration in catalog if registration.name == proposal.tool_name
        ]
        if len(matching) != 1:
            return self._pre_execution_failure(
                f"Selected tool {proposal.tool_name!r} is not uniquely present in the current catalog",
                decision,
            )
        registration = matching[0]

        try:
            validation = self._validator.validate(registration, proposal.arguments)
        except (KeyError, TypeError, ValueError) as exc:
            return self._pre_execution_failure(f"Tool argument validation failed: {exc}", decision)

        if not validation.valid:
            try:
                corrected = await self._corrector.correct(registration, proposal, validation)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                return self._pre_execution_failure(
                    f"Tool argument correction failed: {exc}", decision
                )
            if corrected is None or not isinstance(corrected, ToolInvocationProposal):
                return self._pre_execution_failure(
                    self._validation_failure_message(validation), decision
                )
            if corrected.tool_name != proposal.tool_name:
                return self._pre_execution_failure(
                    "Tool argument correction changed the selected tool identity",
                    decision,
                )
            proposal = corrected
            try:
                validation = self._validator.validate(registration, proposal.arguments)
            except (KeyError, TypeError, ValueError) as exc:
                return self._pre_execution_failure(
                    f"Corrected tool argument validation failed: {exc}", decision
                )
            if not validation.valid:
                return self._pre_execution_failure(
                    self._validation_failure_message(validation), decision
                )

        try:
            request = ToolInvocationRequest.create(
                tool_name=proposal.tool_name,
                arguments=proposal.arguments,
            )
        except (TypeError, ValueError) as exc:
            return self._pre_execution_failure(
                f"Could not construct a valid tool invocation request: {exc}", decision
            )

        authorization_decision = self._authorize(registration, request)
        audit_delivery_outcome = await self._deliver_authorization_event(
            request.call_id, authorization_decision
        )
        if authorization_decision.outcome is ToolAuthorizationOutcome.DENIED:
            assert authorization_decision.reason_code is not None
            return ToolCoordinationResult(
                status=ToolCoordinationStatus.AUTHORIZATION_DENIED,
                decision=decision,
                error=authorization_decision.reason_code.value,
                authorization_decision=authorization_decision,
                audit_delivery_outcome=audit_delivery_outcome,
            )
        if audit_delivery_outcome is ToolAuditDeliveryOutcome.FAILED:
            return ToolCoordinationResult(
                status=ToolCoordinationStatus.AUDIT_DELIVERY_FAILURE,
                decision=decision,
                error="Authorization audit delivery failed",
                authorization_decision=authorization_decision,
                audit_delivery_outcome=audit_delivery_outcome,
            )

        argument_size_bytes = (
            _json_size_bytes(request.arguments) if self._on_telemetry is not None else None
        )
        try:
            started_at = datetime.now(UTC).isoformat() if self._on_telemetry is not None else None
        except Exception:  # noqa: BLE001 - clock failure must not affect tool execution
            logger.debug("Tool execution telemetry start-time measurement unavailable")
            started_at = None
        try:
            started_ns = self._clock() if self._on_telemetry is not None else None
        except Exception:  # noqa: BLE001 - clock failure must not affect tool execution
            logger.debug("Tool execution telemetry duration measurement unavailable")
            started_ns = None

        try:
            invocation_result = await self._execution_boundary.invoke(registration, request)
        except Exception as exc:  # noqa: BLE001 - tool implementations may raise arbitrary execution errors
            duration_ms = self._duration_ms(started_ns)
            invocation_result = ToolInvocationResult.create_failure(
                call_id=request.call_id,
                tool_name=request.tool_name,
                error=str(exc) or type(exc).__name__,
                code=ToolErrorCode.EXECUTION_FAILURE,
            )
        else:
            duration_ms = self._duration_ms(started_ns)

        if (
            not isinstance(invocation_result, ToolInvocationResult)
            or invocation_result.call_id != request.call_id
            or invocation_result.tool_name != request.tool_name
        ):
            invocation_result = ToolInvocationResult.create_failure(
                call_id=request.call_id,
                tool_name=request.tool_name,
                error="Execution boundary returned a malformed tool result",
                code=ToolErrorCode.EXECUTION_FAILURE,
            )

        if duration_ms is not None:
            await self._publish_telemetry(
                registration=registration,
                request=request,
                invocation_result=invocation_result,
                started_at=started_at,
                duration_ms=duration_ms,
                argument_size_bytes=argument_size_bytes,
            )

        if not invocation_result.success:
            return ToolCoordinationResult(
                status=ToolCoordinationStatus.TOOL_INVOCATION_FAILURE,
                decision=decision,
                invocation_result=invocation_result,
                effective_invocation=proposal,
                error=(
                    invocation_result.error.message
                    if invocation_result.error is not None
                    else "Tool invocation failed"
                ),
                authorization_decision=authorization_decision,
                audit_delivery_outcome=audit_delivery_outcome,
                execution_state=ToolExecutionState.ATTEMPTED,
            )
        return ToolCoordinationResult(
            status=ToolCoordinationStatus.READY_FOR_CONTINUATION,
            decision=decision,
            invocation_result=invocation_result,
            effective_invocation=proposal,
            authorization_decision=authorization_decision,
            audit_delivery_outcome=audit_delivery_outcome,
            execution_state=ToolExecutionState.ATTEMPTED,
        )

    def _duration_ms(self, started_ns: int | None) -> float | None:
        if started_ns is None:
            return None
        try:
            elapsed_ns = self._clock() - started_ns
            if elapsed_ns < 0:
                logger.debug("Tool execution telemetry duration measurement was negative")
                return None
            return elapsed_ns / 1_000_000
        except Exception:  # noqa: BLE001 - clock failure must not affect tool execution
            logger.debug("Tool execution telemetry duration measurement unavailable")
            return None

    async def _publish_telemetry(
        self,
        *,
        registration: ToolRegistration,
        request: ToolInvocationRequest,
        invocation_result: ToolInvocationResult,
        started_at: str | None,
        duration_ms: float,
        argument_size_bytes: int | None,
    ) -> None:
        if self._on_telemetry is None:
            return
        try:
            agent_id = self._context_agent_id(self._authorization_context)
            if not isinstance(agent_id, str) or not agent_id.strip():
                logger.warning("Tool execution telemetry has no trusted agent identity")
                return
            result_size_bytes = (
                _json_size_bytes(invocation_result.output)
                if invocation_result.success or invocation_result.output is not None
                else None
            )
            telemetry = ToolExecutionTelemetry(
                tool_name=registration.name,
                agent_id=agent_id,
                call_id=request.call_id,
                task_id=None,
                started_at=started_at,
                duration_ms=duration_ms,
                argument_size_bytes=argument_size_bytes,
                result_size_bytes=result_size_bytes,
                outcome="succeeded" if invocation_result.success else "failed",
                error_code=(
                    invocation_result.error.code if invocation_result.error is not None else None
                ),
            )
            await self._on_telemetry(telemetry)
        except Exception:  # noqa: BLE001 - telemetry must not affect tool execution
            logger.warning("Tool execution telemetry publication failed")

    def _authorize(
        self, registration: ToolRegistration, request: ToolInvocationRequest
    ) -> ToolAuthorizationDecision:
        tool_name = request.tool_name
        context = self._authorization_context
        agent_id = self._context_agent_id(context)
        operation_type = self._operation_type(registration)

        if self._authorizer is None:
            return self._authorization_denial(
                agent_id,
                tool_name,
                operation_type,
                ToolAuthorizationReason.AUTHORIZATION_UNAVAILABLE,
            )
        try:
            authorization = self._authorizer.authorize(context, registration)
        except Exception:  # noqa: BLE001 - authorization failures must fail closed
            authorization = None

        if authorization is None or not isinstance(authorization, ToolAuthorizationDecision):
            return self._authorization_denial(
                agent_id,
                tool_name,
                operation_type,
                ToolAuthorizationReason.AUTHORIZATION_UNAVAILABLE,
            )
        if (
            authorization.agent_id != agent_id
            or authorization.tool_name != tool_name
            or authorization.operation_type is not operation_type
        ):
            return self._authorization_denial(
                agent_id,
                tool_name,
                operation_type,
                ToolAuthorizationReason.AUTHORIZATION_INPUT_INVALID,
            )
        if authorization.outcome is ToolAuthorizationOutcome.ALLOWED:
            if not isinstance(context, ToolAuthorizationContext):
                return self._authorization_denial(
                    agent_id,
                    tool_name,
                    operation_type,
                    ToolAuthorizationReason.AUTHORIZATION_INPUT_INVALID,
                )
            if (
                not agent_id
                or not isinstance(context.permission_scope, ToolPermissionScope)
                or operation_type is None
            ):
                return self._authorization_denial(
                    agent_id,
                    tool_name,
                    operation_type,
                    ToolAuthorizationReason.AUTHORIZATION_INPUT_INVALID,
                )
        return authorization

    @staticmethod
    def _context_agent_id(context: ToolAuthorizationContext | None) -> str | None:
        if not isinstance(context, ToolAuthorizationContext):
            return None
        agent_id = context.agent_id
        return agent_id if isinstance(agent_id, str) and agent_id.strip() else None

    async def _deliver_authorization_event(
        self,
        call_id: str,
        authorization: ToolAuthorizationDecision,
    ) -> ToolAuditDeliveryOutcome:
        if self._audit_sink is None:
            return ToolAuditDeliveryOutcome.FAILED
        try:
            result = await self._audit_sink.deliver(
                ToolSecurityAuditEvent(
                    call_id=call_id,
                    decision=authorization,
                    execution_state=(
                        ToolExecutionState.NOT_ATTEMPTED
                        if authorization.outcome is ToolAuthorizationOutcome.DENIED
                        else None
                    ),
                )
            )
        except Exception:  # noqa: BLE001 - audit failure must block execution
            return ToolAuditDeliveryOutcome.FAILED
        return (
            ToolAuditDeliveryOutcome.DELIVERED
            if result is None
            else ToolAuditDeliveryOutcome.FAILED
        )

    @staticmethod
    def _operation_type(registration: ToolRegistration) -> ToolOperationType | None:
        try:
            metadata = registration.definition.metadata
            operation_type = (
                metadata.operation_type if isinstance(metadata, ToolCapabilityMetadata) else None
            )
        except Exception:  # noqa: BLE001 - unreadable classification must fail closed
            return None
        return operation_type if isinstance(operation_type, ToolOperationType) else None

    @staticmethod
    def _authorization_denial(
        agent_id: str | None,
        tool_name: str,
        operation_type: ToolOperationType | None,
        reason: ToolAuthorizationReason,
    ) -> ToolAuthorizationDecision:
        return ToolAuthorizationDecision(
            outcome=ToolAuthorizationOutcome.DENIED,
            agent_id=agent_id,
            tool_name=tool_name,
            operation_type=operation_type,
            reason_code=reason,
        )

    @staticmethod
    def _no_tool_needed(
        decision: ToolSelectionDecision | None = None,
    ) -> ToolCoordinationResult:
        return ToolCoordinationResult(
            status=ToolCoordinationStatus.NO_TOOL_NEEDED,
            decision=decision or ToolSelectionDecision.no_tool_needed(),
        )

    @staticmethod
    def _pre_execution_failure(
        error: str, decision: ToolSelectionDecision | None = None
    ) -> ToolCoordinationResult:
        return ToolCoordinationResult(
            status=ToolCoordinationStatus.PRE_EXECUTION_FAILURE,
            decision=decision,
            error=error,
        )

    @staticmethod
    def _validation_failure_message(
        validation: ToolArgumentValidationResult,
    ) -> str:
        return "Invalid tool arguments: " + "; ".join(validation.errors)
