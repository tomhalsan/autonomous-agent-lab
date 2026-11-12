"""One bounded discovery-to-invocation cycle with an explicit execution seam."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_contracts import (
    ToolErrorCode,
    ToolInvocationProposal,
    ToolInvocationRequest,
    ToolInvocationResult,
    ToolRegistration,
    ToolSelectionDecision,
    ToolSelectionKind,
    ToolSelector,
)

from .tool_discovery import ToolDiscoveryScope, ToolDiscoveryService
from .tool_validation import (
    DeterministicToolArgumentCorrector,
    ToolArgumentCorrector,
    ToolArgumentValidationResult,
    ToolArgumentValidator,
)


class ToolCoordinationStatus(str, Enum):
    """Observable outcomes from one bounded coordination request."""

    NO_TOOL_NEEDED = "no_tool_needed"
    READY_FOR_CONTINUATION = "ready_for_continuation"
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

        if self.status is ToolCoordinationStatus.NO_TOOL_NEEDED:
            if (
                self.decision is None
                or self.decision.kind is not ToolSelectionKind.NO_TOOL_NEEDED
                or self.invocation_result is not None
                or self.error is not None
                or self.effective_invocation is not None
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
            ):
                raise ValueError("pre-execution failures require an error and no invocation result")
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
        ):
            raise ValueError("invocation failures require a failed tool result")


@runtime_checkable
class ToolExecutionBoundary(Protocol):
    """Seam between validated proposals and execution policy."""

    async def invoke(
        self, registration: ToolRegistration, request: ToolInvocationRequest
    ) -> ToolInvocationResult: ...


class TrustedCompositionToolInvoker:
    """Temporary prototype seam; invokes deliberately supplied tools without authorization.

    Discovery eligibility is not execution authorization. Selection does not
    constitute authorization, and registration does not constitute authorization.
    Issue #27 will replace this trusted-composition assumption with real enforcement.
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
    ) -> None:
        self._discovery = discovery
        self._scope = scope
        self._selector = selector
        self._execution_boundary = execution_boundary
        self._validator = validator or ToolArgumentValidator()
        self._corrector = corrector or DeterministicToolArgumentCorrector()

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

        try:
            invocation_result = await self._execution_boundary.invoke(registration, request)
        except Exception as exc:  # noqa: BLE001 - tool implementations may raise arbitrary execution errors
            invocation_result = ToolInvocationResult.create_failure(
                call_id=request.call_id,
                tool_name=request.tool_name,
                error=str(exc) or type(exc).__name__,
                code=ToolErrorCode.EXECUTION_FAILURE,
            )

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
            )
        return ToolCoordinationResult(
            status=ToolCoordinationStatus.READY_FOR_CONTINUATION,
            decision=decision,
            invocation_result=invocation_result,
            effective_invocation=proposal,
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
