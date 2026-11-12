"""Exact-tool authorization policy and the required security-audit delivery port."""

from typing import Protocol, runtime_checkable

from agent_contracts import (
    ToolAuthorizationContext,
    ToolAuthorizationDecision,
    ToolAuthorizationOutcome,
    ToolAuthorizationReason,
    ToolCapabilityMetadata,
    ToolOperationType,
    ToolPermissionGrant,
    ToolPermissionScope,
    ToolRegistration,
    ToolSecurityAuditEvent,
)


@runtime_checkable
class ToolAuthorizer(Protocol):
    """Trusted policy boundary for deciding whether a registered tool may run."""

    def authorize(
        self, context: ToolAuthorizationContext | None, registration: ToolRegistration
    ) -> ToolAuthorizationDecision: ...


class ExactToolAuthorizer:
    """Authorize only an exact tool-name and operation-type grant."""

    def authorize(
        self, context: ToolAuthorizationContext | None, registration: ToolRegistration
    ) -> ToolAuthorizationDecision:
        tool_name = registration.name
        operation_type = self._operation_type(registration)

        if context is None:
            return self._denied(
                None,
                tool_name,
                operation_type,
                ToolAuthorizationReason.IDENTITY_MISSING,
            )
        if not isinstance(context, ToolAuthorizationContext):
            return self._denied(
                None,
                tool_name,
                operation_type,
                ToolAuthorizationReason.AUTHORIZATION_INPUT_INVALID,
            )
        agent_id = context.agent_id
        if agent_id is not None and not isinstance(agent_id, str):
            return self._denied(
                None,
                tool_name,
                operation_type,
                ToolAuthorizationReason.AUTHORIZATION_INPUT_INVALID,
            )
        if not isinstance(agent_id, str) or not agent_id.strip():
            return self._denied(
                None, tool_name, operation_type, ToolAuthorizationReason.IDENTITY_MISSING
            )
        if operation_type is None:
            return self._denied(
                agent_id,
                tool_name,
                None,
                ToolAuthorizationReason.OPERATION_TYPE_UNCLASSIFIED,
            )

        permission_scope = context.permission_scope
        if permission_scope is None:
            return self._denied(
                agent_id,
                tool_name,
                operation_type,
                ToolAuthorizationReason.PERMISSION_SCOPE_MISSING,
            )
        if not isinstance(permission_scope, ToolPermissionScope):
            return self._denied(
                agent_id,
                tool_name,
                operation_type,
                ToolAuthorizationReason.AUTHORIZATION_INPUT_INVALID,
            )

        grants = permission_scope.grants
        if (
            not isinstance(grants, tuple)
            or any(not isinstance(grant, ToolPermissionGrant) for grant in grants)
            or any(
                not isinstance(grant.tool_name, str)
                or not grant.tool_name.strip()
                or not isinstance(grant.operation_type, ToolOperationType)
                for grant in grants
            )
        ):
            return self._denied(
                agent_id,
                tool_name,
                operation_type,
                ToolAuthorizationReason.AUTHORIZATION_INPUT_INVALID,
            )
        grant_keys = [(grant.tool_name, grant.operation_type) for grant in grants]
        if len(set(grant_keys)) != len(grant_keys):
            return self._denied(
                agent_id,
                tool_name,
                operation_type,
                ToolAuthorizationReason.AUTHORIZATION_INPUT_INVALID,
            )

        if not any(
            grant.tool_name == tool_name and grant.operation_type is operation_type
            for grant in grants
        ):
            reason = (
                ToolAuthorizationReason.MUTATION_GRANT_REQUIRED
                if operation_type is ToolOperationType.STATE_MUTATING
                else ToolAuthorizationReason.TOOL_GRANT_REQUIRED
            )
            return self._denied(agent_id, tool_name, operation_type, reason)

        return ToolAuthorizationDecision(
            outcome=ToolAuthorizationOutcome.ALLOWED,
            agent_id=agent_id,
            tool_name=tool_name,
            operation_type=operation_type,
        )

    @staticmethod
    def _operation_type(registration: ToolRegistration) -> ToolOperationType | None:
        try:
            metadata = registration.definition.metadata
            operation_type = (
                metadata.operation_type if isinstance(metadata, ToolCapabilityMetadata) else None
            )
        except Exception:  # noqa: BLE001 - malformed tool metadata must fail closed
            return None
        return operation_type if isinstance(operation_type, ToolOperationType) else None

    @staticmethod
    def _denied(
        agent_id: str | None,
        tool_name: str,
        operation_type: ToolOperationType | None,
        reason: ToolAuthorizationReason,
    ) -> ToolAuthorizationDecision:
        return ToolAuthorizationDecision(
            outcome=ToolAuthorizationOutcome.DENIED,
            agent_id=agent_id if isinstance(agent_id, str) else None,
            tool_name=tool_name,
            operation_type=operation_type,
            reason_code=reason,
        )


@runtime_checkable
class ToolSecurityAuditSink(Protocol):
    """Required delivery boundary for sanitized authorization-decision events."""

    async def deliver(self, event: ToolSecurityAuditEvent) -> None: ...
