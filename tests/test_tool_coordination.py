"""Bounded tool coordination, schema validation, correction, and dynamic discovery."""

import ast
import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from agent_contracts import (
    ToolAuthorizationContext,
    ToolCapabilityMetadata,
    ToolDefinition,
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
    ToolSelectionKind,
    ToolSelector,
)
from agent_infrastructure import InMemoryToolRegistry
from agent_policy import (
    BoundedToolCoordinator,
    DeterministicToolArgumentCorrector,
    DeterministicToolSelector,
    ToolArgumentCorrector,
    ToolArgumentValidationResult,
    ToolCoordinationResult,
    ToolCoordinationStatus,
    ToolDiscoveryScope,
    ToolDiscoveryService,
    TrustedCompositionToolInvoker,
)


class _Tool:
    def __init__(
        self,
        name: str = "web_search",
        description: str = "Search the web",
        parameters: dict[str, object] | None = None,
        result: ToolInvocationResult | None = None,
        operation_type: ToolOperationType | None = ToolOperationType.READ_ONLY,
    ) -> None:
        self._definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters
            if parameters is not None
            else {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            metadata=(
                ToolCapabilityMetadata(
                    category="research",
                    operation_type=operation_type,
                    risk_level=ToolRiskLevel.LOW,
                    idempotency=ToolIdempotency.IDEMPOTENT,
                )
                if operation_type is not None
                else None
            ),
        )
        self.result = result
        self.requests: list[ToolInvocationRequest] = []

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        self.requests.append(request)
        if self.result is not None:
            return self.result
        return ToolInvocationResult.create_success(
            call_id=request.call_id,
            tool_name=request.tool_name,
            output={"evidence": "observed result"},
        )


class _FixedSelector:
    def __init__(self, decision: object) -> None:
        self.decision = decision
        self.calls = 0

    async def select(self, goal: str, catalog: Sequence[ToolRegistration]) -> ToolSelectionDecision:
        self.calls += 1
        return self.decision  # type: ignore[return-value]


class _CountingCorrector:
    def __init__(
        self,
        result: ToolInvocationProposal | None = None,
        delegate: DeterministicToolArgumentCorrector | None = None,
    ) -> None:
        self.result = result
        self.delegate = delegate
        self.calls = 0

    async def correct(
        self,
        registration: ToolRegistration,
        proposal: ToolInvocationProposal,
        validation: ToolArgumentValidationResult,
    ) -> ToolInvocationProposal | None:
        self.calls += 1
        if self.delegate is not None:
            return await self.delegate.correct(registration, proposal, validation)
        return self.result


def _registration(tool: _Tool, category: str = "research") -> ToolRegistration:
    return ToolRegistration(tool, category=category)


class _RecordingAuditSink:
    def __init__(self) -> None:
        self.events: list[ToolSecurityAuditEvent] = []

    async def deliver(self, event: ToolSecurityAuditEvent) -> None:
        self.events.append(event)


def _coordinator(
    registry: InMemoryToolRegistry,
    selector: ToolSelector,
    *,
    corrector: ToolArgumentCorrector | None = None,
) -> BoundedToolCoordinator:
    return BoundedToolCoordinator(
        discovery=ToolDiscoveryService(registry),
        scope=ToolDiscoveryScope(categories={"research"}),
        selector=selector,
        execution_boundary=TrustedCompositionToolInvoker(),
        corrector=corrector,
        authorization_context=ToolAuthorizationContext(
            agent_id="test-agent",
            permission_scope=ToolPermissionScope(
                [ToolPermissionGrant("web_search", ToolOperationType.READ_ONLY)]
            ),
        ),
        audit_sink=_RecordingAuditSink(),
    )


@pytest.mark.anyio
async def test_dynamic_registration_is_discovered_by_same_coordinator() -> None:
    registry = InMemoryToolRegistry()
    selector = DeterministicToolSelector()
    coordinator = _coordinator(registry, selector)

    first = await coordinator.coordinate("search the web for climate research")
    tool = _Tool()
    registration = registry.register(tool, category="research")
    second = await coordinator.coordinate("search the web for climate research")

    assert first.status is ToolCoordinationStatus.NO_TOOL_NEEDED
    assert first.invocation_result is None
    assert first.effective_invocation is None
    assert second.status is ToolCoordinationStatus.READY_FOR_CONTINUATION
    assert second.invocation_result is not None
    assert second.invocation_result.output == {"evidence": "observed result"}
    assert second.decision is not None and second.decision.invocation is not None
    assert second.decision.invocation.tool_name == registration.name
    assert len(tool.requests) == 1
    assert tool.requests[0].arguments == {"query": "search the web for climate research"}


@pytest.mark.anyio
async def test_no_tool_and_empty_catalog_never_execute() -> None:
    registry = InMemoryToolRegistry()
    coordinator = _coordinator(registry, DeterministicToolSelector())

    result = await coordinator.coordinate("do something unrelated")

    assert result.status is ToolCoordinationStatus.NO_TOOL_NEEDED
    assert result.decision == ToolSelectionDecision.no_tool_needed()
    assert result.invocation_result is None
    assert result.effective_invocation is None


@pytest.mark.anyio
async def test_no_matching_tool_in_nonempty_catalog_never_executes() -> None:
    registry = InMemoryToolRegistry()
    tool = _Tool()
    registry.register(tool, category="research")
    coordinator = _coordinator(registry, DeterministicToolSelector())

    result = await coordinator.coordinate("add two integers")

    assert result.status is ToolCoordinationStatus.NO_TOOL_NEEDED
    assert tool.requests == []


@pytest.mark.anyio
async def test_invalid_selector_result_is_pre_execution_failure() -> None:
    registry = InMemoryToolRegistry()
    tool = _Tool()
    registry.register(tool, category="research")
    coordinator = _coordinator(registry, _FixedSelector(None))

    result = await coordinator.coordinate("search the web")

    assert result.status is ToolCoordinationStatus.PRE_EXECUTION_FAILURE
    assert "malformed selection decision" in (result.error or "")
    assert tool.requests == []


@pytest.mark.anyio
async def test_selected_tool_must_be_present_in_current_catalog() -> None:
    registry = InMemoryToolRegistry()
    tool = _Tool()
    registry.register(tool, category="research")
    selector = _FixedSelector(
        ToolSelectionDecision.proposed_invocation("missing_tool", {"query": "x"})
    )

    result = await _coordinator(registry, selector).coordinate("search the web")

    assert result.status is ToolCoordinationStatus.PRE_EXECUTION_FAILURE
    assert "not uniquely present" in (result.error or "")
    assert tool.requests == []


@pytest.mark.anyio
async def test_duplicate_catalog_identities_fail_before_selection_or_execution() -> None:
    tool = _Tool()
    registration = _registration(tool)

    class _DuplicateReader:
        def list_all(self) -> list[ToolRegistration]:
            return [registration, registration]

    selector = _FixedSelector(ToolSelectionDecision.no_tool_needed())
    coordinator = BoundedToolCoordinator(
        ToolDiscoveryService(_DuplicateReader()),  # type: ignore[arg-type]
        ToolDiscoveryScope(categories={"research"}),
        selector,  # type: ignore[arg-type]
        TrustedCompositionToolInvoker(),
    )

    result = await coordinator.coordinate("search the web")

    assert result.status is ToolCoordinationStatus.PRE_EXECUTION_FAILURE
    assert "duplicate tool identities" in (result.error or "")
    assert selector.calls == 0
    assert tool.requests == []


@pytest.mark.anyio
async def test_discovery_failure_is_reported_before_selection() -> None:
    class _FailingReader:
        def list_all(self) -> list[ToolRegistration]:
            raise RuntimeError("registry unavailable")

    selector = _FixedSelector(ToolSelectionDecision.no_tool_needed())
    coordinator = BoundedToolCoordinator(
        ToolDiscoveryService(_FailingReader()),  # type: ignore[arg-type]
        ToolDiscoveryScope(),
        selector,  # type: ignore[arg-type]
        TrustedCompositionToolInvoker(),
    )

    result = await coordinator.coordinate("search the web")

    assert result.status is ToolCoordinationStatus.PRE_EXECUTION_FAILURE
    assert "Tool discovery failed: registry unavailable" == result.error
    assert selector.calls == 0


@pytest.mark.anyio
async def test_bounded_correction_removes_only_disallowed_extra_argument_once() -> None:
    registry = InMemoryToolRegistry()
    tool = _Tool()
    registry.register(tool, category="research")
    decision = ToolSelectionDecision.proposed_invocation(
        "web_search", {"query": "weather", "unexpected": "discard me"}
    )
    selector = _FixedSelector(decision)
    corrector = _CountingCorrector(delegate=DeterministicToolArgumentCorrector())

    result = await _coordinator(registry, selector, corrector=corrector).coordinate(
        "search the web"
    )

    assert result.status is ToolCoordinationStatus.READY_FOR_CONTINUATION
    assert corrector.calls == 1
    assert len(tool.requests) == 1
    assert result.decision is decision
    selected_decision = result.decision
    assert selected_decision is not None
    assert selected_decision.kind is ToolSelectionKind.PROPOSED_INVOCATION
    selected_invocation = selected_decision.invocation
    assert selected_invocation is not None
    assert selected_invocation.arguments == {
        "query": "weather",
        "unexpected": "discard me",
    }
    assert result.effective_invocation is not None
    assert result.effective_invocation.arguments == {"query": "weather"}
    assert tool.requests[0].arguments == result.effective_invocation.arguments


@pytest.mark.anyio
async def test_failed_correction_is_attempted_once_and_never_executes() -> None:
    registry = InMemoryToolRegistry()
    tool = _Tool()
    registry.register(tool, category="research")
    proposal = ToolInvocationProposal("web_search", {})
    selector = _FixedSelector(ToolSelectionDecision.proposed_invocation("web_search", {}))
    corrector = _CountingCorrector(proposal)

    result = await _coordinator(registry, selector, corrector=corrector).coordinate(
        "search the web"
    )

    assert result.status is ToolCoordinationStatus.PRE_EXECUTION_FAILURE
    assert "Missing required argument 'query'" in (result.error or "")
    assert result.effective_invocation is None
    assert corrector.calls == 1
    assert tool.requests == []


@pytest.mark.anyio
async def test_default_correction_failure_does_not_execute_invalid_request() -> None:
    registry = InMemoryToolRegistry()
    tool = _Tool()
    registry.register(tool, category="research")
    selector = _FixedSelector(ToolSelectionDecision.proposed_invocation("web_search", {}))

    result = await _coordinator(registry, selector).coordinate("search the web")

    assert result.status is ToolCoordinationStatus.PRE_EXECUTION_FAILURE
    assert "Missing required argument 'query'" in (result.error or "")
    assert tool.requests == []


@pytest.mark.anyio
async def test_selector_does_not_invent_unconstructable_required_arguments() -> None:
    tool = _Tool(
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        }
    )
    registry = InMemoryToolRegistry()
    registry.register(tool, category="research")

    result = await _coordinator(registry, DeterministicToolSelector()).coordinate("search the web")

    assert result.status is ToolCoordinationStatus.PRE_EXECUTION_FAILURE
    assert "Missing required argument 'limit'" in (result.error or "")
    assert tool.requests == []


@pytest.mark.anyio
async def test_tool_failure_is_distinct_from_pre_execution_failure() -> None:
    class _FailingTool(_Tool):
        async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
            self.requests.append(request)
            return ToolInvocationResult.create_failure(
                call_id=request.call_id,
                tool_name=request.tool_name,
                error="provider failed",
            )

    tool = _FailingTool()
    registry = InMemoryToolRegistry()
    registry.register(tool, category="research")
    decision = ToolSelectionDecision.proposed_invocation(
        "web_search", {"query": "weather", "unexpected": "discard me"}
    )
    selector = _FixedSelector(decision)
    corrector = _CountingCorrector(delegate=DeterministicToolArgumentCorrector())

    result = await _coordinator(registry, selector, corrector=corrector).coordinate(
        "search the web"
    )

    assert result.status is ToolCoordinationStatus.TOOL_INVOCATION_FAILURE
    assert result.invocation_result is not None
    assert result.invocation_result.error is not None
    assert result.invocation_result.error.message == "provider failed"
    assert result.decision is decision
    selected_decision = result.decision
    assert selected_decision is not None
    selected_invocation = selected_decision.invocation
    assert selected_invocation is not None
    assert selected_invocation.arguments == {
        "query": "weather",
        "unexpected": "discard me",
    }
    assert result.effective_invocation is not None
    assert result.effective_invocation.arguments == {"query": "weather"}
    assert tool.requests[0].arguments == result.effective_invocation.arguments
    assert corrector.calls == 1
    assert len(tool.requests) == 1


@pytest.mark.anyio
async def test_cancellation_from_execution_boundary_propagates_without_retry() -> None:
    class _CancellingInvoker:
        def __init__(self) -> None:
            self.calls = 0

        async def invoke(
            self, registration: ToolRegistration, request: ToolInvocationRequest
        ) -> ToolInvocationResult:
            self.calls += 1
            raise cancellation

    tool = _Tool()
    registry = InMemoryToolRegistry()
    registry.register(tool, category="research")
    cancellation = asyncio.CancelledError("cancel invocation")
    invoker = _CancellingInvoker()
    coordinator = BoundedToolCoordinator(
        ToolDiscoveryService(registry),
        ToolDiscoveryScope(categories={"research"}),
        _FixedSelector(
            ToolSelectionDecision.proposed_invocation("web_search", {"query": "weather"})
        ),  # type: ignore[arg-type]
        invoker,  # type: ignore[arg-type]
        authorization_context=ToolAuthorizationContext(
            "test-agent",
            ToolPermissionScope([ToolPermissionGrant("web_search", ToolOperationType.READ_ONLY)]),
        ),
        audit_sink=_RecordingAuditSink(),
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        await coordinator.coordinate("search the web")

    assert raised.value is cancellation
    assert invoker.calls == 1
    assert tool.requests == []


@pytest.mark.parametrize(
    ("status", "decision", "invocation_result", "effective_invocation", "error"),
    [
        (
            ToolCoordinationStatus.NO_TOOL_NEEDED,
            ToolSelectionDecision.no_tool_needed(),
            None,
            ToolInvocationProposal("web_search", {"query": "weather"}),
            None,
        ),
        (
            ToolCoordinationStatus.PRE_EXECUTION_FAILURE,
            ToolSelectionDecision.proposed_invocation("web_search", {"query": "weather"}),
            None,
            ToolInvocationProposal("web_search", {"query": "weather"}),
            "invalid arguments",
        ),
        (
            ToolCoordinationStatus.READY_FOR_CONTINUATION,
            ToolSelectionDecision.proposed_invocation("web_search", {"query": "weather"}),
            ToolInvocationResult.create_success("call", "web_search", {}),
            None,
            None,
        ),
        (
            ToolCoordinationStatus.TOOL_INVOCATION_FAILURE,
            ToolSelectionDecision.proposed_invocation("web_search", {"query": "weather"}),
            ToolInvocationResult.create_failure("call", "web_search", "failed"),
            None,
            "failed",
        ),
    ],
)
def test_coordination_result_rejects_missing_or_contradictory_effective_invocation(
    status: ToolCoordinationStatus,
    decision: ToolSelectionDecision,
    invocation_result: ToolInvocationResult | None,
    effective_invocation: ToolInvocationProposal | None,
    error: str | None,
) -> None:
    with pytest.raises(ValueError):
        ToolCoordinationResult(
            status=status,
            decision=decision,
            invocation_result=invocation_result,
            error=error,
            effective_invocation=effective_invocation,
        )


@pytest.mark.anyio
async def test_tool_exception_becomes_invocation_failure() -> None:
    class _RaisingTool(_Tool):
        async def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
            self.requests.append(request)
            raise RuntimeError("execution broke")

    tool = _RaisingTool()
    registry = InMemoryToolRegistry()
    registry.register(tool, category="research")
    selector = _FixedSelector(
        ToolSelectionDecision.proposed_invocation("web_search", {"query": "weather"})
    )

    result = await _coordinator(registry, selector).coordinate("search the web")

    assert result.status is ToolCoordinationStatus.TOOL_INVOCATION_FAILURE
    assert result.invocation_result is not None
    assert result.invocation_result.error is not None
    assert result.invocation_result.error.message == "execution broke"
    assert len(tool.requests) == 1


@pytest.mark.anyio
async def test_malformed_execution_result_becomes_invocation_failure() -> None:
    class _MalformedInvoker:
        calls = 0

        async def invoke(
            self, registration: ToolRegistration, request: ToolInvocationRequest
        ) -> ToolInvocationResult:
            self.calls += 1
            return None  # type: ignore[return-value]

    tool = _Tool()
    registry = InMemoryToolRegistry()
    registry.register(tool, category="research")
    invoker = _MalformedInvoker()
    coordinator = BoundedToolCoordinator(
        ToolDiscoveryService(registry),
        ToolDiscoveryScope(categories={"research"}),
        _FixedSelector(
            ToolSelectionDecision.proposed_invocation("web_search", {"query": "weather"})
        ),  # type: ignore[arg-type]
        invoker,  # type: ignore[arg-type]
        authorization_context=ToolAuthorizationContext(
            "test-agent",
            ToolPermissionScope([ToolPermissionGrant("web_search", ToolOperationType.READ_ONLY)]),
        ),
        audit_sink=_RecordingAuditSink(),
    )

    result = await coordinator.coordinate("search the web")

    assert result.status is ToolCoordinationStatus.TOOL_INVOCATION_FAILURE
    assert result.invocation_result is not None
    assert result.invocation_result.error is not None
    assert "malformed tool result" in result.invocation_result.error.message
    assert invoker.calls == 1
    assert tool.requests == []


def test_generic_coordination_modules_do_not_import_infrastructure_or_providers() -> None:
    import agent_policy.tool_authorization as authorization_module
    import agent_policy.tool_coordination as coordination_module
    import agent_policy.tool_validation as validation_module

    for module in (authorization_module, coordination_module, validation_module):
        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert not any(
            name == "mcp"
            or name.startswith(
                (
                    "mcp.",
                    "agent_infrastructure",
                    "research_agent",
                    "google",
                    "openai",
                    "groq",
                    "ollama",
                    "cloudflare",
                    "anthropic",
                    "tavily",
                )
            )
            for name in imports
        )
