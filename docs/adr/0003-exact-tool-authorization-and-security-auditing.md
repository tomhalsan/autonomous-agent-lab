# ADR 0003: Define Exact-Tool Authorization and Security Auditing

Status: Accepted
Date accepted: 2026-11-11
Related issue: https://github.com/tomhalsan/autonomous-agent-lab/issues/27
Supersedes: N/A

## Context

[ADR 0002](0002-bounded-provider-neutral-tool-use-coordination.md) establishes
the bounded provider-neutral tool-use lifecycle and preserves an authorization
seam after argument validation and before execution. It assigns permission
grants, authorization decisions, denial behavior, auditability, and
security-scope enforcement to Issue #27. The current tool contracts describe
tool capabilities, including operation type and risk, but do not define
agent-specific permission state.

Issue #27 requires invocations to be checked against an agent's permitted
capability scope, explicit permission for state-mutating operations, blocked
unauthorized execution, and security audit events for rejected attempts. Its
security motivation mentions file access, network egress, and destructive
mutations. Invocation authorization can prevent a tool call through the
coordinator, but it does not constrain what an already-invoked tool can do.

Authorization and security auditing establish consequential security
architecture under the threshold in [the ADR policy](README.md). This record
was accepted on 2026-11-11 and should guide implementation.

## Decision

1. Preserve the bounded tool-use lifecycle and the separation between
   discovery, selection, validation/correction, authorization, execution, and
   continuation established by ADR 0002. Discovery, selection, registration,
   and argument validity are not authorization.

2. Keep authorization state separate from tool capability metadata.
   `ToolCapabilityMetadata` describes a tool; it must not carry agent grants,
   credentials, JWTs, or transport-authentication state. Advertised
   `AgentCapabilityMetadata` is not an authorization source.

3. Begin with exact-tool grants. A grant identifies one exact registered tool
   identity and the operation type it permits. The current exact tool name is
   the identity available at this boundary; a renamed tool therefore requires
   a corresponding grant update. Do not add category-wide grants or wildcard
   grants in Issue #27.

4. Treat the agent identity and permission scope as trusted runtime input
   supplied by host composition. They must not originate from model output,
   tool selection, invocation arguments, or advertised agent capabilities.
   This decision establishes the authority boundary; it does not add
   authentication, credential management, or persistent grant storage.

5. Apply default-deny semantics. Missing identity, missing permission scope,
   malformed authorization input, absent or invalid operation classification,
   or any inability to establish authorization must not result in invocation.
   Read-only operations also require an exact matching grant; read-only is not
   an implicit allow.

6. Require an explicit grant for the `state_mutating` operation type. For this
   increment, the existing `state_mutating` classification is the mutation
   boundary. Do not introduce a separate destructive classification unless
   later evidence shows the existing contract cannot express the requirement.
   Keep operation type orthogonal to `ToolRiskLevel`; risk alone never grants
   or denies permission.

7. Authorize the effective proposed invocation only after argument validation
   and any bounded correction, using the trusted identity/scope and the
   registered tool's operation classification. Perform this check immediately
   before the existing execution boundary. Authorization policy decides
   whether invocation is permitted; the execution boundary remains responsible
   for invoking the tool and must not derive grants from tool descriptions.

8. Represent authorization as an explicit allow or deny outcome, distinct from
   the coordination result and from tool execution. A denial produces a
   distinct observable denied result and prevents the execution boundary from
   being called. Use stable, non-sensitive denial reason codes, including:
   - `identity_missing`;
   - `permission_scope_missing`;
   - `authorization_input_invalid`;
   - `operation_type_unclassified`;
   - `tool_grant_required`;
   - `mutation_grant_required`; and
   - `authorization_unavailable`.

   Reasons must describe the policy outcome, not include raw exception,
   provider, or tool content. Implementations may refine the code set while
   preserving stable meanings.

9. Treat security auditing as a separate boundary that records authorization
   decisions, not as a synonym for execution telemetry. The architecture must
   allow both denied decisions and authorized security-relevant decisions,
   especially state-mutating authorizations, to be audited. An authorization
   audit event may include trusted agent identity, invocation/call ID, exact
   tool identity, known operation type, authorization outcome, and a stable
   reason code. Include other identifiers only when the runtime already
   provides them and they are appropriate for the audit purpose.

   Do not include prompts, tool arguments, tool output, credentials, secrets,
   or raw provider/network error content. A denial event records that execution
   was not attempted. An allowed decision records permission to proceed; it
   does not assert that execution started, succeeded, or had any particular
   effect.

10. Keep authorization outcome, execution state, and audit-delivery outcome
    independently observable. Audit delivery must never change DENY into
    ALLOW or hide the denial. If delivery of a required authorization audit
    event fails, do not invoke the execution boundary. Report the authorization
    result unchanged, report audit delivery as failed, and report execution as
    not attempted. This applies to allowed as well as denied decisions; an
    allowed decision blocked because its required audit event could not be
    delivered remains an authorization allow, not an authorization denial.
    Surface the audit-delivery failure rather than silently dropping it.

11. Limit Issue #27 security auditing to authorization decisions and their
    security-relevant context. It may record that a decision was allowed or
    denied, and that a denied operation was not attempted. It must not claim
    execution success or failure. Tool execution telemetry, including
    invocation lifecycle, duration, output, and execution outcome, belongs to
    Issue #28 and must not be pulled into this increment. An authorization
    audit event for an allowed state-mutating operation is evidence of the
    authorization decision, not evidence that the mutation occurred.

12. Keep contracts provider-neutral, agent-type-neutral where appropriate,
    Clean Architecture compliant, and JSON-compatible where they cross
    boundaries. Keep authorization policy out of provider, transport, and tool
    adapters. A configured audit-delivery adapter is infrastructure; no
    particular storage or telemetry platform is selected here.

13. Do not expand Issue #27 into credential management, JWT or
    agent-to-agent authentication, persistent grant storage, human approval
    workflows, category-wide authorization, filesystem/network sandboxing, a
    general telemetry platform, or arguments-sensitive authorization unless
    existing architecture is shown to require it.

## Consequences and tradeoffs

Benefits:

- Establishes a default-deny authorization check at the explicit seam from
  ADR 0002, without coupling permission to tool discovery or selection.
- Requires read-only access to be granted and state mutation to be explicitly
  granted, while keeping operation classification distinct from risk.
- Makes denied outcomes, authorization decisions, and audit-delivery failures
  independently observable.
- Allows security review of authorized state-mutating decisions without
  treating an authorization event as proof of execution or duplicating Issue
  #28 telemetry.
- Avoids coupling grants to a provider, agent type, category, credential, or
  transport protocol.

Costs and limitations:

- Exact tool-name grants require updates when a tool is renamed and do not
  provide a stable capability identifier across such changes.
- Requiring delivery of an authorization audit event before invocation makes
  audit availability part of execution availability. This is intentional for
  the initial security boundary; an audit failure does not alter the
  authorization verdict.
- The runtime must supply trustworthy identity and scope. This ADR does not
  authenticate that runtime input or define how grants are provisioned.
- Tool operation classification is descriptive metadata. Fail-closed handling
  of missing classification does not prove that a tool's declaration is
  truthful or that its implementation has no hidden side effects.
- Invocation authorization is not runtime sandboxing. It does not prevent an
  invoked tool from accessing files, networks, or other resources beyond its
  intended behavior.
- The initial model does not express argument-dependent permissions or
  distinguish destructive operations from other state mutations.
- An authorization audit event for an allowed mutation records permission,
  not execution. Execution evidence and outcomes remain for Issue #28.

## Alternatives considered

- **Treat registration, discovery scope, or advertised agent capabilities as
  permission.** Rejected because visibility and advertised capability are not
  trusted authorization state, and this would collapse boundaries established
  by ADR 0002.
- **Grant entire categories or use wildcards.** Rejected for Issue #27 because
  new tools could become authorized implicitly; exact grants make the scope
  explicit and reviewable.
- **Allow read-only operations without grants.** Rejected because every
  invocation must be checked against the agent's permitted scope.
- **Infer permission from `ToolRiskLevel`.** Rejected because operation type
  and risk are orthogonal; risk is descriptive, not an authorization grant.
- **Combine authorization, execution, and audit delivery into a single
  success/failure value.** Rejected because a denial, an allowed decision
  blocked by audit-delivery failure, and an execution failure are distinct
  security and lifecycle outcomes.
- **Treat invocation authorization as filesystem or network containment.**
  Rejected because checking permission before a tool call cannot sandbox the
  effects of the invoked implementation.
- **Include tool outputs and execution results in Issue #27 audit events.**
  Rejected because it risks leaking sensitive content and overlaps the
  execution telemetry scope reserved for Issue #28.
