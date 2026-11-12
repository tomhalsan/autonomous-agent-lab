# ADR 0004: Define Minimal Tool Execution Telemetry

Status: Accepted
Date accepted: 2026-11-11
Related issue: https://github.com/tomhalsan/autonomous-agent-lab/issues/28
Supersedes: N/A

## Context

[ADR 0002](0002-bounded-provider-neutral-tool-use-coordination.md) establishes
the bounded tool-use lifecycle: discovery, selection, validation/correction,
authorization, execution, and continuation. [ADR 0003](0003-exact-tool-authorization-and-security-auditing.md)
defines the required authorization decision and audit-delivery boundary before
execution. The current Issue #28 wording requests operational telemetry for
completed or failed tool invocations over the existing event stream and
explicitly defers discovery/selection observability and real-time
in-flight/hung-execution detection.

The generic coordinator has an execution boundary after validation,
authorization, and required security-audit delivery. Its execution result
provides a suitable source for a single terminal telemetry record. The
provider-neutral `AgentEvent` contract already carries structured inference
telemetry, and the orchestrator exposes recent events through the existing
`GetRecentEvents` / `GET /api/events` query path.

The current event contract requires `AgentEvent.state`. Worker and Research
Agent publishers both append events to the shared `agent.events` Redis stream
and mirror their event mappings into `agent.state:{agent_id}` while updating
in-memory current state. Telemetry-only history must not be forced to invent a
current state or overwrite one. The event contract and publishers therefore
need to support an event with no state transition. Existing query/result
fallbacks that use every event as if it carried state must ignore such
telemetry-only records where state is required.

The shared event stream is bounded at approximately 500 entries. Existing
unrelated lifecycle events may contain prompt or result content in
`AgentEvent.detail`; this ADR does not make a platform-wide event privacy
decision and does not expand Issue #28 into a refactor of those events.

Tool arguments, results, errors, and exception text may contain sensitive
content. Tool execution telemetry must explain execution behavior without
persisting that content. This decision crosses the ADR threshold in
the [ADR policy](README.md): it establishes shared event-contract, persistence,
timing, privacy, and historical-event/current-state boundaries.

## Decision

1. **Purpose and minimum necessary data.** Tool execution telemetry records
   only operational facts needed to explain and measure an actual selected-tool
   execution. It is not a second store for content flowing through execution.
   For this telemetry, persist no prompt text, raw tool arguments, raw results,
   credentials, secrets, PII, provider/network response bodies, or arbitrary
   exception text. Interpret “safely sanitized” as omission, not
   redaction-and-retention. This rule is specific to tool execution telemetry;
   it does not assert that every existing `AgentEvent` is free of prompt or
   result content.

2. **Scope and lifecycle.** Attempt to publish exactly one terminal
   tool-execution telemetry event after `ToolExecutionBoundary` returns or an
   ordinary `Exception` is converted to a failed invocation result. Do not
   publish a separate STARTED event. This event describes execution only;
   authorization denial or required-audit failure is not execution because
   the boundary was not invoked.

   Do not emit discovery counts or a generic `discovered` phase. Discovery and
   selection have already occurred upstream. Their observability is deferred
   until a concrete diagnostic requirement exists. If information that should
   have been available upstream would change the selected tool later, treat
   that as an upstream discovery/selection defect, stale-catalog or runtime
   availability concern, or intentional architecture redesign. Do not
   rediscover or reselect after authorization and audit delivery.

   Do not use this increment for real-time in-flight, hung, or process-crash
   detection. Absence of a terminal telemetry record is not proof that
   execution did not occur.

3. **Provider-neutral telemetry payload and structured discrimination.**
   Carry a typed, JSON-compatible `tool_execution_telemetry` payload on the
   existing `AgentEvent`. This dedicated field is the structured discriminator
   for tool execution; consumers, including a future Issue #29 experience,
   must not parse `AgentEvent.detail` to identify or interpret telemetry.
   Any `detail` associated with a telemetry event is fixed and non-sensitive
   and is not required for machine interpretation. Do not add a generic
   `event_type` or `event_kind` solely for Issue #28.

   The payload may contain only:
   - exact registered tool identity (`tool_name`);
   - trusted invoking `agent_id`;
   - the existing invocation `call_id`;
   - `task_id` only when supplied by legitimate trusted runtime context;
   - execution start-time context (`started_at`) when available;
   - boundary duration in milliseconds (`duration_ms`);
   - `argument_size_bytes` when measurable, otherwise `null`;
   - `result_size_bytes` when measurable, otherwise `null`;
   - outcome (`succeeded` or `failed`); and
   - an existing stable `ToolErrorCode` on failure, when applicable.

   Source `agent_id` from the trusted `ToolAuthorizationContext` and `call_id`
   from the `ToolInvocationRequest`; neither may come from selector-controlled
   values. Do not add identifiers, selection rationale, goal text, tool
   descriptions or schemas, payload content, or authorization payloads.

4. **Historical events versus current state.** Tool execution telemetry is
   historical event data, not current agent state. The event contract must
   represent an event without a state transition; `AgentEvent.state` may be
   `None` for telemetry-only events. Serialization must omit the absent state
   (or otherwise represent it without claiming a state value).

   Publishers must append telemetry-only events to the existing event stream
   without changing in-memory `current_state` or writing telemetry/event
   fields to `agent.state:{agent_id}`. Existing current-state transitions
   continue to update state through their normal path. Any existing
   result/query fallback that assumes every event has a state must tolerate or
   ignore telemetry-only events rather than treating them as state changes.

5. **Timing.** Measure only the awaited `ToolExecutionBoundary.invoke`
   operation. Use the repository's monotonic high-resolution
   `time.perf_counter_ns()` approach. Perform operations in this order:

   1. Measure effective post-correction argument size on a best-effort basis.
   2. Capture execution start-time context, if available.
   3. Start `perf_counter_ns()` immediately before invoking the boundary.
   4. Invoke and await `ToolExecutionBoundary`.
   5. Stop `perf_counter_ns()` immediately when that await returns or raises.
   6. Measure returned-result size on a best-effort basis.
   7. Construct and publish the single terminal telemetry event on a
      best-effort basis.

   Report duration in milliseconds from the monotonic interval. Exclude
   discovery, selection, validation/correction, authorization, required
   security-audit delivery, size measurement, telemetry publication, and
   continuation or other post-processing outside the execution boundary.
   Argument/result serialization and telemetry publication must remain
   outside the measured interval.

6. **Payload-size measurements.** Measure only effective post-correction
   `ToolInvocationRequest.arguments` and the returned
   `ToolInvocationResult.output`. The established proposal path validates
   effective arguments as JSON-compatible. The general result contract does
   not guarantee JSON compatibility.

   Where safely and deterministically serializable, use:

   ```python
   json.dumps(
       value,
       allow_nan=False,
       separators=(",", ":"),
       sort_keys=True,
       ensure_ascii=True,
   ).encode("utf-8")
   ```

   The measurement is the encoded value's UTF-8 byte length, excluding the
   invocation and event envelopes. Do not retain serialized content. If the
   value is unsupported or cannot be measured safely and deterministically,
   report `null`; do not estimate or stringify arbitrary Python objects. Do
   not introduce generic datetime, UUID, Decimal, Enum, dataclass, Pydantic,
   or other coercion solely to obtain a size. Measurement failure must never
   alter, fail, replace, or obscure execution or its true result. A successful
   JSON `null` output is measurable; an absent or unmeasurable result size is
   `null`.

7. **Outcomes, exceptions, and cancellation.** Derive success or failure from
   the actual `ToolInvocationResult` and use its stable `ToolErrorCode` when
   present. An ordinary `Exception` raised by `ToolExecutionBoundary` is
   deterministically represented as a failed execution with
   `ToolErrorCode.EXECUTION_FAILURE`. Do not copy `ToolError.message`,
   `ToolError.details`, raw exception text, MCP/provider responses, or network
   error content into telemetry.

   Preserve existing cancellation and control-flow behavior. In particular,
   `asyncio.CancelledError` and equivalent control-flow `BaseException`
   signals propagate; do not classify them as ordinary execution failures.
   Do not add cancellation shielding, retry, or redesign for Issue #28.
   Cancellation or process shutdown may result in no terminal telemetry
   record.

   Do not report authorization denial or required-audit failure as execution
   failure: in those cases `ToolExecutionBoundary` was never invoked.

8. **Best-effort delivery.** Operational telemetry publication is best-effort
   and never execution-gating. Publication failure must not convert success to
   failure, failure to success, discard or replace the true tool result, or
   cause another invocation. It may be independently reported through safe
   operational handling without raw exception text. Telemetry publication
   must not weaken, replace, or satisfy the required security audit.

   This is distinct from ADR 0003: failure to deliver the required
   authorization audit prevents execution. Operational telemetry failure
   does not change the execution outcome.

9. **Existing event and query architecture.** Append telemetry through the
   existing `agent.events` platform event stream and make it queryable through
   the existing recent-event interface. Do not create a telemetry database,
   parallel stream, or separate persistence subsystem. Do not add a dedicated
   task-specific tool telemetry API unless implementation evidence shows the
   existing query cannot satisfy Issue #28.

   The stream is a bounded recent window of approximately 500 entries.
   `GET /api/events` is a bounded recent-events query without task/call
   filtering or pagination. This ADR does not expand retention or query
   capabilities. The existing task-scoped inference telemetry query remains
   inference-specific and is not redefined as a tool query.

10. **Correlation and authorization boundary.** Use exact tool identity,
    trusted agent identity, existing invocation `call_id`, and `task_id` only
    when a trusted runtime caller already supplies it. Do not fabricate task
    identifiers. `ToolAuthorizationContext` remains the trusted security
    context for agent identity and permission scope; do not add `task_id` to
    it. Task correlation is execution/runtime context, not authorization.

    The generic coordinator currently does not receive a task ID. This is an
    acknowledged limitation, not a reason to expand the authorization
    contract or introduce a broad execution-context abstraction. If a
    concrete runtime consumer later needs to pass an existing task ID, a
    narrow optional coordinator/runtime parameter is acceptable when
    supported by repository evidence.

11. **Clean Architecture and scope.** Keep telemetry contracts
    provider-neutral and JSON-compatible. The coordinator exposes an optional
    asynchronous callback receiving the typed telemetry payload, following
    the existing inference-telemetry callback pattern. Runtime composition
    adapts that callback to the existing event publisher; policy does not
    depend on service ports, Redis, or presentation code. Callback publication
    failures are contained without changing execution results. Do not wire
    unrelated runtime consumers merely because the generic coordinator is not
    yet used by them.

12. **Issue #29 boundary.** Provide structured operational evidence that a
    later Prototype Console can present as an execution chain. Do not
    implement UI or make UI-specific architecture decisions here. The
    intended experience explains what happened without duplicating prompt,
    input, or output content.

## Consequences and tradeoffs

Benefits:

- Provides one compact terminal execution record with the tool, trusted
  agent, call, optional task, start-time context, duration, outcome, and safe
  error classification.
- Makes duration semantics precise and separates tool execution from
  discovery, authorization, audit delivery, size measurement, telemetry
  publication, and continuation.
- Reuses the existing JSON event contract, platform stream, and query path.
- Preserves the strict security-audit gate from ADR 0003 while keeping
  operational telemetry best-effort.
- Avoids storing content that may contain credentials, secrets, or PII and
  avoids turning event history into agent state.
- Avoids a second record per invocation and the need to join start and
  terminal events.

Costs and limitations:

- A terminal event cannot provide real-time in-flight, hung, or
  process-crash detection. Those capabilities are deferred until a concrete
  requirement exists.
- Best-effort delivery can omit the terminal event. Absence of a record is
  not proof that execution did not occur.
- The shared event stream is bounded at approximately 500 entries. Separate
  STARTED and COMPLETED events would double tool telemetry event volume per
  invocation and consume the shared recent-event window faster. No claim is
  made about actual production event rates.
- The generic recent-event query has no task/call filters or pagination.
  Task-specific tool queries remain deferred unless evidence requires them.
- `task_id` may be absent because the generic coordinator does not currently
  receive it. No identifier is fabricated to fill that gap.
- Serialized byte counts reveal approximate payload size even though the
  content is omitted. They are retained only as the minimum measurement
  requested and may be `null` for values that cannot be safely serialized.
- The event contract and publishers must represent telemetry-only events
  without mutating state; state-dependent query/result fallbacks must ignore
  those events where appropriate.
- A consumer that does not configure the telemetry callback will not emit
  tool telemetry; adding the boundary does not itself wire existing agent
  runtimes to the generic coordinator.
- This ADR does not wire the generic coordinator into the Research Agent or
  Worker runtime, add discovery/selection telemetry, redesign privacy for
  unrelated lifecycle events, or implement the Prototype Console.

## Alternatives considered

- **Persist raw payloads or sanitized/redacted payload excerpts.** Rejected
  because arbitrary arguments and outputs may embed sensitive content in
  unrecognized keys or free text. Size and outcome metadata provide
  operational evidence without retaining payload values.
- **Emit a discovery-count or `discovered` execution event.** Rejected for
  Issue #28 because discovery and selection occur upstream, and a count at
  execution time does not explain the selection or improve the execution
  record. Upstream discovery/selection observability is deferred until a
  concrete diagnostic need exists.
- **Rediscover or reselect after authorization.** Rejected because it would
  change the selected invocation after its authorization and required audit
  decision; availability changes are execution outcomes, not implicit
  permission to choose another tool.
- **Emit separate STARTED and COMPLETED records.** Rejected for this
  increment because the shared stream is bounded at approximately 500
  entries, two records double event volume per invocation, best-effort
  delivery can leave orphan pairs, and Issue #29 would need correlation/join
  logic. Current Issue #28 requirements do not establish a need for
  real-time in-flight, hung, or process-crash detection.
- **Require successful operational telemetry publication as a precondition to
  execution or successful completion.** Rejected because operational
  observability must not gate or alter execution truth; publication may fail
  independently after the boundary. This differs intentionally from required
  authorization-audit delivery in ADR 0003.
- **Create a separate telemetry store or event stream.** Rejected because
  the existing `AgentEvent`, Redis stream, and recent-event query provide the
  required cross-boundary path for this increment.
- **Write execution telemetry into current agent state.** Rejected because
  telemetry describes historical events, while the existing state hash
  represents current agent state. A telemetry-only event must be appendable
  without a state transition or state-hash write.
- **Add `task_id` to `ToolAuthorizationContext`.** Rejected because task
  correlation is execution/runtime context, while ADR 0003 deliberately
  limits that context to trusted agent identity and permission scope.
- **Add a generic `event_type`/`event_kind` solely for Issue #28.** Rejected
  because the dedicated structured `tool_execution_telemetry` payload
  identifies these events without parsing free text or adding a parallel
  discriminator.
- **Coerce arbitrary result values to obtain a size.** Rejected because
  stringification or generic type conversion can be lossy, unsafe, or
  nondeterministic. Unsupported values have unknown size.
- **Add a dedicated tool telemetry query or Issue #29 UI now.** Rejected
  because the existing event query is sufficient for this increment and
  dedicated query/UI requirements have not been demonstrated. Retention and
  query limitations remain explicit rather than silently expanded.
- **Include full execution telemetry in the task-scoped inference query.**
  Rejected because that query currently returns the newest successful
  inference telemetry and has different semantics. Tool telemetry remains
  queryable through the existing general event feed.
