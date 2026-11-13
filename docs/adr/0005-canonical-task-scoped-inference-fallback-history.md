# ADR 0005: Canonical Task-Scoped Inference Fallback Attempt History

Status: Accepted
Date accepted: 2026-11-13
Related issue: https://github.com/tomhalsan/autonomous-agent-lab/issues/189
Supersedes: N/A

## Context

During the Cloudflare response-parser incident (resolved in commit `8cec470`),
the Cloudflare provider adapter returned HTTP 200 with an unparsed body shape,
correctly raising:

```python
FallbackEligibleInferenceError(
    message="Cloudflare response missing expected text content",
    provider_name="cloudflare",
    reason="invalid_response",
    status_code=200,
)
```

The deterministic fallback traversal engines (`FallbackInferenceProvider` in
Worker and `FallbackResearchSynthesizer` in Research Agent) operated as designed:
they measured attempt latency, recorded a sanitized `ProviderFailureSummary`,
emitted an `InferenceExecutionTelemetry` event (`status="fallback_eligible_failure"`),
emitted an operational `FallbackTransition` event (`failed_provider="cloudflare"`,
`target_provider="gemini"`, `reason="invalid_response"`), and successfully
transitioned to Gemini.

However, subsequent inspection of the production read paths revealed that
task-scoped observability loses this evidence:

1. **Successful fallback filters out failure history:**
   `RedisGateway.task_inference_telemetry()` enforces `validated.status == "succeeded"`.
   Consequently, `/api/tasks/{task_id}/telemetry` discards all prior failed
   attempts, exposing only the single successful attempt. Operators cannot see
   that an initial provider failed, what status code was returned, or how much
   latency was consumed before fallback occurred.

2. **All-providers-failed discards structured summaries:**
   When all candidate providers fail, `AllInferenceProvidersFailedError` contains
   a tuple of sanitized `ProviderFailureSummary` objects. However, both
   `TaskProcessor` and `ResearchProcessor` stringify the exception when publishing
   the terminal error event (`AgentEvent(state="error", detail=str(exc))`),
   reducing the failure to `"All inference providers failed"`. Furthermore, because
   no attempt succeeded, `/api/tasks/{task_id}/telemetry` returns HTTP 404,
   leaving operators with zero structured visibility into provider failures.

3. **Failed-attempt model identity is currently absent:**
   `InferenceExecutionTelemetry` defines `model_name: str | None = None`, but
   in the current implementation, `model_name` is populated only upon successful
   response parsing. When `FallbackEligibleInferenceError` or other errors are
   caught, `model_name` remains `None`. `FallbackTransition` and
   `ProviderFailureSummary` capture only `provider_name`.

4. **Zero-attempt health elimination gap:**
   When per-task concurrent health probing rejects all candidate providers
   before generation begins, `WorkerFallbackCoordinator` and
   `ResearchFallbackCoordinator` raise `AllInferenceProvidersFailedError(failures=())`.
   In this scenario, zero inference attempts occur, and no attempt telemetry or
   transition events are published to the event log.

This decision crosses the ADR threshold defined in [docs/adr/README.md](README.md):
it establishes the canonical read model for inference fallback history, governs
the cross-service representation of multi-attempt execution telemetry, and
determines backward compatibility for the task telemetry API.

The architectural question is:

> What is the canonical task-scoped representation of multi-attempt inference
> execution history, including failed attempts and fallback transitions?

## Operational Event Log Semantics

The shared Redis stream `agent.events` is a **bounded operational event log**,
not a durable event-sourcing store or guaranteed audit ledger:

- **Bounded retention:** In both Worker (`redis_bus.py:99`) and Orchestrator
  (`redis_adapter.py:167`), stream entries are written with `maxlen=500, approximate=True`.
  Entries age out as subsequent events are appended across all agents and tasks.
- **Best-effort delivery:** In both `FallbackInferenceProvider` (`fallback.py:117-137`)
  and `FallbackResearchSynthesizer` (`fallback.py:145-165`), `_notify_telemetry()`
  and `_notify_transition()` catch all exceptions (`except Exception: logger.warning(...)`).
  A transient Redis timeout, network partition, or serialization error drops the
  event without failing the inference execution.
- **Potential duplicates:** Redis transport retries or multi-step publications
  can produce duplicate stream entries with identical content.

Consequently, any task-scoped telemetry history derived from `agent.events` is a
**best-effort event-derived telemetry projection**, not a guaranteed complete or
permanent audit trail.

## Alternatives Considered

### Alternative A — Raw Event-Derived Access

In this approach, `agent.events` remains the only representation of fallback
and attempt telemetry. No task-scoped projection or API expansion is provided.
Consumers (such as the Prototype Console UI or CLI diagnostics) must query the
global event stream via `GET /api/events` and reconstruct task attempt history
client-side.

**Assessment:**
- **Architectural purity:** High backend simplicity; avoids defining new read
  projections or expanding query contracts.
- **Coupling to event internals:** High. External consumers must understand the
  subtle semantics of `AgentEvent.state`, `fallback_transition`, and
  `inference_telemetry`, as well as handle interleaving from concurrent tasks.
- **Query complexity & network overhead:** High. Clients must repeatedly fetch
  and parse large raw event lists to piece together a single task's lifecycle.
- **API & UI implications:** Poor. If a task's events are pushed past the recent
  events fetch limit (e.g. 50–100 events), the UI loses the history even if
  the task is currently being inspected.
- **Retention:** Highly sensitive to stream turnover.

### Alternative B — Event-Derived Task Telemetry Projection (Selected)

In this approach, `agent.events` remains the operational source of evidence,
and the orchestrator read layer (`GetTaskInferenceTelemetry` / `RedisGateway`)
projects the recorded events for a given `task_id` into a coherent, sanitized,
task-scoped attempt-history representation.

The projected representation conceptually contains:
- `task_id: str`
- `attempts: list[InferenceExecutionTelemetry]` (deduplicated, ordered by `attempt_index`)
- `fallback_transitions: list[FallbackTransition]`
- `history_status: str` (`"complete"`, `"partial"`, `"aged_out"`, `"no_inference_recorded"`)
- `final_status: str` (`"succeeded"`, `"all_providers_failed"`, or `"failed"`)
- `inference_telemetry: InferenceExecutionTelemetry | None` (the successful attempt, for backward compatibility)

**Assessment:**
- **Architectural boundaries:** Preserves Clean Architecture. The domain and
  execution layers continue emitting decoupled operational events; the
  orchestrator query layer handles read-model assembly.
- **Zero duplicate storage:** No secondary store to get out of sync or suffer
  dual-write partial failures.
- **Backward compatibility:** Fully backward compatible. Existing clients
  inspecting `envelope.inference_telemetry` continue to receive the successful
  attempt record unchanged.
- **Honest availability:** Explicitly surfaces whether the reconstructed history
  is complete, partial, or aged out.
- **No contract churn:** Requires no changes to `agent_contracts` for Issue #189.

### Alternative C — Persist Separate Inference Execution Summary Read Model

In this approach, the execution coordinator or task processor writes a dedicated
task inference summary document to Redis (e.g., `task.inference:{task_id}`)
upon task completion or terminal failure.

**Assessment:**
- **Read efficiency:** High. Direct O(1) key-value lookup without scanning stream
  entries.
- **Duplicate state & consistency hazards:** High risk. If a worker process
  crashes or is terminated mid-fallback, the separate summary document may never
  be written or updated, creating divergent state between `agent.events` and
  `task.inference:{task_id}`.
- **Operational complexity:** Requires managing TTLs, key lifecycles, and failure
  semantics for a secondary storage structure.
- **Justification for Tiny Agent Lab:** Not justified. The current workload and
  single-stream event architecture do not suffer from stream-read performance
  bottlenecks that warrant secondary write-model caching.

## Design Questions and Technical Evidence

### 1. Is `ProviderFailureSummary` actually required in `AgentEvent`?

**Finding:** No.
Inspection of `FallbackInferenceProvider.generate()` (`fallback.py:200-216`) and
`FallbackResearchSynthesizer.synthesize()` (`fallback.py:223-239`) demonstrates
that *before* appending to `failures: list[ProviderFailureSummary]`, the engine
already emits an `InferenceExecutionTelemetry` event carrying:
- `provider_name`
- `attempt_index`
- `latency_ms`
- `status="fallback_eligible_failure"`
- `failure_reason` (sanitized)
- `status_code` (sanitized)

`ProviderFailureSummary` carries a strict subset of `InferenceExecutionTelemetry`
(`provider_name`, `reason`, `status_code`). Adding `ProviderFailureSummary` to
`AgentEvent` would redundantly duplicate data that is already emitted immediately
prior to the failure. `ProviderFailureSummary` remains an internal, sanitized
in-memory carrier for Python exception propagation via
`AllInferenceProvidersFailedError`. Issue #189 explicitly does not expand
`AgentEvent` to carry `ProviderFailureSummary`.

### 2. Can all-providers-failed be reconstructed from existing events?

**Finding:** Yes, for ordinary provider attempts.
When all candidate providers fail during execution, the stream contains:
- `InferenceExecutionTelemetry` for each attempt (indices `0` through `N-1`);
- `FallbackTransition` for transitions `0 -> 1` through `(N-2) -> (N-1)`;
- A terminal `AgentEvent` with `state="error"` and `detail="All inference providers failed"`.

The failure history is reconstructed by collecting the `InferenceExecutionTelemetry`
records for that `task_id` whose status is `fallback_eligible_failure` (or other
failure statuses).

*Limitation:* If health probing rejects all candidates prior to attempt 0, no
inference attempt events exist. This zero-attempt condition is not an inference
attempt failure and is explicitly out of scope for Issue #189.

### 3. Should `model_name` be added to `ProviderFailureSummary` / `FallbackTransition`?

**Finding:** No.
`InferenceExecutionTelemetry` already defines `model_name: str | None = None`.
Duplicating `model_name` into `FallbackTransition` or `ProviderFailureSummary`
introduces redundancy without diagnostic benefit.

*Live Implementation Note:* In the current codebase, when an attempt fails,
`model_name` is not captured in `InferenceExecutionTelemetry` and is reported as
`None`. Populating model identity on failed attempts requires future adapter and
coordinator changes and must not be solved by polluting transition contracts.

### 4. Value of `FallbackTransition` in the Projection

**Finding:** Retain `FallbackTransition` as projection input.
`FallbackTransition` records transition intent and policy handoff
(`failed_provider -> target_provider`) before the subsequent attempt is invoked.
If a worker process terminates or crashes during a subsequent attempt before its
telemetry can be emitted, `FallbackTransition` provides durable evidence that
the preceding attempt failed, why it failed, and which provider was targeted next.
`FallbackTransition` will not be modified to include `model_name`.

### 5. Ordering, Deduplication, and Conflicting Events

**Finding:** Conservative projection logic.
- **Primary attempt index:** `attempt_index` is an integer assigned monotonically
  by the fallback loop (`0, 1, ...`). It is authoritative for logical attempt
  sequence.
- **Chronological ordering:** Redis stream entry IDs (`<milliseconds>-<sequence>`)
  and `occurred_at` provide secondary chronological sequence.
- **Idempotent deduplication:** If multiple telemetry records carry identical
  `attempt_index`, `provider_name`, `status`, and `latency_ms`, duplicate entries
  are collapsed to a single attempt record.
- **Conflicting records:** If stream entries carry the same `attempt_index` but
  conflicting outcomes (e.g. one `succeeded` and one `failed`), the projection
  does NOT silently overwrite. It records the conflict and marks `history_status="partial"`.

### 6. Conceptual Completeness Semantics

Because telemetry delivery is best-effort, the projection must honestly state
the reliability of the reconstructed history using a small, well-defined vocabulary:

- **`complete`:** Attempt indices form a contiguous sequence `[0, 1, ..., k]`,
  no conflicting duplicates exist, recorded transitions align with attempts, and
  a definitive terminal event exists.
- **`partial`:** Inference events exist for the task, but gaps are detected in
  `attempt_index` sequence, duplicate records conflict, or transitions reference
  unrecorded attempts.
- **`aged_out`:** Durable evidence confirms the task existed (via `task.result:{task_id}`,
  `task.routing:{task_id}`, or `task.status:{task_id}`), but all corresponding
  inference events have rolled off the Redis stream.
- **`no_inference_recorded`:** Zero inference events exist in the stream. (See
  retention limitation below).

### 7. Event Retention and Aged-Out Distinguishability

**Finding:** Partial distinguishability based on durable task records.
- When `task.result:{task_id}` exists (successful task) or `task.routing:{task_id}`
  exists (routed task), the orchestrator can positively confirm that the task ID
  was valid. If `agent.events` contains no inference records for that task, the
  projection reliably reports `history_status="aged_out"`.
- When a task fails before completion, `task.result:{task_id}` is not written.
  If the task was dispatched via orchestrator, `task.status:{task_id}` may exist.
  However, if a failed task leaves no durable Redis keys and its events roll off
  `agent.events`, the system **cannot** distinguish an aged-out failed task from
  an unknown or non-existent task ID. In that scenario, the endpoint returns
  HTTP 404.

### 8. Hard Security Boundary: `AgentEvent.detail` Exclusion

**Finding:** `AgentEvent.detail` must NEVER be projected across the task telemetry boundary.
Repository inspection reveals that existing agent lifecycle events format
unredacted, sensitive content into `AgentEvent.detail`:
- `processor.py:55`: `AgentEvent.create(..., "working", f"Processing: {task.prompt}", ...)`
- `processor.py:77`: `AgentEvent.create(..., "idle", f"Completed: {task.prompt}", ...)`
- `processor.py:85`: `AgentEvent.create(..., "error", str(exc), ...)`
- `research_agent/processor.py:88`: `AgentEvent.create(..., "idle", f"Completed: {result.output}", ...)`

Projecting `AgentEvent.detail` into task telemetry would risk leaking raw prompt
text, research output text, or internal exception messages.
The task inference telemetry projection MUST operate exclusively on validated,
structured fields from `InferenceExecutionTelemetry` and `FallbackTransition`.
`AgentEvent.detail` is strictly excluded.

### 9. Backward Compatibility and Endpoint Strategy

**Finding:** Update `/api/tasks/{task_id}/telemetry` with a backward-compatible envelope.
The current endpoint returns:
```json
{
  "task_id": "...",
  "inference_telemetry": {
    "provider_name": "gemini",
    "attempt_index": 1,
    "latency_ms": 412.5,
    "status": "succeeded",
    "model_name": "gemini-2.5-flash"
  }
}
```
Existing consumers (including the prototype console's `renderInferenceMeta()`)
read `envelope.inference_telemetry`.
The revised endpoint preserves `inference_telemetry` as the winning/successful
attempt (or `null` if no attempt succeeded) and adds sibling fields:
- `history_status: str` (`"complete"`, `"partial"`, `"aged_out"`, `"no_inference_recorded"`)
- `final_status: str` (`"succeeded"`, `"all_providers_failed"`, or `"failed"`)
- `attempts: list[dict]` (chronological attempt telemetries)
- `fallback_transitions: list[dict]` (recorded transitions)

When all providers fail, the endpoint returns **HTTP 200** with:
- `inference_telemetry: null`
- `final_status: "all_providers_failed"`
- `history_status: "complete"` (or `"partial"`)
- `attempts: [...]` (list of failed attempt telemetry records)

HTTP 404 is returned only when the task ID is unknown or no durable/event records
exist for it.

## Decision

1. **Adopt Alternative B (Event-Derived Task Telemetry Projection).**
   The bounded operational event log (`agent.events`) is the source of evidence.
   No secondary persistent inference store will be introduced.

2. **Project Task Inference History in the Orchestrator Read Layer.**
   `GetTaskInferenceTelemetry` and `RedisGateway.task_inference_telemetry` will
   project matching events for `task_id` into an ordered `TaskInferenceHistory`
   structure containing:
   - `task_id: str`
   - `history_status: str` (`"complete"`, `"partial"`, `"aged_out"`, `"no_inference_recorded"`)
   - `final_status: str` (`"succeeded"`, `"all_providers_failed"`, `"failed"`)
   - `attempts: list[InferenceExecutionTelemetry]` (deduplicated, ordered by `attempt_index`)
   - `fallback_transitions: list[FallbackTransition]`
   - `inference_telemetry: InferenceExecutionTelemetry | None` (winning attempt,
     retained for backward compatibility)

3. **Keep `agent_contracts` Unchanged for Issue #189.**
   Existing contracts already define `InferenceExecutionTelemetry` and
   `FallbackTransition`. Do not add `ProviderFailureSummary` to `AgentEvent` and
   do not add `model_name` to `FallbackTransition`.

4. **Enforce Hard Security Boundary on Read Projection.**
   The projection layer must strictly ignore `AgentEvent.detail`. Only sanitized,
   structured telemetry fields (`provider_name`, `model_name`, `latency_ms`,
   `status`, `failure_reason`, `status_code`, token counts, `finish_reason`) may
   cross the read boundary.

5. **Acknowledge Implementation Limitations Honestly.**
   - Model identity is currently `None` on failed attempts.
   - Zero-attempt elimination during health probing produces no inference
     telemetry and is out of scope for Issue #189.
   - History availability is bound to the Redis stream retention buffer (~500 entries).

## Security and Availability

- **Sanitized Reason and Status Validation:**
  Only sanitized values crossing existing contract boundaries are projected.
  `failure_reason` remains constrained to `^[a-z0-9_]+$`. `status_code` remains
  constrained to integers between 100 and 599.
- **Strict Content and Secret Exclusion:**
  The projection operates strictly over `InferenceExecutionTelemetry` and
  `FallbackTransition`. It never reads, extracts, or exposes prompts, system
  instructions, generated text, evidence, tool outputs, authorization headers,
  credentials, or raw error messages.
- **Fault-Tolerant Projection:**
  Errors during projection assembly must never propagate as HTTP 500. If
  telemetry records are partially corrupted, valid records are returned with
  `history_status="partial"`; if no records can be parsed, an empty history or
  404 is returned.
- **Availability Limits:**
  Historical telemetry is available as long as the task's events reside within
  the Redis `agent.events` stream buffer. Once events roll off, durable keys
  permit an `aged_out` response if available, or HTTP 404.

## Consequences and Tradeoffs

### Positive
- **Operational Diagnostic Visibility:** Operators can identify why fallback
  occurred, which provider failed, the HTTP status code, and latency overhead.
- **Honest Telemetry Semantics:** Replaces overstatements of "authoritative"
  completeness with realistic `complete`, `partial`, and `aged_out` statuses.
- **Zero Storage Synchronization Hazards:** Avoids dual-write consistency
  problems between event streams and separate read models.
- **Non-Breaking Evolution:** Existing API callers and Prototype Console UI
  components continue to operate without modification.
- **No Contract Churn:** Avoids premature or speculative changes to shared
  `agent_contracts`.

### Negative / Tradeoffs
- **Stream Scan Cost:** Assembling the read model requires scanning events in
  `agent.events` matching `task_id`. Given the bounded stream size (~500 events),
  this CPU cost is negligible, but it is less direct than a point lookup.
- **Bounded Retention:** Telemetry for older tasks expires when their events
  roll off the Redis stream.
- **Failed Model Name Limitation:** Model names remain `None` for failed attempts
  until provider adapters and exceptions are updated in a separate increment.
