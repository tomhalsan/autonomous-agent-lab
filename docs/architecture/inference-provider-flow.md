# Tiny Agent Lab Architecture Handbook #1: Provider Selection, Fallback, and Inference Telemetry

> **Subsystem Architecture Specification • Release v0.4.0 (V4 Reference Baseline)**
> *Baseline: Released Runtime Behavior (Git tag `v0.4.0` / Commit `62ca661`)*

---

## Architectural Question Answered

> **"What happens from the moment a Worker or Research Agent needs inference until a successful result or terminal provider failure is produced, including provider selection, fallback traversal, attempt telemetry, and Prototype Console observability?"**

This document and its accompanying standalone vector diagram detail the inner lifecycle of inference resilience in Tiny Agent Lab. Sitting one level below the high-level platform architecture, this specification focuses on how autonomous agents dynamically probe, rank, invoke, measure, and gracefully fall back across an arbitrary chain of AI inference providers without single-provider fragility or hardcoded vendor assumptions.

---

## Authoritative Architecture Diagram (V4)

The standalone vector diagram below illustrates the numbered primary flow, the explicit `TASK COMPLETE` success endpoint, the fallback retry loop, and parallel operational telemetry streams:

![Tiny Agent Lab Provider Selection, Fallback, and Inference Telemetry](inference-provider-flow.svg)

*Direct SVG link:* [`inference-provider-flow.svg`](inference-provider-flow.svg)

---

## Primary Execution Grammar (The Main Story)

The core architecture follows a single dominant numbered execution grammar that reads naturally from the upper-left toward the result:

```
1. Inference Needed
        ↓
2. Check Provider Health
        ↓
3. Select Eligible Provider
        ↓
4. Invoke Provider
        ↓
5. Succeeded?
    ↙           ↘
 YES             NO
  ↓               ↓
6. Return Result   7. Fallback Allowed?
  ↓                  ↙             ↘
TASK COMPLETE       YES             NO
                     ↓               ↓
                8. Next Provider   9. Stop / Fail
                     │
                     └──────────────↺ 4. Invoke Provider
```

### Step 1: Inference Needed (Agent)
- **Shared Semantics**: General Worker and Research Agent share identical inference selection, fallback, and telemetry contracts. Worker inference generates direct completions (`provider.generate()`), while Research Agent inference conditions synthesis on retrieved research evidence (`provider.synthesize()`).
- **Initiation**: The agent receives a task from the transport queue (`TaskRequested`: `task_id`, `prompt`).
- **Coordinator Lifecycle**: `WorkerFallbackCoordinator` or `ResearchFallbackCoordinator` initializes per-task execution.

### Step 2: Check Provider Health (Selection)
- **Configured Priority Chain**: Providers are configured in ordered priority: `Provider A → Provider B → Provider C → ...` (e.g., `cloudflare,gemini`). Configured providers are validated and instantiated once at service startup without performing network pings or inference calls.
- **Concurrent Per-Task Probing**: Before selecting a provider for a task, the coordinator executes `check_health()` concurrently across all configured providers using `asyncio.gather(*probes, return_exceptions=True)`.
- **Failure Isolation**: Each probe is bounded by an outer timeout. If one provider probe times out or crashes, remaining probes continue unaffected.
- **Truthful Snapshots**: Probes return point-in-time `InferenceProviderHealth` snapshots. A failed or timed-out probe produces **no snapshot**; missing health is treated as unknown. **The system never assumes or fabricates a healthy state.**

### Step 3: Select Eligible Provider (Selection)
- **Deterministic Priority Policy**: `DeterministicProviderSelector` evaluates the configured preferences and authentic health snapshots using `SelectionStrategy.PRIORITY`.
- **Ranking Invariants**: All verified `HEALTHY` candidates are ranked first in configured priority order, followed by any `DEGRADED` candidates in configured priority order.
- **Exclusion**: Providers that are `RATE_LIMITED`, `QUOTA_EXHAUSTED`, `UNAVAILABLE`, or unknown (missing health snapshot) are strictly excluded.
- **Outcome**: Produces an immutable, ordered candidate sequence for traversal.

### Step 4: Invoke Provider (Execution)
- **Active Candidate Invocation**: The coordinator attempts inference against the current candidate.
- **Lazy Resolution**: Provider adapters are resolved lazily on demand (`resolver(candidate.provider_name)`) only when reached. Target providers are resolved exactly once.
- **Examples in v0.4.0**: Cloudflare Workers AI and Google Gemini serve as concrete production adapters. Stubs and mock providers support deterministic unit testing.

### Step 5: Succeeded? (Decision)
- **Verification**: Evaluates whether the model returned a valid, complete inference response.
- **Branch YES**: Advances directly to **Step 6 (Return Result)**.
- **Branch NO**: Evaluates failure cause and advances to **Step 7 (Fallback Allowed?)**.

### Step 6: Return Result & TASK COMPLETE (Success Path)
- **Output Processing**: Normalizes response text and metadata.
- **Persistence**: The agent constructs a `TaskResult` (or `ResearchResult` with citations and attribution sources) and persists it to the `ResultStore`.
- **Explicit Success Endpoint (`TASK COMPLETE`)**: Execution reaches an unmistakable terminal destination. The task is fully complete, the result is saved, and the agent state transitions from `working` to `idle`.

### Step 7: Fallback Allowed? (Decision)
- **Eligibility Evaluation**: Determines whether execution may proceed to an alternative provider:
  1. Is the failure transient and provider-specific (e.g., rate limit, timeout, server outage, malformed response)?
  2. Does another candidate provider remain in the prioritized list?
- **Branch YES**: Advances to **Step 8 (Next Provider)**.
- **Branch NO**: Advances to **Step 9 (Stop / Fail)**.

### Step 8: Next Provider (Fallback Loop)
- **Candidate Advancement**: Selects the next eligible candidate in the priority chain (e.g., advancing from Provider A to Provider B).
- **Audit Event**: Emits a `FallbackTransition` operational event recording the switch reason and target provider.
- **Loopback Execution**: Curves directly back into **Step 4 (Invoke Provider)** to attempt inference with the newly resolved provider.

### Step 9: Stop / Fail (Terminal Failure)
- **Execution Halt**: Traversal immediately terminates without further retries.
- **Terminal Error State**: Occurs when the failure is permanent (non-fallback client error, invalid credentials, policy/safety block) or when all candidate providers in the chain have been exhausted.
- **Error Preservation**: The original underlying exception is preserved and re-raised (`TaskExecutionError` or `ResearchExecutionError`).
- **Agent State**: Emits an `AgentEvent` recording the failure state, then returns the agent to `idle` so it can accept subsequent tasks.

---

## Semantic Failure Taxonomy

Runtime failures are classified into four semantic categories:

| Category | Typical Conditions | Traversal Action | Telemetry Status |
| :--- | :--- | :--- | :--- |
| **Eligible for Fallback** (`FallbackEligibleInferenceError`) | HTTP 429 rate limit, HTTP 500-599 server error, connection drop, network timeout, malformed provider payload. | Records sanitized `ProviderFailureSummary`. If candidates remain, resolves next provider, emits `FallbackTransition`, and retries. | `fallback_eligible_failure` |
| **Not Eligible for Fallback** (`NonFallbackInferenceError`) | HTTP 400 bad request, HTTP 401/403 authentication/authorization error, HTTP 404 endpoint not found, content safety block. | Traversal stops immediately. Falling back would be invalid or unsafe. Preserves and re-raises original error. | `non_fallback_failure` |
| **Unexpected Failure** (`Exception`) | Unhandled runtime crash, process error, memory error, or unmapped exception. | Traversal stops immediately. Preserves and re-raises original exception with `failure_reason="unexpected_error"`. | `unexpected_failure` |
| **Provider Exhaustion** (`AllInferenceProvidersFailedError`) | All ranked candidate providers failed with fallback-eligible errors. | Raises `AllInferenceProvidersFailedError(failures=(...))` containing the immutable tuple of sanitized summaries. **No duplicate terminal telemetry is emitted.** | *(All attempted providers already emitted telemetry)* |

---

## Telemetry & Observability (Parallel Secondary Streams)

Operational telemetry is a secondary side channel watching execution. Attempt telemetry and fallback transitions are **parallel tributaries**, not a sequential pipeline:

```
[ Step 4: Provider Attempt ]                 [ Step 8: Next Provider ]
             │                                            │
             │ (Every invocation)                         │ (On provider switch)
             ▼                                            ▼
   [ Attempt Telemetry ]                        [ Fallback Transition ]
             │                                            │
             └─────────────────────┬──────────────────────┘
                                   ▼
                       [ agent.events Stream ]
                                   │
                                   ▼
                       [ Prototype Console ]
```

### 1. Attempt Telemetry (`InferenceExecutionTelemetry`)
- **Principle**: *"One actual provider invocation = one attempt telemetry record."*
- **Origin**: Connects directly from **Step 4 (Invoke Provider)** on every attempt.
- **Fields Captured**:
  - Authoritative provider name;
  - Zero-based attempt index (`0, 1, ...`);
  - High-precision execution latency in milliseconds;
  - Execution status (`succeeded`, `fallback_eligible_failure`, `non_fallback_failure`, `unexpected_failure`);
  - Model name and finish reason (when genuinely returned);
  - Token usage: `input_tokens`, `output_tokens`, `total_tokens` (when reported by the upstream provider; unknown fields remain `None`, never estimated or fabricated);
  - Sanitized failure reason and HTTP status code on failures.

### 2. Fallback Transition Events (`FallbackTransition`)
- **Origin**: Connects directly from **Step 8 (Next Provider)**.
- **Semantics**: Emitted strictly when execution actually advances from a failed provider to a target provider.
- **Sequencing**: Published strictly *after* the next candidate provider has been successfully resolved.
- **Fields Captured**: `failed_provider`, `target_provider`, sanitized `reason`, and `attempt_index`.

### 3. Operational Stream & Prototype Console
- **Asynchronous Transport**: Both tributaries merge into standard `AgentEvent` envelopes and publish over Redis to `agent.events`. Transport failures are caught and logged; they never crash or abort ongoing inference fallback.
- **Bounded Retention**: `agent.events` acts as a sliding-window operational audit stream for live observability, not a permanent analytics warehouse.
- **Prototype Console Visibility**:
  - **Live Activity Timeline**: Visual badges indicate `failed_provider → target_provider` switches with sanitized reasons.
  - **Telemetry Card**: Displays execution latency (ms), provider name, model name, and attempt number.
  - **Grounded Results**: Formatted markdown with numbered findings and verified clickable citation links (`rel="noopener noreferrer"`).
  - **Credential Security**: Tokens, API keys, private endpoints, and raw tracebacks are strictly quarantined on the server side and never sent to the browser.
  - **Execution Semantics**: Prototype Console observes the outcome; it does not cause or gate task completion.

---

## Latency Measurement Boundary

> **Latency measures raw provider invocation time only.**

The measurement boundary wraps strictly around the provider network call:

```python
t_start = self._clock()  # time.perf_counter_ns
try:
    response = await current_provider.generate(prompt)
finally:
    t_end = self._clock()
latency_ms = max(0.0, (t_end - t_start) / 1_000_000.0)
```

### What Latency INCLUDES
- Network transit time to the upstream provider endpoint;
- Remote model inference and response token generation;
- Response payload reception and deserialization.

### What Latency STRICTLY EXCLUDES
- Service startup validation and provider instantiation;
- Concurrent per-task health probes (`check_health()`);
- Policy selection and candidate ranking;
- Provider instance resolution (`resolver(candidate.provider_name)`);
- `FallbackTransition` event construction and bus dispatch;
- `InferenceExecutionTelemetry` construction and Redis stream publication;
- Result store database operations or file writes.

---

## Architectural Truth Invariants

Tiny Agent Lab enforces four core truthfulness rules:

1. **`Configured ≠ Healthy`**: Static presence in `INFERENCE_PROVIDER` proves only that credentials or configurations were syntactically present at startup. It does not indicate that the remote endpoint is reachable or healthy at execution time.
2. **`Selected ≠ Successful`**: Being ranked as Candidate #1 by policy does not guarantee that inference will succeed. Live network errors, upstream capacity drops, and rate limits can occur immediately after selection.
3. **`Unknown ≠ Healthy`**: A provider whose health probe times out, throws an exception, or fails to connect produces no health snapshot. Missing health is treated as unknown and excluded from selection. **The system never fabricates or assumes a healthy state.**
4. **`1 Provider Invocation = 1 Telemetry Record`**: Telemetry is emitted on every actual attempt. If a provider is not invoked, zero telemetry is emitted for it. When candidate exhaustion occurs, every attempted provider already has its own attempt record; no synthetic or duplicate terminal attempt telemetry is created.
5. **No Fabricated State**: Token counts, health status, and failure summaries are never invented or guessed. Unknown values remain explicitly unknown.
