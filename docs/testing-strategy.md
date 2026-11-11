# Tiny Agent Lab Validation and Release-Testing Strategy

> **Authoritative Specification • Verification Layers, Empirical Lessons, and Deployment Guidance**
> *Baseline: Post-v0.4.0 Testing Architecture (Test Strategy Increments A, B, and C)*

---

## 1. Core Testing Philosophy

Tiny Agent Lab follows a foundational testing principle:

> **Validate every architectural and runtime claim at the cheapest boundary capable of disproving it, and never present simulated or modeled evidence as observed runtime evidence.**

Testing is not an exercise in maximizing test counts or achieving superficial coverage percentages. Automated tests, static linters, mocks, container checks, and live deployments each answer distinct questions:

- **Mocks and fakes** validate internal algorithms, state machines, and contracts against *our modeled assumptions*.
- **Static tools** validate stylistic, syntactic, and structural rules against *our source code*.
- **Container checks** validate file presence and dependency graphs against *our image filesystems*.
- **Integration tests** validate wire protocols and connection semantics against *external engine software*.
- **Live deployment validation** validates operational assumptions against *real network conditions, physical hardware, and live third-party services*.

No single layer replaces another. Mocks are not flawed because they fail to predict undocumented vendor behavior, and live deployments are not superior simply because they use real networks. Each boundary must be understood for what it **can prove** and what it **cannot prove**.

---

## 2. The Seven Validation Layers

```
Layer 7: Real Deployment (home-core / physical host, browser console, live providers)
   ▲
Layer 6: External Provider Contracts (manual / opt-in live checks: Cloudflare, Gemini, Tavily)
   ▲
Layer 5: Real Redis Integration (PR CI: ephemeral redis:7-alpine, wire protocol, DB 15)
   ▲
Layer 4: Compose Configuration (PR CI: docker compose config --quiet)
   ▲
Layer 3: Container Boundary (PR CI: build images, runtime import smoke checks)
   ▲
Layer 2: Unit / Contract / Policy (PR CI: host pytest -m "not redis", fast deterministic fakes)
   ▲
Layer 1: Static / Hygiene (PR CI: Ruff lint, git diff --check)
```

### Layer 1: Static / Hygiene
- **Mechanism**: `ruff check .`, `git diff --check`, `git diff --cached --check`.
- **Execution**: Runs in PR CI and pre-commit checks.
- **Purpose**: Fast, deterministic static analysis, code quality, syntax checking, and pre-commit whitespace hygiene.
- **What it proves**: Code adheres to project formatting, type-safety annotations, linting standards, and clean Git whitespace hygiene.
- **What it cannot prove**: Functional correctness, runtime importability, or dynamic system behavior.

### Layer 2: Unit / Contract / Policy
- **Mechanism**: Host-side `pytest -q -m "not redis"`, using mocks (`AsyncMock`), fakes (`InMemoryRedis`, `FakeMessageBus`), and deterministic fixtures.
- **Execution**: Runs in PR CI (~950+ tests in under 3 seconds).
- **Purpose**: Validate pure domain logic, contract serializations, candidate selection policies, routing classifiers, fallback traversal loops, telemetry schemas, input validations, and error handling.
- **What it proves**: State transitions, algorithms, and application policies operate correctly under defined, controlled assumptions.
- **What it cannot prove**: Container packaging completeness, TCP socket behavior, Redis wire protocol compatibility, remote vendor API realities, or physical network latency.

### Layer 3: Container Boundary
- **Mechanism**: Build service Dockerfiles (`services/orchestrator/Dockerfile`, `services/worker/Dockerfile`, `services/research_agent/Dockerfile`) and execute entrypoint runtime import smoke checks inside the built container images in PR CI:
  - `docker run --rm test-orchestrator python -c "import orchestrator.presentation.api"`
  - `docker run --rm test-worker python -c "import agent_policy, worker.main"`
  - `docker run --rm test-research-agent python -c "import agent_policy, research_agent.main"`
- **Execution**: Runs on every pull request to `main`.
- **Purpose**: Detect missing package copies, broken `PYTHONPATH` declarations, and container-specific dependency omissions before merge.
- **What it proves**: Required application modules and third-party dependencies exist on the container filesystem and import successfully without runtime `ModuleNotFoundError` crashes.
- **What it cannot prove**: Inter-container network communication, daemon liveness, or runtime Compose stack health.

### Layer 4: Compose Configuration
- **Mechanism**: `docker compose config --quiet`.
- **Execution**: Runs on every pull request to `main`.
- **Purpose**: Validate the Docker Compose configuration artifact (`compose.yaml`).
- **What it proves**: Compose YAML syntax is valid, service definitions are well-formed, environment variable substitutions parse correctly, and network/volume bindings conform to specification.
- **What it cannot prove**: That the multi-container application stack starts up cleanly, reaches a healthy state, or maintains inter-service communication.

### Layer 5: Real Redis Integration
- **Mechanism**: Ephemeral `redis:7-alpine` service container in PR CI running exactly four focused integration tests in [`tests/test_redis_integration.py`](../tests/test_redis_integration.py) via:
  ```bash
  REDIS_URL=redis://localhost:6379/15 pytest -q -m redis
  ```
- **Execution**: Runs in PR CI in a dedicated workflow step isolated from standard unit tests.
- **Safety & Isolation**:
  - Requires explicit `REDIS_URL`. If `REDIS_URL` is unset, tests safely skip.
  - Requires database 15 (`/15`). Rejects DB 0 or missing database indices before connection.
  - Flushes DB 15 before and after each test.
  - Generates unique queue and task identifiers.
  - Wraps potentially blocking queue consumption in short `asyncio.wait_for` timeouts.
  - If `REDIS_URL` is explicitly provided but unreachable, fails immediately rather than silently skipping.
- **What it proves**: Real TCP socket connectivity, connection pool lifecycle (`aclose`), RESP wire protocol serialization/deserialization, FIFO queue transport (`rpush`/`blpop`), string result persistence, and stream operations (`xadd`/`xrevrange`) against the actual Redis 7 engine.
- **What it cannot prove**: End-to-end multi-agent orchestration, browser console functionality, or external AI provider availability.

### Layer 6: External-Provider Contract Validation
- **Mechanism**: Targeted, opt-in contract checks against live remote vendor endpoints (Cloudflare Workers AI, Google Gemini, Tavily Search API).
- **Execution**: Manual or pre-release verification; **never** part of ordinary PR CI.
- **Design Rules**:
  - Keep real-provider tests minimal and focused on contract assertions (e.g. metadata lookups, model aliases, authentication).
  - Do not create expansive live test suites.
  - Avoid making public network connectivity, upstream vendor outages, rate limits, API quotas, secret credentials, or billing costs a gate for merging pull requests.
  - Do not claim scheduled CI automation exists if it has not been implemented.
- **What it proves**: Live remote endpoints authenticate, accept our request payloads, expose expected model aliases, and return parseable responses.
- **What it cannot prove**: Deterministic behavior under edge-case failures, which must be simulated in Layer 2.

### Layer 7: Real Deployment / Release Validation
- **Mechanism**: Live verification on `home-core` (bare-metal home server) running the full Docker Compose stack (`compose.yaml`) with real browser interaction via the Prototype Console and authentic provider keys.
- **Execution**: Milestone release candidates and post-release validation.
- **Purpose**: Exercise deployment workflows, physical host networking, reverse proxy tunnels, process supervisors, and human-visible UI flows that cannot be replicated inside CI runners.
- **What it proves**: The integrated platform functions end-to-end under real operating conditions for actual users.
- **What it cannot prove**: Cross-platform portability across untested hardware or broad permutations of environment variables.

---

## 3. Escaped-Defect Case Studies (v0.4.0 Experience)

The post-release deployment of `v0.4.0` to `home-core` surfaced three distinct classes of runtime defects that escaped earlier validation layers, plus an essential CI isolation lesson during test-strategy hardening:

### Case Study 1: Container Packaging Omission (Issue #153 / PR #154)
- **Symptom**: Runtime crash in Worker and Research Agent containers:
  ```
  ModuleNotFoundError: No module named 'agent_policy'
  ```
- **Root Cause**: Host-side pytest executed within a virtual environment where `packages/policy/src` was in `PYTHONPATH`. The Dockerfiles copied `contracts` and `infrastructure` but omitted `packages/policy/src`.
- **Lesson**: **Host tests validate host environments, not container images.** Packaging claims must be verified by building container images and inspecting runtime imports from within the built filesystems (Layer 3).

### Case Study 2: Undocumented Vendor Aliasing (Issue #155 / PR #156)
- **Symptom**: Cloudflare health probe classified configured model `@cf/meta/llama-3.1-8b-instruct` as `UNAVAILABLE`, preventing task execution despite direct inference working perfectly.
- **Root Cause**: Cloudflare catalog search returned only `-fp8` variants, and model schema lookup returned HTTP 404 (`Model schema not found`), yet direct generation succeeded. Mocks assumed that an executable model would always appear in catalog search and schema endpoints.
- **Lesson**: **Mocks validate our modeled assumptions, not remote vendor reality.** Occasional real-provider contract checks (Layer 6) are required to observe vendor metadata quirks, aliasing behavior, and schema omissions.

### Case Study 3: Unrealistic Network Latency Thresholds (Issue #157 / PR #158)
- **Symptom**: The coordinator health probe intermittently cancelled Cloudflare probes with:
  ```
  Health probe timed out for provider cloudflare (timeout=1.00s)
  ```
- **Root Cause**: Cloudflare's two-stage zero-generation probe required 0.620s–0.827s under live conditions. Normal network variance regularly pushed elapsed probe latency over the hardcoded 1.0-second timeout.
- **Lesson**: **Unit tests prove timeout mechanics, not realistic network thresholds.** Network-sensitive timeouts must be configurable and calibrated against empirical measurements of live multi-endpoint overhead under variance.

### Case Study 4: CI Integration Environment Contamination (Increment B Hardening)
- **Symptom**: Four existing unit tests verifying the application's default unconfigured Redis URL (`redis://localhost:6379/0`) failed when `REDIS_URL` was injected into CI.
- **Root Cause**: Injecting `REDIS_URL=redis://localhost:6379/15` across the entire `pytest -q` step contaminated application startup tests that intentionally verify default environment variable fallbacks.
- **Lesson**: **Integration test configuration must remain isolated from unrelated unit tests.** PR CI must separate unit execution (`pytest -q -m "not redis"`) from integration execution (`REDIS_URL=... pytest -q -m redis`).

---

## 4. Release-Validation Guidance for `home-core`

Release validation on `home-core` provides empirical proof of platform stability before or during a milestone release. It should be lightweight, repeatable, and transparent about what was and was not verified.

### Standard Release-Validation Checklist (11 Steps)

1. **Verify Source State**: Confirm the repository is clean and checked out to the intended release candidate or `main` commit.
2. **Pull Source**: Update repository files (`git pull`).
3. **Build Images**: Build application images normally:
   ```bash
   docker compose build
   ```
   *(Do NOT require `--no-cache` on every routine release validation; use normal layer caching unless packaging manifests changed).*
4. **Start Stack**: Launch the Compose stack in detached mode:
   ```bash
   docker compose up -d
   ```
5. **Verify Service Health**: Inspect running containers (`docker compose ps`) and verify Orchestrator, Worker, and Research Agent become healthy and register on Redis.
6. **Verify Version Presentation**: Open the Prototype Console in a browser (`GET /`) and verify the console header visibly displays the expected release version (e.g., `v0.4.0`).
7. **Execute GENERAL Task**: Submit a general reasoning or formatting task through the console. Verify automatic routing to `worker-1`, execution completion, and return to `idle`.
8. **Execute RESEARCH Task**: Submit an information-seeking research prompt. Verify automatic routing to `research-agent-1`, Tavily retrieval execution, synthesis completion, and return to `idle`.
9. **Verify Result & Telemetry Presentation**: Inspect the console output. Confirm formatted output, clickable source links for research results, persistent storage in Redis, and accurate attempt telemetry (provider name, model, attempt index, latency in milliseconds).
10. **Verify Provider Selection & Probing**: Confirm the configured provider is probed truthfully and selected according to configured priority.
11. **Record Unvalidated Boundaries**: Explicitly document any capability that was intentionally not exercised during the release run (e.g., live fallback or secondary providers).

### Multi-Provider Fallback Validation Guidance

- **Do NOT manufacture artificial passes**: Do not configure an invalid provider name (e.g. `invalid-provider`) to test fallback. Invalid provider names are rejected during initialization and do not simulate an operational provider failure.
- **Genuine Requirements**: Valid live cross-provider fallback requires **two genuinely configured, valid providers** (e.g., Cloudflare and Gemini).
- **Controlled Failure**: The primary provider must experience a genuine, fallback-eligible transient error (such as temporary rate limiting, simulated outage, or transient network rejection) while the secondary provider remains healthy and eligible.
- **Truthful Reporting**: Until an opt-in, controlled failure mechanism is established, **record live fallback as unvalidated in the release log** rather than improvising fragile configurations or implying that simulation tests represent live deployment proof. Live cross-provider fallback traversal was subsequently operationally validated on `home-core` (commit `8e9fe54`) using a controlled depleted-prepayment Gemini configuration falling back to Cloudflare Workers AI (`gemini,cloudflare`). See [Inference Provider Flow](architecture/inference-provider-flow.md#live-cross-provider-fallback-validation-home-core-post-release) for the full empirical event log and exact validation boundary.

---

## 5. Tool-Using Agents Testing Invariant (v0.5 Preview)

As Tiny Agent Lab expands toward tool-using specialist agents (v0.5), verification must enforce a strict epistemological boundary:

$$\text{LLM-generated text} \ne \text{Simulated tool output} \ne \text{Actual tool observation} \ne \text{Verified persisted state}$$

### Provenance Invariant
An autonomous agent system must preserve sufficient operational provenance to distinguish whether any piece of information originated from:
1. **Model Generation**: Speculative tokens emitted by an LLM without external grounding.
2. **Deterministic Simulation / Fixture**: Synthetic mock data returned by test doubles or fakes.
3. **Actual Tool Observation**: Empirical data returned by a real, executed tool or external sensor.
4. **Verified Persisted State**: Authoritative, committed platform records stored in durable persistence (e.g. Redis hashes, task results).

### Conceptual Lesson: Simulated System Health
If a model responds to a prompt with formatted CPU percentages, memory statistics, process IDs, or worker counts in an environment where no real system-monitoring tool was invoked, that response represents **pure text generation, not system observability**.

An agent platform must never present generated model speculation as observed runtime health. If a tool is not executed, the observation does not exist. Tests for tool-using agents must verify the invocation boundary and the fidelity of the tool output, not merely whether the agent produced believable prose.

---

## 6. Summary of Validation Responsibilities

| Validation Layer | Target Boundary | Primary Tools | Execution Cadence | Catches |
| :--- | :--- | :--- | :--- | :--- |
| **1. Static / Hygiene** | Source code | `ruff`, `git diff --check` | Pre-commit & PR CI | Syntax, styling, whitespace errors |
| **2. Unit / Policy** | Application logic | `pytest -m "not redis"` | Pre-commit & PR CI | Broken contracts, invalid state transitions, routing logic errors |
| **3. Container Boundary** | Image filesystem | `docker build`, import checks | PR CI | Missing packages (`agent_policy`), bad `PYTHONPATH` |
| **4. Compose Config** | Configuration artifact | `docker compose config` | PR CI | YAML syntax, invalid service attributes, bad variable expansions |
| **5. Redis Integration** | External engine wire protocol | `pytest -m redis` (DB 15) | PR CI | Serialization wire errors, stream syntax, queue protocol bugs |
| **6. Provider Contract** | Vendor API semantics | Opt-in live test scripts | Pre-release / manual | Vendor metadata quirks, alias 404s, schema discrepancies |
| **7. Real Deployment** | Full physical stack | `compose.yaml` on `home-core` | Release validation | Latency thresholds, tunnel routing, browser UI rendering |
