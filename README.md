# Autonomous Agent Lab

Autonomous Agent Lab is Lab #1 under the [Tiny Robot Lab](https://tinyrobotlab.com) learning project.

It is a non-commercial, open-source educational testbed for exploring how modern AI and autonomous-agent systems actually work by building them from scratch: asynchronous messaging queues, dynamic capability registration, heartbeat liveness policies, evidence-conditioned synthesis, and provider-neutral Clean Architecture boundaries.

- **Current Release:** `v0.4.0 — Resilient Multi-Provider Inference`
- **Live Experimental Lab:** [Tiny Agent Lab](https://agents.tinyrobotlab.com)
- **Umbrella Website:** [Tiny Robot Lab](https://tinyrobotlab.com)

## Architecture

- **Domain:** rules and data models with no framework dependencies.
- **Application:** use cases and ports (interfaces).
- **Infrastructure:** Redis adapters and external implementations.
- **Presentation:** HTTP API and browser dashboard.

Dependency direction is inward: presentation/infrastructure -> application -> domain.

## Milestones

- **v0.1.0 — Initial Worker Prototype**: The orchestrator accepts a task, Redis transports it, and a worker processes it while publishing state. A small dashboard shows agent state and recent events.
- **v0.2.0 — Multi-Agent Research Prototype**: Automatic capability-based task routing, independent Research Agent service, grounded Tavily research retrieval, pluggable Cloudflare inference synthesis, persistent task results, observable agent capabilities and providers, and the Prototype Console interface.
- **v0.3.0 — Dynamic Orchestration**: Runtime capability discovery, heartbeat liveness tracking, deterministic candidate selection, advertised delivery target queues, and comprehensive routing observability.
- **v0.4.0 — Resilient Multi-Provider Inference**: Provider-neutral inference availability and health observation, integration of Google Gemini as a second real inference provider alongside Cloudflare Workers AI, deterministic provider selection policy, graceful fallback traversal across configured provider priority chains, per-attempt operational inference telemetry, deterministic provider failure and fallback simulation test suites, and Prototype Console visibility into provider execution and fallback transitions without dependency on any single model provider.

## Start

1. Copy `.env.example` to `.env` if you want local overrides.
2. Run `docker compose up --build -d`.
3. Open `http://localhost:8080`.
4. Submit a task from the dashboard.
5. Inspect with `docker compose ps` and `docker compose logs -f`.

From another computer on your LAN, use `http://<windows-host-ip>:8080` after allowing TCP 8080 through the Windows firewall. Do not expose Redis directly.

## Prototype Console (v0.4.0)

The Prototype Console provides a unified, approachable browser interface (`GET /`) for demonstrating and interacting with the lab:

- **Task Submission & Observability**: Enter natural-language prompts with client-side validation and live progress feedback.
- **Automatic Capability Routing**: Agent selection is fully automatic—the platform analyzes prompt requirements on the backend and dispatches to either the General Worker or Research Agent without requiring manual agent selection.
- **Result Presentation & Inference Metadata**: View synthesized research output with formatted headings, numbered findings, preserved line breaks, and safe clickable links (`http`/`https` only, `rel="noopener noreferrer"`). When complete, displays successful inference execution metadata including provider name, model (when known), latency in milliseconds, and human-friendly attempt number. Includes quick "Copy Result" and "New Task" workflows.
- **Available Agents Registry & Provider Priority**: Inspect registered autonomous agents, their operational state (`idle`, `working`, `error`), active task assignments, separate retrieval provider configuration, and configured inference-provider priority chains (`1. provider1 → 2. provider2`), with truthful indication that provider health is evaluated per task rather than continuously persisted.
- **Activity Timeline & Fallback Transitions**: Monitor live operational transitions and task events emitted asynchronously over Redis, featuring visually highlighted entries and compact transition details (`failed provider → target provider`, reason, and attempt) when inference fallbacks occur.
- **Server-Side Credential Isolation**: All provider keys and tokens remain strictly server-side within container environments and are never transmitted to or stored within the browser console.

To access the console:
1. Start the stack: `docker compose up --build -d`
2. Open `http://localhost:8080` in any modern web browser.


## Quality checks

Use the repository-local Python environment for development and tests:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
docker compose config --quiet
```

External-provider tests are always deselected unless explicitly enabled. To run
the live Tavily check, use
`.venv/bin/python -m pytest --run-external-providers -m external_provider tests/test_tavily_mcp_live.py -s`.
This deliberately contacts the configured external provider.

For full details on our seven validation layers, container packaging tests, isolated Redis integration suite, and release-validation guidance, see the [Testing Strategy](docs/testing-strategy.md).

## Next increments

1. Add task-level retries and failure recovery.
2. Replace polling with server-sent events or WebSockets.
3. Add PostgreSQL for durable task history.
4. Add authentication before LAN-wide or remote access.
5. Add OpenTelemetry and metrics.
6. Add a container-management UI separately, with restricted Docker socket access.

## Scope & Limitations

This project is an experimental learning laboratory, not an enterprise product, marketing tool, or commercial SaaS. We build features to understand distributed agent patterns and tradeoffs:
- Evidence ranking heuristics filter noise but do not provide mathematical proof of factual truth.
- Cross-domain keyword overlap indicates topical relevance, not deep semantic corroboration.
- Retrieval quality is strictly bounded by what upstream search providers return.

## Security & CI

Public repository integrity and security rely on standard GitHub-native mechanisms:
- **Continuous Integration:** GitHub Actions runs automated unit tests, isolated Redis integration tests against an ephemeral service container, container packaging validation, and Ruff lint checks without external provider credentials. See the [Testing Strategy](docs/testing-strategy.md).
- **Dependabot:** Regularly monitors Python dependencies, GitHub Actions, and Docker base images for known vulnerabilities.
- **Security Policies:** See [SECURITY.md](SECURITY.md) for vulnerability reporting guidelines and credential hygiene policies.

## License

This project is open-source software licensed under the [GNU Affero General Public License v3.0](LICENSE) (SPDX: `AGPL-3.0-only`). See [LICENSE](LICENSE) for the full license text.
