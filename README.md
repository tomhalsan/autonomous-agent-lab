# Autonomous Agent Lab

A new, independent project for experimenting with containerized autonomous agents, messaging, state, and observability. It has no dependency on the music project.

## Architecture

- **Domain:** rules and data models with no framework dependencies.
- **Application:** use cases and ports (interfaces).
- **Infrastructure:** Redis adapters and external implementations.
- **Presentation:** HTTP API and browser dashboard.

Dependency direction is inward: presentation/infrastructure -> application -> domain.

## Milestones

- **v0.1.0 — Initial Worker Prototype**: The orchestrator accepts a task, Redis transports it, and a worker processes it while publishing state. A small dashboard shows agent state and recent events.
- **v0.2.0 — Multi-Agent Research Prototype**: Automatic capability-based task routing, independent Research Agent service, grounded Tavily research retrieval, pluggable Cloudflare inference synthesis, persistent task results, observable agent capabilities and providers, and the Prototype Console interface.

## Start

1. Copy `.env.example` to `.env` if you want local overrides.
2. Run `docker compose up --build -d`.
3. Open `http://localhost:8080`.
4. Submit a task from the dashboard.
5. Inspect with `docker compose ps` and `docker compose logs -f`.

From another computer on your LAN, use `http://<windows-host-ip>:8080` after allowing TCP 8080 through the Windows firewall. Do not expose Redis directly.

## Prototype Console (v0.2.0)

The Prototype Console provides a unified, approachable browser interface (`GET /`) for demonstrating and interacting with the lab:

- **Task Submission & Observability**: Enter natural-language prompts with client-side validation and live progress feedback.
- **Automatic Capability Routing**: Agent selection is fully automatic—the platform analyzes prompt requirements on the backend and dispatches to either the General Worker or Research Agent without requiring manual agent selection.
- **Result Presentation**: View synthesized research output with formatted headings, numbered findings, preserved line breaks, and safe clickable links (`http`/`https` only, `rel="noopener noreferrer"`). Includes quick "Copy Result" and "New Task" workflows.
- **Available Agents Registry**: Inspect registered autonomous agents, their operational state (`idle`, `working`, `error`), active task assignments, and configured providers (e.g., Tavily retrieval, Cloudflare inference).
- **Activity Timeline**: Monitor live operational transitions and task events emitted asynchronously over Redis.
- **Server-Side Credential Isolation**: All provider keys and tokens remain strictly server-side within container environments and are never transmitted to or stored within the browser console.

To access the console:
1. Start the stack: `docker compose up --build -d`
2. Open `http://localhost:8080` in any modern web browser.


## Quality checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check .
docker compose config
```

## Next increments

1. Add task correlation and retries.
2. Replace polling with server-sent events or WebSockets.
3. Add PostgreSQL for durable task history.
4. Add authentication before LAN-wide or remote access.
5. Add OpenTelemetry and metrics.
6. Add a container-management UI separately, with restricted Docker socket access.
