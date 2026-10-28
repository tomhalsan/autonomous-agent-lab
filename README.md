# Autonomous Agent Lab

A new, independent project for experimenting with containerized autonomous agents, messaging, state, and observability. It has no dependency on the music project.

## Architecture

- **Domain:** rules and data models with no framework dependencies.
- **Application:** use cases and ports (interfaces).
- **Infrastructure:** Redis adapters and external implementations.
- **Presentation:** HTTP API and browser dashboard.

Dependency direction is inward: presentation/infrastructure -> application -> domain.

## First milestone

The orchestrator accepts a task, Redis transports it, and a worker processes it while publishing state. A small dashboard shows agent state and recent events.

## Start

1. Copy `.env.example` to `.env` if you want local overrides.
2. Run `docker compose up --build -d`.
3. Open `http://localhost:8080`.
4. Submit a task from the dashboard.
5. Inspect with `docker compose ps` and `docker compose logs -f`.

From another computer on your LAN, use `http://<windows-host-ip>:8080` after allowing TCP 8080 through the Windows firewall. Do not expose Redis directly.

## Quality checks

```bash
python -m pip install pytest ruff
python -m pytest
docker compose config
```

## Next increments

1. Add task correlation and retries.
2. Replace polling with server-sent events or WebSockets.
3. Add PostgreSQL for durable task history.
4. Add authentication before LAN-wide or remote access.
5. Add OpenTelemetry and metrics.
6. Add a container-management UI separately, with restricted Docker socket access.
