# Architecture decisions

## Boundaries

Each deployable service owns its presentation and infrastructure adapters. Shared code is limited to stable contracts. Agents must communicate over explicit messages, not through imports into another service.

## Current message flow

1. The UI calls the orchestrator API.
2. The application use case validates and enqueues a `TaskRequested` message.
3. Redis is an infrastructure adapter.
4. The worker consumes the message and publishes state/events.
5. The UI reads an operational projection through the orchestrator API.

## Rules

- Domain modules do not import FastAPI, Redis, or Docker libraries.
- Application code depends on protocols, not concrete adapters.
- Environment-specific configuration enters at composition roots.
- Containers are disposable; state belongs in an external store.
- Message contracts are versioned.
- Secrets never enter source control or image layers.
- Use one explicitly supported Python minor version across local development, tests, and service containers (Python 3.12).

