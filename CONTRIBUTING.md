# Contributing to Autonomous Agent Lab

Thank you for your interest in contributing! Autonomous Agent Lab is an open-source educational project under [Tiny Robot Lab](https://tinyrobotlab.com). Tiny Robot Lab itself is currently operated as a non-commercial learning project. The repository code is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0-only), and the license text governs permitted use. Our goal is to explore, build, and observe autonomous agent patterns, distributed messaging, and Clean Architecture from first principles.

## Educational Mission

We prioritize:
- **Learning by building:** Implementing core concepts directly rather than wrapping monolithic black-box frameworks.
- **Architectural clarity:** Maintaining clear boundaries so components can be tested and replaced independently.
- **Honest engineering:** Documenting limitations, edge cases, and design tradeoffs openly.

## Contributor Workflow

1. **Discuss First:** For non-trivial features or architectural changes, please open an issue or discussion first so we can align on design goals and scope.
2. **Branch from `main`:** Create a focused, descriptive branch:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-fix-name
   ```
3. **Scope Discipline:** Keep changes small and focused on a single responsibility. Avoid mixing unrelated refactorings, formatting changes, or dependency upgrades with functional changes.
4. **Adhere to Clean Architecture:**
   - **Domain Layer:** Pure entities and invariants. No imports from external frameworks, FastAPI, Redis, or cloud providers.
   - **Application Layer:** Use cases and abstract ports (interfaces). Agnostic of concrete delivery mechanisms.
   - **Infrastructure & Presentation Layers:** Concrete adapters for external APIs, Redis queues, and HTTP routers.
5. **Deterministic Testing:** All tests must run quickly and deterministically using in-memory fakes without requiring live cloud APIs, paid tokens, or running Docker containers.
6. **Local Verification:** Before submitting a pull request, run the complete verification suite locally:
   ```bash
   .venv/bin/python -m pytest -q
   .venv/bin/python -m ruff check .
   git diff --check
   ```
   Deterministic tests isolate provider environment variables. Tests marked
   `external_provider` require the deliberate `--run-external-providers` option;
   see the [testing strategy](docs/testing-strategy.md) before running them.
7. **Submit a Pull Request:** Open a PR against the `main` branch with a clear description of the change, architectural rationale, and verification steps.

## Credential & Secret Discipline

- **Never commit credentials:** Do not commit `.env` files, API keys (Tavily, Cloudflare), tunnel tokens, private keys, or passwords.
- **Use Mock Values in Tests:** Use explicit mock strings (e.g. `"tvly-test-mock"`) in automated test fixtures.
- All real provider credentials must remain strictly inside ignored local environment files or container environment overrides.

## Documentation & Tradeoffs

When submitting changes:
- Explain *why* a particular architectural choice was made.
- Document any known limitations, performance tradeoffs, or heuristic boundaries honestly.

## Licensing Terms

All contributions to this repository are accepted under the terms of the **GNU Affero General Public License v3.0** (SPDX: `AGPL-3.0-only`).

By submitting a pull request, you agree that your contributions will be licensed under the AGPL-3.0-only license without additional restrictions. The license text governs permitted use. No Contributor License Agreement (CLA) or copyright assignment is required.

