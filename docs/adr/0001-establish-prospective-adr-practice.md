# ADR 0001: Establish a Prospective ADR Practice

Status: Accepted
Date accepted: 2026-11-11
Related issue: N/A (repository-maintainer decision)
Supersedes:

## Context

The repository's [architecture overview](../architecture.md) calls for
significant decisions to have an ADR, but a working ADR process had not yet
been established. The architecture implemented through GitHub Issue #25
predates this operational process. This record establishes the practice
prospectively; it does not claim that earlier decisions followed it.

The current architecture at adoption is described in
[docs/architecture.md](../architecture.md). No retrospective ADRs will be
created for Issues #20 through #25, and no historical deliberations, approvals,
or decision dates will be inferred.

## Decision

Adopt the lightweight prospective ADR process in
[docs/adr/README.md](README.md). Future consequential architectural decisions
that cross its threshold require review and acceptance before implementation.
The architecture overview remains the readable description of the current
architecture; ADRs record the rationale, alternatives, and tradeoffs for
significant decisions.

## Consequences and tradeoffs

Future architectural changes have a concise, reviewable record before
implementation, while routine changes within accepted boundaries avoid
unnecessary governance. Existing architecture through Issue #25 remains the
baseline without invented decision history.

## Alternatives considered

- **No formal ADR practice:** rejected because the architecture overview already
  establishes a requirement, and a defined process is needed to make it
  operational.
- **An ADR for every technical decision:** rejected as disproportionate and
  likely to obscure consequential decisions with routine implementation detail.
- **Retrospectively reconstructing ADRs for prior issues:** rejected because
  that would risk inventing chronology, alternatives, or approval history that
  was not recorded.
