# Architecture Decision Records

## Purpose

ADRs record why Tiny Agent Lab makes consequential, long-lived architectural
decisions and the alternatives and tradeoffs considered. They complement, rather
than replace, [the architecture overview](../architecture.md).

## When an ADR is required

Create and accept an ADR before implementation when a change establishes or
materially changes a consequential architectural decision, including:

- Clean Architecture boundaries or dependency direction;
- shared architectural abstractions or contracts;
- cross-service communication patterns;
- shared infrastructure services;
- protocols or persistent integration boundaries;
- security or authorization architecture; or
- decisions with meaningful long-term effects or costly reversal.

If it is uncertain whether the threshold is crossed, raise the question during
planning and decide before implementation. Do not make every technical decision
an ADR.

An ADR is not normally required for ordinary bug fixes, tests,
architecture-preserving refactoring, documentation corrections, routine
dependency maintenance, or normal implementation within an already accepted
architectural boundary.

## Workflow

1. During planning, assess the change against the threshold above.
2. If required, create a sequentially numbered ADR with status **Proposed**.
3. Review and accept the decision before implementing it. A Proposed decision
   must not be implemented as accepted architecture.
4. Link the ADR to its related GitHub issue. Link the implementation PR to both
   the issue and the relevant ADR.
5. Update [the architecture overview](../architecture.md) when needed to keep it
   an accurate description of the currently accepted architecture.

ADRs are not required for every issue or PR. An Accepted ADR does not need a
later edit just to add an implementation PR URL.

## Statuses

- **Proposed** — the decision is under review and must not be implemented as
  accepted architecture.
- **Accepted** — the decision is approved and implementation should follow it.
- **Superseded** — the decision has been replaced by a newer ADR.

## Numbering and filenames

Use sequential four-digit numbers and a short kebab-case title:

```text
0001-establish-prospective-adr-practice.md
0002-short-decision-title.md
```

Choose the next number by checking the existing ADR files. Use
[the template](template.md) for new records.

## Supersession and history

Accepted ADRs are historical decision records. Correct trivial mistakes only.
When an accepted decision changes materially, create a new ADR that links to
the decision it supersedes, then mark the earlier ADR **Superseded**. Do not
rewrite the original rationale to make history appear different.

Architecture implemented before this process became operational is not to be
reconstructed as retrospective ADRs. The initial baseline and timing are
described truthfully in
[ADR 0001](0001-establish-prospective-adr-practice.md).
