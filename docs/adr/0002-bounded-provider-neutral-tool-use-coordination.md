# ADR 0002: Define a Bounded, Provider-Neutral Tool-Use Coordination Boundary

Status: Accepted
Date accepted: 2026-11-11
Related issue: https://github.com/tomhalsan/autonomous-agent-lab/issues/26
Supersedes: N/A

## Context

Issue #25 provides dynamic runtime tool catalog discovery through
`ToolDiscoveryService`. Discovery answers which `ToolRegistration` values are
currently visible in a discovery scope. It deliberately does not select,
authorize, or execute tools.

The tool contracts already define `ToolDefinition`, `ToolRegistration`,
`ToolInvocationRequest`, `ToolInvocationResult`, and related error and schema
structures. The repository does not currently have a generic tool-selection
coordinator, and the existing `ResearchProcessor` and Worker flows do not
implement a generic tool-use coordination loop.

Issue #26 introduces the first generic transition from a discovered catalog
toward selecting a tool, proposing arguments, validating that proposal,
optionally invoking one tool, and continuing after its result. Issue #27
separately owns real permission and authorization enforcement.

## Decision

1. Establish a bounded, provider-neutral tool-use coordination boundary.

2. Tool discovery supplies the currently discoverable `ToolRegistration`
   catalog. Selection consumes that catalog through the existing discovery
   boundary; selection policy must not duplicate registry filtering or cache
   the registry.

3. A provider-neutral selector evaluates the task or goal together with the
   discovered catalog and returns an explicit selection decision. Selection
   semantics remain agent-neutral and must not depend on research-specific
   domain concepts.

4. The selection decision has two first-class outcomes:
   - no tool needed; or
   - proposed invocation.

   No tool needed is a normal decision, not an exception, failure, empty tool
   name, or malformed invocation.

5. A proposed invocation contains the selected tool identity and candidate
   argument values. This ADR does not prescribe additional
   implementation-specific fields.

6. Keep selection distinct from discovery, validation, authorization,
   execution, and continuation. Each remains a separate responsibility.

7. Validate proposed arguments against the project's existing bounded
   tool-schema profile before execution. Do not adopt a general-purpose JSON
   Schema framework merely for Issue #26.

8. Keep correction policy separate from the core selection decision contract.
   The initial implementation permits at most one correction attempt and must
   fail explicitly if the corrected proposal remains invalid. Correction
   policy must always be bounded and have explicit stopping behavior. Future
   bounded retry or correction policies may evolve without changing the core
   selection decision contract.

9. Keep coordination limits separate from the stable discovery, selection,
   validation, authorization, execution, and continuation boundaries. The
   initial Issue #26 implementation performs at most one tool invocation,
   followed by one continuation or synthesis step. This proves one
   understandable iteration of the tool-use lifecycle; it is not an
   unrestricted autonomous tool-using loop.

   Future bounded multi-step coordination may repeat the same lifecycle with
   explicit state, limits, and stopping conditions. This ADR does not design
   that future loop.

   The stable boundaries are:

   ```text
   discovery
       -> selection
       -> validation
       -> authorization
       -> execution
       -> continuation
   ```

   The initial coordination policy is at most one correction attempt, at most
   one invocation, and one bounded continuation path.

10. Preserve an explicit authorization seam between validated selection and
    execution. Discovery eligibility is not execution authorization.
    Selection does not constitute authorization. Registration does not
    constitute authorization. Tool risk and category metadata do not
    constitute authorization.

11. Issue #27 remains responsible for permission grants, authorization
    decisions, denial behavior, approval policy, auditability, and
    security-scope enforcement.

12. Until Issue #27 exists, any Issue #26 execution demonstration may operate
    only under an explicitly documented trusted-composition or
    pre-authorization assumption, using deliberately supplied prototype or
    test tools. The trusted-composition assumption is temporary and must not
    become the meaning of registration. It is not a permission system.

13. Selection contracts and policy remain provider-neutral. Do not couple
    selection to Gemini, Cloudflare, Ollama, Groq, Tavily, MCP, or another
    concrete provider or protocol. MCP-originated tools are presented to
    selection through the existing provider-neutral `Tool` and
    `ToolRegistration` contracts.

14. A first concrete integration may use the Research Agent if implementation
    inspection shows it is the smallest meaningful end-to-end demonstration.
    This does not assign generic tool coordination permanently to
    `ResearchProcessor` or make tool use a Research Agent responsibility.

15. The core selection contract is the architectural boundary. Consumers
    depend on that contract, not on details of any selector implementation.
    Deterministic-selection rules, keyword matching, scoring heuristics, and
    other implementation-specific mechanics remain behind the selector
    boundary. The core selection input/output model must support deterministic,
    model-backed, or hybrid selectors.

    Conceptually:

    ```text
    task/goal + discovered ToolRegistration catalog
                        |
                        v
                tool selector boundary
                 /                  \
          deterministic          model-backed
            selector               selector
                 \                  /
                        |
                        v
              ToolSelectionDecision
                 /               \
             no tool       proposed invocation
    ```

    Changing from a deterministic selector to a model-backed or hybrid
    selector should not require redesigning tool discovery, `ToolRegistration`,
    the core selection decision contract, argument validation, the
    authorization seam, execution, or continuation.

    Begin with a deterministic, transparent selector as the initial reference
    implementation. It establishes and tests the contracts, lifecycle,
    no-tool behavior, validation behavior, and continuation seam; it is not
    the architectural definition of tool selection. An LLM is not required
    merely because Issue #26 uses the word "reasoning."

This architecture deliberately provides insertion points for later evolution:

```text
tool discovery
    -> tool selection
    -> argument validation/correction
    -> authorization (Issue #27)
    -> execution
    -> continuation
```

## Consequences and tradeoffs

Benefits:

- Establishes an explicit provider-neutral selection contract and normal
  no-tool outcome.
- Makes validation and the bounded correction attempt testable boundaries.
- Provides a clear authorization insertion point for Issue #27 without
  treating selection or discovery as permission.
- Keeps generic selection semantics reusable across future agents.
- Uses a deterministic first implementation to make behavior observable and
  produce evidence before introducing model-backed selection.
- Allows deterministic, model-backed, or hybrid selection to evolve behind
  the same provider-neutral contract.

Costs and limitations:

- The initial deterministic selector will have limited natural-language
  generality; a future model-backed or hybrid selector may improve it behind
  the same architectural boundary.
- The initial one-invocation, one-continuation policy proves one bounded
  iteration, not a fully autonomous agent loop. Future work should primarily
  evolve selector adapters or policies and bounded coordination policy rather
  than replace the architecture, assuming evidence continues to support this
  decision; this does not preclude future architectural change.
- The temporary trusted-composition assumption is suitable only for the
  bounded Issue #26 demonstration and is not authorization.
- Later structured or model-backed selection may require additional adapter
  work.
- Richer multi-step reasoning remains future work.

## Alternatives considered

- **Put generic tool-selection logic directly inside `ResearchProcessor`.**
  Rejected as the generic architecture because it would couple shared
  selection semantics to a research-specific processor. A Research Agent
  integration may still serve as the first concrete demonstration.
- **Use `ToolInvocationResult` or existing synthesis output as the selection
  decision.** Rejected because neither provides a clean first-class
  representation of both no-tool and proposed-invocation decisions.
- **Require model-backed or LLM tool selection immediately.** Rejected for
  the initial implementation because deterministic selection can establish
  observable contracts and lifecycle semantics first. Delaying it does not
  prevent later adoption: deterministic, model-backed, or hybrid selection is
  intentionally replaceable behind the same provider-neutral selector
  boundary.
- **Combine selection, authorization, and execution into one coordinator
  decision.** Rejected because Issue #27 owns permission enforcement and
  selection must not silently become authorization.
- **Build an unrestricted autonomous multi-tool loop now.** Rejected as
  premature and difficult to reason about. Issue #26 should prove one bounded
  cycle first.
