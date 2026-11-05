# Autonomous Agent Lab Roadmap

## Roadmap Principle

**Versions represent demonstrated platform capabilities, not the amount of code written.**

Autonomous Agent Lab follows a capability-oriented development roadmap.

A version milestone is reached when the associated capability works end-to-end, is tested, and can be demonstrated through the platform. Individual implementation details may change as the architecture evolves.

The roadmap is intentionally flexible. Capabilities may move between milestones when experimentation shows that a different sequence produces a stronger architecture.

---

# v0.1.0 — Foundation

**Status: Complete**

### Goal

Establish the basic autonomous-agent platform architecture and prove that independently running services can exchange tasks and operational state.

### Capabilities

- Clean Architecture foundation
- Containerized services using Docker Compose
- Orchestrator service
- General Worker service
- Redis messaging
- Task submission
- Basic task execution
- Agent operational events and state
- Initial browser dashboard
- Home server deployment foundation
- Secure tunnel deployment foundation

### Demonstrated Flow

User → Orchestrator → Redis → Worker → Completion

### Milestone Outcome

The platform can receive a task, route the work through its messaging infrastructure, execute it using a worker, and expose operational state.

---

# v0.2.0 — Specialized Autonomous Agent

**Status: Current**

### Goal

Demonstrate that the platform can automatically delegate useful work to an independently deployable specialist agent.

### Capabilities

- Independent Research Agent service
- Automatic capability classification
- GENERAL and RESEARCH task routing
- Capability-specific Redis queues
- Pluggable ResearchExecutor abstraction
- Grounded public-web research
- Tavily research adapter
- Pluggable InferenceProvider abstraction
- Cloudflare Workers AI synthesis
- Grounded evidence preservation
- Persistent task results
- Task result retrieval API
- Agent capability registration
- Observable provider metadata
- Agent operational event stream
- Prototype Console
- Human-readable research results with sources
- Cross-machine development workflow
- Home server deployment

### Demonstrated Flow

User
→ Orchestrator
→ Capability Routing
→ Research Agent
→ Grounded Retrieval
→ Inference / Synthesis
→ Persistent Result
→ Prototype Console

### Milestone Outcome

A user can submit a natural-language research goal without selecting an agent manually.

The platform identifies the required capability, delegates the work to the Research Agent, gathers public evidence, synthesizes the evidence through a replaceable inference provider, persists the result, and presents the completed answer through the browser console.

---

# v0.3.0 — Dynamic Orchestration

**Status: Planned**

### Goal

Move from configured agent routing toward runtime capability-driven orchestration.

### Target Capabilities

- Runtime capability discovery
- Orchestrator awareness of registered capabilities
- Agent availability awareness
- Capability-based agent selection
- Routing based on registered capabilities rather than assumed topology
- Improved task lifecycle state
- Routing decision observability
- Better agent failure handling
- Source-quality awareness for research
- Prototype Console visualization of routing decisions

### Target Flow

Task
→ Determine Required Capability
→ Query Capability Registry
→ Identify Available Agent
→ Delegate
→ Observe Execution

### Milestone Outcome

The orchestrator can discover which currently registered agent provides a requested capability and use that information when selecting where work should be executed.

---

# v0.4.0 — Resilient Multi-Provider Inference

**Status: Planned**

### Goal

Allow agents to use different inference resources without becoming dependent on a single model provider or execution environment.

### Target Capabilities

- Multiple InferenceProvider adapters
- Cloud inference
- Lightweight local inference experiments
- Higher-performance local inference
- Provider availability monitoring
- Provider-selection policy
- Graceful provider fallback
- Basic workload/resource awareness
- Inference telemetry
- Configurable provider priorities

### Target Architecture

InferenceProvider
├── Cloud Provider
├── Lightweight Local Provider
├── High-Performance Local Provider
└── Future Providers

### Milestone Outcome

The application layer requests inference as a capability while infrastructure determines where that inference should execute.

---

# v0.5.0 — Tool-Using Agents

**Status: Planned**

### Goal

Allow agents to use discoverable tools instead of being limited to integrations explicitly embedded into each agent implementation.

### Target Capabilities

- Tool abstraction
- Tool registry
- Tool capability metadata
- MCP client experimentation
- MCP tool discovery
- Multiple tools available to a specialist agent
- Tool-selection behavior
- Tool execution telemetry
- Tool permission boundaries
- Failure handling for unavailable tools

### Target Flow

Agent Goal
→ Determine Required Tool
→ Discover Available Tools
→ Select Tool
→ Execute
→ Observe Result
→ Continue Work

### Milestone Outcome

Agents can select and invoke tools through standardized capability boundaries rather than requiring each integration to be permanently embedded into an agent.

---

# v0.6.0 — Multi-Agent Collaboration

**Status: Planned**

### Goal

Allow multiple specialist agents to contribute independently to a larger goal.

### Target Capabilities

- Additional specialist agents
- Agent-to-agent task delegation
- Parent/child task relationships
- Subtask execution
- Specialist collaboration
- Parallel task execution where appropriate
- Result aggregation
- Cross-agent execution tracing
- Collaboration visibility in the Prototype Console

### Target Flow

Goal
→ Orchestrator
→ Decompose Work
→ Specialist Agents
→ Independent Results
→ Aggregate
→ Final Result

### Milestone Outcome

A task can require multiple specialist capabilities and the platform can coordinate those specialists into one observable execution.

---

# v0.7.0 — Planning and Multi-Step Work

**Status: Planned**

### Goal

Allow the platform to determine how a complex goal should be accomplished, not only which agent should receive it.

### Target Capabilities

- Planning abstraction
- Multi-step plans
- Task dependencies
- Sequential execution
- Conditional execution
- Evaluation between steps
- Retry behavior
- Replanning
- Human approval points
- Execution guardrails
- Plan visualization

### Target Flow

Goal
→ Create Plan
→ Execute Step
→ Evaluate
→ Continue / Retry / Replan
→ Complete Goal

### Milestone Outcome

The platform can execute structured, multi-step work while preserving visibility into why each action occurs.

---

# v0.8.0 — Memory and Knowledge

**Status: Planned**

### Goal

Allow agents to retain and retrieve useful context across tasks while maintaining clear lifecycle and provenance boundaries.

### Target Capabilities

- Short-term working memory
- Cross-task contextual memory
- Knowledge retrieval
- Long-term memory where justified
- Semantic retrieval where justified
- Memory provenance
- Memory lifecycle management
- Scope isolation
- Forget/delete semantics
- Context-window management

### Architectural Principle

Memory technology will be introduced only when demonstrated use cases require it.

A vector database or similar technology is not considered a prerequisite for autonomous-agent architecture.

### Milestone Outcome

Agents can safely reuse relevant previous knowledge without requiring users to repeat all context for every task.

---

# v0.9.0 — Hardened Generic Prototype

**Status: Planned**

### Goal

Stop expanding capabilities temporarily and harden the complete generic platform.

### Target Capabilities

- Feature stabilization
- Security review
- Secret-management improvements
- Failure recovery
- Retry policies
- Improved observability
- GitHub Actions CI
- Automated quality checks
- Deployment automation
- Configuration cleanup
- End-to-end regression testing
- Provider failure simulation
- Tool failure simulation
- Contributor documentation
- Architecture documentation
- Demo scenarios
- Prototype Console polish
- Installation and setup documentation

### Development Policy

**Feature freeze.**

The goal of v0.9.0 is not to introduce another major architectural capability. The goal is to make the capabilities already present reliable, understandable, repeatable, and demonstrable.

### Milestone Outcome

The platform is ready for broader demonstration, experimentation, and contribution without requiring intimate knowledge of its development history.

---

# v1.0.0 — Stable Generic Autonomous-Agent Platform

**Status: Future**

### Goal

Graduate the project from experimental prototype to a stable generic autonomous-agent platform.

### Graduation Criteria

The platform must be:

- Stable
- Reproducible
- Documented
- Testable
- Observable
- Secure enough for its intended demonstration environment
- Installable by someone other than the original developer
- Extensible through documented capability boundaries
- Demonstrable without developer-only command sequences
- Independent of any single inference provider, research provider, tool vendor, or specialist agent

### Target Architecture

User / Console
→ Orchestrator
→ Capability Registry
→ Planning / Routing
→ Specialist Agents
→ Tools
→ Inference Providers
→ Knowledge / Memory
→ Results / Observability

### Milestone Outcome

Autonomous Agent Lab provides a stable framework for demonstrating and experimenting with autonomous-agent patterns using interchangeable agents, tools, and inference providers.

---

# Versioning Policy

Autonomous Agent Lab uses capability-oriented semantic versioning.

## Minor Releases

A minor release represents a new demonstrated platform capability.

Examples:

- `0.2.0` — Specialized autonomous agent
- `0.3.0` — Dynamic orchestration
- `0.4.0` — Multi-provider inference

Minor versions do **not** increase simply because significant amounts of code were added.

## Patch Releases

Patch versions represent fixes or improvements that do not introduce a new architectural capability.

Examples:

- `0.2.1` — Fix synthesis truncation
- `0.2.2` — Fix Prototype Console formatting
- `0.2.3` — Improve provider error handling

## Major Release

`1.0.0` represents the point at which the generic autonomous-agent platform is considered stable enough to be installed, understood, extended, and demonstrated without relying on the project's development history.

---

# Current Position

**Current milestone: v0.2.0 — Specialized Autonomous Agent**

Completed:

Foundation
→ Specialized Research Agent
→ Grounded Retrieval
→ Pluggable Inference
→ Real AI Synthesis
→ Persistent Results
→ Observable Capabilities
→ Prototype Console

Next major target:

**v0.3.0 — Dynamic Orchestration**

The next phase will focus on teaching the orchestrator to use runtime capability information when deciding where work should be delegated.
