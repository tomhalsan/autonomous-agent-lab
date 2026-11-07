# Routing Evaluation, Deterministic Assessment & Baseline Benchmarks

This document describes the routing evaluation dataset, benchmark runner, empirical baseline, and the Phase B deterministic assessment architecture established for the capability routing subsystem in `autonomous-agent-lab`.

---

## 1. Purpose

The capability router directs incoming task prompts to the appropriate specialized agent (such as `AgentCapability.GENERAL` or `AgentCapability.RESEARCH`). 

As the platform evaluates potential improvements to capability routing (GitHub issue #91), an empirical baseline and typed assessment architecture are required to:
- Objectively quantify routing accuracy, precision, recall, and false-positive rates against ground truth.
- Evolve routing from returning a bare capability enum into producing a transparent [`RoutingAssessment`](file:///Users/tomhalsan/dev/autonomous-agent-lab/services/orchestrator/src/orchestrator/domain/assessment.py#L76-L108).
- Distinguish between **CLEAR** and **AMBIGUOUS** deterministic signals without relying on uncalibrated numeric confidence scores.
- Establish an architectural seam for a future `RoutingAdvisor` that is consulted only when deterministic evidence is ambiguous.
- Preserve backward-compatible `route(prompt) -> AgentCapability` execution across the system.

---

## 2. Dataset Architecture: Development vs. Development-Adjacent Validation

Routing evaluation is separated into two version-controlled datasets to distinguish development iteration from regression and adversarial coverage:

1. **Development Dataset** ([`evaluation/routing/dataset.json`](file:///Users/tomhalsan/dev/autonomous-agent-lab/evaluation/routing/dataset.json)): 44 curated test cases across 9 categories used to define and evaluate initial baseline signals. A 100% score on this dataset indicates that all known development cases pass, not universal production accuracy.
2. **Validation Dataset** ([`evaluation/routing/validation_dataset.json`](file:///Users/tomhalsan/dev/autonomous-agent-lab/evaluation/routing/validation_dataset.json)): 34 diverse test cases and adversarial prompts created during the same implementation cycle. This dataset provides regression and adversarial coverage for weak-verb disambiguation, diagnostic artifacts, and edge cases.

> [!IMPORTANT]
> Development-set and validation-set metrics are always reported separately.
> - The 44-case dataset is the primary development dataset.
> - The 34-case validation dataset was created during the same implementation cycle; its results provide regression and adversarial coverage rather than an unbiased production-accuracy estimate.
> - An unbiased production-accuracy estimate requires a future independent lockbox dataset created independently after the implementation is frozen.

### Schema

```json
{
  "id": "eval-001",
  "prompt": "What is 15% of 850?",
  "expected_capability": "general",
  "clarity": "clear",
  "rationale": "Direct arithmetic computation that does not require external evidence or live web retrieval.",
  "category": "clear_general",
  "tags": ["math", "calculation", "deterministic"]
}
```

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique identifier (e.g., `eval-001` or `val-001`). |
| `prompt` | `string` | The task instruction presented to the router (including short inline artifacts where applicable). |
| `expected_capability` | `"general" \| "research"` | Ground-truth capability required to complete the task. |
| `clarity` | `"clear" \| "ambiguous"` | Whether the intent is unequivocally clear or a borderline/mixed-intent case. |
| `rationale` | `string` | Architectural justification for the assigned ground truth label. |
| `category` | `string` | The functional evaluation category. |
| `tags` | `list[string]` | Keywords indexing prompt characteristics. |

### Categories Covered Across Datasets

- **Development Dataset (44 cases)**: `clear_general`, `clear_research`, `ambiguous_borderline`, `freshness_current_info`, `explicit_sources_evidence`, `explanatory_general`, `code_transformation`, `multi_clause_mixed`, `research_keywords_negative`.
- **Validation Dataset (34 cases)**: `weak_verb_supplied_artifact`, `freshness_current_info`, `supported_versions_compatibility`, `changing_prices_availability`, `external_standards_guidance`, `historical_established_advances`, `negative_current_program_state`, `negative_current_electrical`, `vague_insufficient_evidence`, `mixed_supplied_and_external`, `compound_retrieval_prerequisite`, `weak_verb_alone`, `package_management_installation`, `personal_activity_temporal`, `diagnostic_evidence_supplied_artifact`, `volatile_demographic_freshness`.

---

## 3. Ground Truth Labeling Principles

### Independence from Router Implementations
Ground truth labels represent **human architectural intent**, not the behavior of any existing router. An expected label must never be assigned simply because an existing regular expression or keyword list triggers on it.

### Evaluation Criteria
A prompt is labeled `expected_capability: "research"` if and only if:
- It requires up-to-date, live, or changing information past standard model training cutoffs (e.g., current pricing, recent releases, breaking news).
- It explicitly demands external source retrieval, URL citations, whitepapers, or peer-reviewed literature.
- Completing the task accurately depends on querying external search engines or verification tools.

A prompt is labeled `expected_capability: "general"` if:
- It asks for code synthesis, refactoring, formatting, debugging, or transformation of user-supplied code or inline artifacts.
- It asks for established concepts, definitions, mathematical calculations, or textbook algorithms.
- It uses words like "analyze", "compare", or "benchmark" to describe a local calculation or code analysis on user-supplied input.

### Criteria for Clear vs. Ambiguous
- **`clear`**: Intent is unambiguous. Any experienced software engineer would classify the required agent capability identically without needing further context.
- **`ambiguous`**: The prompt combines mixed signals, lacks temporal context, or could be answered adequately by either general synthesis or live retrieval depending on user expectations. For example, in `eval-013` (*"Tell me about quantum computing advancements"*), both a general historical overview and a current research interpretation are plausible; assigning `expected_capability: "research"` is a deliberate Tiny Robot policy choice for an ambiguous prompt to prioritize recent developments.

---

## 4. Phase B: Deterministic Routing Assessment Architecture

Phase B transitions the capability router from returning a flat `AgentCapability` enum to producing an immutable, inspectable [`RoutingAssessment`](file:///Users/tomhalsan/dev/autonomous-agent-lab/services/orchestrator/src/orchestrator/domain/assessment.py#L76-L108).

### Domain Model

```python
@dataclass(frozen=True)
class RoutingAssessment:
    provisional_capability: AgentCapability
    clarity: RoutingClarity
    signals: tuple[RoutingSignal, ...]
    reason: str
```

### Routing Authority Order

```mermaid
flowchart TD
    Start["Incoming Task Submission"] --> Override{"Explicit Capability<br/>Supplied by Caller?"}
    Override -- Yes --> DirectDispatch["Bypass Inference<br/>Use Explicit Capability<br/>(Record Provenance)"]
    Override -- No --> DetAssess["Deterministic Routing Assessment<br/>(DeterministicCapabilityRouter.assess)"]
    DetAssess --> ClarityCheck{"Clarity Assessment"}
    ClarityCheck -- CLEAR --> RouteProvisional["Dispatch with Provisional Capability"]
    ClarityCheck -- AMBIGUOUS --> AdvisorSeam["Future RoutingAdvisor<br/>(Phase C Seam)"]
    AdvisorSeam -- "Advisor Available" --> ConsultAdvisor["Consult Advisor"]
    AdvisorSeam -- "Advisor Absent / Fails" --> FallbackProvisional["Preserve Deterministic<br/>Provisional Capability"]
    ConsultAdvisor --> Dispatch["Dispatch Target Agent"]
    DirectDispatch --> CandidateDiscovery["Candidate Discovery &<br/>Availability Evaluation"]
    RouteProvisional --> CandidateDiscovery
    FallbackProvisional --> CandidateDiscovery
    CandidateDiscovery --> Dispatch
```

1. **Caller-Supplied Explicit Capability**: An explicit capability override is always authoritative. It bypasses deterministic inference and any future advisory model, while strictly preserving candidate discovery, availability checks, selection, and transport dispatch.
2. **Deterministic Routing**: The primary inference mechanism. Evaluates typed signals at microsecond speed.
3. **Future RoutingAdvisor Seam**: May be consulted **only** when the assessment clarity is `AMBIGUOUS`.
4. **Resilient Fallback**: If a future advisor is absent, times out, or fails, the router preserves the deterministic provisional capability rather than failing or unprincipled fallback.

---

## 5. Signal Taxonomy & Core Routing Policies

### Signal Strength Tiers

| Signal Type | Target Capability | Strength | Description & Trigger Conditions |
|---|---|---|---|
| `EXPLICIT_SOURCES_OR_CITATIONS` | `RESEARCH` | `STRONG` | Explicit requests for literature, peer-reviewed studies, whitepapers, or source citations. |
| `FRESHNESS_OR_CURRENT_INFO` | `RESEARCH` | `STRONG` | Reusable temporal qualifiers (`today's`, `currently supported`, `latest release`, `current guidance`). |
| `CHANGING_EXTERNAL_INFO` | `RESEARCH` | `STRONG` | High-frequency changing information (`pricing tiers`, `stock market trends`, `breaking changes`). |
| `OUTSIDE_AUTHORITY_OR_PROVIDER` | `RESEARCH` | `STRONG` | Inquiries targeting specific external conferences, standards bodies (NIST, ISO), or vulnerability registries (NVD, CVE). |
| `EXPLICIT_EXTERNAL_RETRIEVAL` | `RESEARCH` | `STRONG` | Explicit directions to query live external search engines. |
| `SUPPLIED_LOCAL_ARTIFACT` | `GENERAL` | `STRONG` | Actual presence of structural markers (fenced code, diffs, stack traces, INI configs, JSON payloads). |
| `LOCAL_CODE_OR_DEBUGGING` | `GENERAL` | `STRONG` | Local coding/debugging tasks (`write a python script`, `fix this bug`, `loop termination analysis`). |
| `TRANSFORMATION_OR_REWRITE` | `GENERAL` | `STRONG` | Code or text transformations (`refactor this function`, `convert correlated subquery to CTE`). |
| `ESTABLISHED_CONCEPTUAL_EXPLANATION` | `GENERAL` | `STRONG` | Established computing theory, protocol mechanics, basic math, or physical science explanation. |
| `ALL_DATA_SUPPLIED_INLINE` | `GENERAL` | `STRONG` | Tasks where all necessary input data is provided inline. |
| `WEAK_RESEARCH_VERB` | `RESEARCH` | `WEAK` | Contextual verbs (`analyze`, `assess`, `compare`, `evaluate`, `benchmark`, `survey`, `conduct`, `research`, `investigate`, `study`, `explore`). |
| `COMPOUND_MIXED_TASK` | `RESEARCH` | `STRONG` | Multi-clause prompts combining external retrieval and local code synthesis. |

### Core Deterministic Policies

#### 1. Weak-Verb Policy
Weak research-like verbs (`analyze`, `assess`, `benchmark`, `compare`, `evaluate`, `survey`, `conduct`, `research`, `investigate`, `study`, `explore`) are contextual indicators and **must never independently determine RESEARCH**:
- **Weak verb + strong General evidence** -> `GENERAL / CLEAR` (e.g. *"Research this function and explain why the loop never terminates"*).
- **Weak verb + strong Research evidence** -> follows strong Research evidence (e.g. *"Research the latest developments in quantum computing"* -> `RESEARCH / CLEAR`).
- **Weak verb alone** -> provisional `GENERAL`, with clarity marked `AMBIGUOUS` (e.g. *"Research consensus algorithms"* -> `GENERAL / AMBIGUOUS`).
- The word "research" alone is never proof of external retrieval.

#### 2. Generalized Freshness & Negative Guards
Freshness is recognized when temporal qualifiers (`today`, `today's`, `current`, `currently`, `latest`, `recent`, `up-to-date`, `as of`) attach to volatile external facts (supported versions, releases, pricing, vulnerabilities, office holders, product capabilities, market conditions, standards or guidance).

**Negative Guards**: The word "current" does not trigger research when it refers to local program state (`current_user`, `current_node`, `current index`) or electrical current (`current of 10A`, `direct current`, `voltage drop given a current`).

#### 3. Generalized Supplied-Artifact Detection
`SUPPLIED_LOCAL_ARTIFACT` requires actual structural markers (fenced code blocks, stack traces, unified diffs, structured JSON/YAML/XML data, INI/TOML configuration blocks, or code constructs). Prompts merely containing words like "this" or "below" without actual content do not trigger artifact signals.

#### 4. Dynamic Reason Generation
Reasons are generated dynamically from the exact list of detected signals. Every reason accurately reflects the attached signals, never claims absent signals, and remains fully deterministic.

---

## 6. Benchmark Runner & CLI Options

The benchmark runner ([`evaluation/routing/benchmark.py`](file:///Users/tomhalsan/dev/autonomous-agent-lab/evaluation/routing/benchmark.py)) decouples correctness analysis from isolated latency profiling.

### Command-Line Interface

```bash
# 1. Run Phase B benchmark on the development dataset (Default)
PYTHONPATH="services/orchestrator/src:services/worker/src:services/research_agent/src:packages/contracts/src:." \
python -m evaluation.routing.benchmark

# 2. Run Phase B benchmark on the development-adjacent validation dataset
PYTHONPATH="services/orchestrator/src:services/worker/src:services/research_agent/src:packages/contracts/src:." \
python -m evaluation.routing.benchmark --validation

# 3. Run both development and validation datasets sequentially
PYTHONPATH="services/orchestrator/src:services/worker/src:services/research_agent/src:packages/contracts/src:." \
python -m evaluation.routing.benchmark --all

# 4. Reproduce frozen historical Routing v2 baseline (63.64%)
PYTHONPATH="services/orchestrator/src:services/worker/src:services/research_agent/src:packages/contracts/src:." \
python -m evaluation.routing.benchmark --baseline-v2
```

---

## 7. Empirical Results & Performance Summary

### Comparative Performance Table

| Metric | Phase A: Routing v2 Baseline | Phase B: Development Dataset | Phase B: Validation Dataset |
|---|:---:|:---:|:---:|
| **Cases Evaluated** | 44 | 44 | 34 |
| **Provisional Capability Accuracy** | **63.64%** (28/44) | **100.00%** (44/44) | **100.00%** (34/34) |
| **Clarity Label Alignment** | 88.64% (39/44) | **100.00%** (44/44) | **100.00%** (34/34) |
| **Hypothetical Advisor Rate** | 0.00% (0/44) | **11.36%** (5/44) | **23.53%** (8/34) |
| **Capability Acc (Assessed CLEAR)** | 63.64% | **100.00%** (39/39) | **100.00%** (26/26) |
| **Capability Acc (Assessed AMBIGUOUS)** | 0.00% | **100.00%** (5/5) | **100.00%** (8/8) |
| **Research Precision / Recall / F1** | 0.56 / 0.56 / 0.56 | **1.00 / 1.00 / 1.00** | **1.00 / 1.00 / 1.00** |
| **General Precision / Recall / F1** | 0.69 / 0.69 / 0.69 | **1.00 / 1.00 / 1.00** | **1.00 / 1.00 / 1.00** |
| **Median Execution Latency (P50)** | **~50 µs** | **~110 µs** | **~100 µs** |

---

## 8. Generalization Lessons & Phase C Prerequisites

### Generalization Lessons from Phase B
Initial testing of Phase B against unseen prompts revealed that high development scores can mask narrow regex assumptions:
1. Overly specific literal patterns (e.g. naming specific APIs or conferences) fail when users query different vendors or events.
2. Treating weak verbs like "research", "compare", or "assess" as authoritative research triggers causes severe false positives on local code debugging and configuration comparisons.
3. Matching "current" without negative guards misclassifies local variables (`current_user`, `current_node`) and electrical measurements as freshness research.
4. Unqualified temporal phrases like "this week" or "this month" misclassify personal user statements ("I started learning Rust this week") unless gated against personal activity.
5. "Find evidence" applied to supplied diagnostic artifacts (such as heap dumps or stack traces) represents local inspection rather than academic/literature retrieval.

### Deterministic Limits & Readiness for Phase C
While the generalized deterministic rules achieve 100% on both the 44-case development set and the 34-case validation set, deterministic pattern matching has fundamental limits:
- Novel phrasings without explicit freshness keywords or structural cues may be classified as ambiguous.
- True natural language understanding is required to disambiguate subtle contextual trade-offs.

**Phase C (Ollama RoutingAdvisor)** should only be explored because:
1. Validation/adversarial coverage and ambiguity delegation behavior are now well-calibrated and validated.
2. The advisory delegation rate is bounded (11% on dev, 24% on validation), ensuring that an LLM advisor is invoked only when deterministic evidence is genuinely ambiguous.
3. 76% to 89% of requests continue to route deterministically at sub-150-microsecond speeds.

---

## 9. Non-Blocking Architectural Follow-ups

The following improvements are documented for execution prior to or alongside Phase C:
1. **Persist RoutingAssessment Diagnostics**: Extend `RoutingDecision` (or add a companion `RoutingAssessmentRecord`) to persist assessment clarity, typed signals, and signal-backed reasoning alongside the dispatch decision, exposing them to the observatory and debugging APIs.
2. **Decompose Freshness Regex**: Decompose the large monolithic `FRESHNESS_OR_CURRENT_INFO` pattern into a collection of smaller named semantic sub-patterns (e.g., `supported_versions`, `pricing_queries`, `service_status`, `demographic_counts`) compiled at router initialization.
3. **Invert Freshness Whitelisting**: Replace the current term-based whitelist in `_is_false_freshness` with positively matched context guards, ensuring that new freshness qualifiers are not inadvertently suppressed.
4. **Independent Lockbox Dataset**: Create a third, truly independent lockbox evaluation dataset authored by an external contributor or at a later date after deterministic rules are completely frozen, establishing an unbiased production-accuracy estimate.
