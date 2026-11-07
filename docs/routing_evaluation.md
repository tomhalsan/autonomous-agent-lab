# Routing Evaluation & Benchmark Baseline

This document describes the routing evaluation dataset, benchmark runner, and empirical baseline established for the capability routing subsystem in `autonomous-agent-lab`.

---

## 1. Purpose

The capability router directs incoming task prompts to the appropriate specialized agent (such as `AgentCapability.GENERAL` or `AgentCapability.RESEARCH`). 

As the platform evaluates potential improvements to capability routing (GitHub issue #91), an empirical baseline is required to:
- Objectively quantify routing accuracy, precision, recall, and false-positive rates against ground truth.
- Establish empirical latency profiles using high-resolution monotonic timing.
- Document boundary failures where keyword matching misinterprets user intent.
- Ensure any future LLM classifier, advisory model, or hybrid routing architecture is judged against consistent, reproducible ground truth.

---

## 2. Dataset Schema & Structure

The evaluation dataset is stored in `evaluation/routing/dataset.json`. Each entry adheres to the following schema:

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

### Fields

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique identifier (e.g., `eval-001`). |
| `prompt` | `string` | The task instruction presented to the router (including short inline artifacts where applicable). |
| `expected_capability` | `"general" \| "research"` | Ground-truth capability required to complete the task. |
| `clarity` | `"clear" \| "ambiguous"` | Whether the intent is unequivocally clear or a borderline/mixed-intent case. |
| `rationale` | `string` | Architectural justification for the assigned ground truth label. |
| `category` | `string` | The functional evaluation category (see below). |
| `tags` | `list[string]` | Keywords indexing prompt characteristics. |

### Evaluation Categories

The dataset includes 44 curated test cases across 9 distinct categories:

1. **`clear_general`**: Fundamental tasks (arithmetic, language translation, textbook coding, static facts) requiring no external research.
2. **`clear_research`**: Explicit requests for current live data, API pricing, product releases, or market events.
3. **`ambiguous_borderline`**: Prompts where reasonable developers might debate scope (e.g., broad technology trade-offs or trends).
4. **`freshness_current_info`**: Prompts with explicit temporal constraints (`this week`, `this month`, `current status`, `most recent`).
5. **`explicit_sources_evidence`**: Prompts explicitly demanding citations, peer-reviewed literature, empirical whitepapers, or primary sources.
6. **`explanatory_general`**: Conceptual, architectural, or protocol explanations (OAuth PKCE, OS processes vs threads, TCP congestion, Raft election) that remain general knowledge.
7. **`code_transformation`**: Local code refactoring, SQL query rewriting, and bug diagnoses operating on user-supplied code snippets.
8. **`multi_clause_mixed`**: Multi-sentence or compound prompts mixing explanation, code generation, and research dependencies.
9. **`research_keywords_negative`**: Negative control prompts containing research-associated words (`benchmark`, `assess`, `survey`, `analyze`, `compare`) without actual external retrieval intent.

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

## 4. Benchmark Architecture & Execution

### Two-Pass Evaluation Design
The benchmark runner (`evaluation/routing/benchmark.py`) deliberately decouples correctness analysis from latency profiling:

1. **Correctness Pass**: Evaluates each dataset prompt exactly once, recording predictions, checking intent diagnostics via `detect_intent` (if supported), building the confusion matrix, and reporting classification errors.
2. **Latency Pass**: Repeatedly invokes only `router.route(prompt)`. It completely excludes dataset loading, report formatting, `detect_intent` queries, and console I/O, collecting high-volume timing samples over multiple passes across the prompt set.

### CLI Execution
Run the benchmark directly from the repository root using the explicit, reproducible module invocation:

```bash
# Display help and CLI options without running the benchmark
PYTHONPATH="services/orchestrator/src:services/worker/src:services/research_agent/src:packages/contracts/src:." \
python -m evaluation.routing.benchmark --help

# Run with defaults (20 measured passes = 880 latency samples, 10 warmup passes)
PYTHONPATH="services/orchestrator/src:services/worker/src:services/research_agent/src:packages/contracts/src:." \
python -m evaluation.routing.benchmark

# Run with custom dataset path and latency sample configuration
PYTHONPATH="services/orchestrator/src:services/worker/src:services/research_agent/src:packages/contracts/src:." \
python -m evaluation.routing.benchmark --dataset path/to/dataset.json --latency-iterations 50 --warmup 5
```

### CLI Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--help` | flag | - | Show help message and exit without running benchmark. |
| `--dataset` | `Path` | `dataset.json` | Path to custom routing dataset JSON file. |
| `--latency-iterations` | `int` | `20` | Number of repeated measurement passes over all prompts in the latency pass. |
| `--warmup` | `int` | `10` | Number of unmeasured warmup passes over all prompts before timing. |

### Test Suite Execution
Unit and integration tests for the dataset schema, metrics calculations, CLI behavior, and decoupled evaluation passes are located in `tests/test_routing_benchmark.py`:

```bash
pytest tests/test_routing_benchmark.py -v
```

---

## 5. Baseline Performance: Deterministic Routing v2

Running `DeterministicCapabilityRouter` against the 44-case dataset yields the following empirical baseline:

```text
==============================================================================
ROUTING BENCHMARK REPORT: DeterministicCapabilityRouter
==============================================================================

SUMMARY METRICS
------------------------------------------------------------------------------
Total Cases:          44
  Clear Cases:        39   (Accuracy:  66.67% | 26/39)
  Ambiguous Cases:     5   (Accuracy:  40.00% | 2/5)
Overall Accuracy:    63.64% (28/44)

CONFUSION MATRIX (Research = Positive, General = Negative)
------------------------------------------------------------------------------
True Research  (TP):   10    False Research (FP):    8
False General  (FN):    8    True General   (TN):   18

CLASS-LEVEL PERFORMANCE
------------------------------------------------------------------------------
Research Capability -> Precision:   0.56 | Recall:   0.56 | F1:   0.56
General Capability  -> Precision:   0.69 | Recall:   0.69 | F1:   0.69

ACCURACY BY CATEGORY
------------------------------------------------------------------------------
  ambiguous_borderline            40.00% (2/5)
  clear_general                  100.00% (5/5)
  clear_research                  80.00% (4/5)
  code_transformation             80.00% (4/5)
  explanatory_general            100.00% (5/5)
  explicit_sources_evidence      100.00% (4/4)
  freshness_current_info          40.00% (2/5)
  multi_clause_mixed              40.00% (2/5)
  research_keywords_negative       0.00% (0/5)

LATENCY PROFILE (microseconds - monotonic perf_counter_ns)
------------------------------------------------------------------------------
  Warmup Passes:         10
  Measured Iterations:   20 passes over prompt set
  Total Latency Samples: 880
  Mean:                  ~63 µs
  Median (P50):          ~49 µs
  P90:                  ~158 µs
  P95:                  ~218 µs
  P99:                  ~254 µs
  Min:                    ~1 µs
  Max:                  ~275 µs
```

> [!NOTE]
> **Local Development Observation**: The latency profile above reflects empirical observations recorded on local developer hardware (macOS Apple Silicon). These figures serve as an experimental baseline rather than universal performance guarantees.

### Key Observations & Vulnerabilities
1. **False-Positive Keyword Traps (`research_keywords_negative`: 0%, `code_transformation`: 80%)**: 
   Regex rules matching isolated words (`benchmark`, `assess`, `survey`, `analyze`, `compare`, `evaluating`) unconditionally route prompts to `research` even when the task is local code review, statistical calculation on inline data, or regex optimization (`eval-034`).
2. **Freshness Blindness (`freshness_current_info`: 40%)**:
   Queries like *"Who is the current Prime Minister of the United Kingdom?"* or *"What are the trending open-source AI projects on GitHub this month?"* lack specific trigger phrases like `recent developments` or `latest pricing` and fall through to `general`.
3. **Sub-millisecond Latency**:
   Deterministic regex evaluation operates in the low two-digit microsecond range (~50 µs median).

---

## 6. Phased Evolution for Routing v3 (Issue #91)

Issue #91 structures capability routing exploration into four deliberate phases:

- **Phase A**: Dataset and Routing v2 baseline (this phase)
- **Phase B**: Deterministic routing assessment and ambiguity contract
- **Phase C**: Optional Ollama RoutingAdvisor
- **Phase D**: Comparative evaluation and ADR

### The Runtime Ambiguity Gap
Notice that Routing v2 currently misclassifies several human-labeled clear cases (e.g. negative keyword cases like code review or traceback analysis, and freshness queries). Ambiguity cannot simply be copied from the dataset at runtime; rather, **Phase B** must define how deterministic signals produce a typed clear/ambiguous assessment in production before invoking any external advisory model.

### Latency Trade-offs in Later Phases
Ollama inference latency and throughput will be measured empirically on home-core in a later phase. However, because local LLM inference requires orders of magnitude more time than microsecond-level regex evaluation, Phase B and Phase C will evaluate whether a hybrid architecture (fast deterministic routing for clear cases, advisory fallback for ambiguous cases) is necessary to protect platform responsiveness.

---

## 7. Calibration Warning: LLM Confidence Scores

> [!WARNING]
> **Model-reported confidence is not a calibrated probability.**
>
> When designing Phase C (Ollama RoutingAdvisor):
> - Large language models prompting for numeric confidence scores (e.g. `{"confidence": 0.95}`) exhibit severe overconfidence and poor calibration.
> - A self-reported confidence of `0.90` does **not** mean the model is correct 90% of the time.
> - Routing thresholds (e.g., *"fallback to deterministic if confidence < 0.8"*) cannot rely on raw verbalized model floats without empirical reliability diagrams and temperature/Platt scaling.
