# Tiny Agent Lab Inference Evaluation

## Purpose and boundary

This package is an offline-first evaluation harness for later comparing the
inference configurations used by Tiny Agent Lab's General Worker and Research
Agent. Its 12 fixed cases are a representative screening set for these
workloads, not an estimate of broad real-world accuracy.

The evaluator is outside production provider selection. It accepts an injected
invocation boundary; it does not construct adapters, read credentials, probe
provider health, retry, or fall back. Stage 0 uses only deterministic fake
invokers and makes zero live calls. No production model/profile decision is
made here.

## Frozen candidate matrix

These are experiment candidates, not approved profiles or recommendations.

| Workload | Provider | Model |
| --- | --- | --- |
| General | Cloudflare | `@cf/google/gemma-4-26b-a4b-it` |
| General | Cloudflare | `@cf/openai/gpt-oss-20b` |
| General | Gemini | `gemini-3.5-flash-lite` |
| Research | Cloudflare | `@cf/qwen/qwen3.8-27b` |
| Research | Cloudflare | `@cf/openai/gpt-oss-20b` |
| Research | Gemini | `gemini-3.5-flash-lite` |

## Frozen datasets

`general_dataset.json` contains six fixed cases: local debugging,
transformation/rewrite, supplied-content synthesis, concise technical
explanation, instruction following, and JSON-only structured output.

`research_dataset.json` contains six controlled evidence fixtures: supported
versus unsupported claims, conflicting findings, claim/source attribution,
insufficient evidence, longer multi-excerpt synthesis with a distractor, and a
fixed-count cited summary. Source URLs use the reserved `.example` domain;
they are identifiers for fixture evidence, not live sources. No Tavily or
dynamic retrieval is used.

Each dataset has a version and a SHA-256 over canonical JSON (UTF-8, sorted
keys, compact separators, no NaN). Every attempt stores that version and hash.
Change the version and the hash-locking test when intentionally changing a
fixture.

## Deterministic checks and human review

The deterministic checks cover observable properties such as non-empty output,
literal required elements, known prohibited phrases, bullet counts/lengths,
JSON parseability, exact JSON keys/types/selected values, and citation-ID
allowlists. A lexical match is not proof that a response is semantically
correct or supported. In particular, deterministic citation checks do not
prove that a cited source entails the adjacent claim. Known prohibited literals
do not establish the semantic absence of unsupported claims. A fixture may
separately prohibit a known distractor citation; an allowlist only establishes
that cited identifiers were supplied. It neither requires a citation nor
establishes relevance or entailment.

Use the frozen `human-scoring-rubric.md` for blinded quality review. Review
artifacts contain opaque attempt IDs, case/workload identifiers, outputs, and
contract-check results, but omit requested and observed provider/model
identity. This is identity masking, not cryptographic blinding. Join scores
back using the attempt ID after scoring. There is no LLM judge.

## Measurements and limitations

Latency is measured around one injected inference/synthesis call. The result
records provider/model requested, provider/model observed when returned,
adapter name, explicitly supplied sampling metadata, status, finish/truncation
metadata, and provider-reported token counts. Missing usage stays null; the
harness does not estimate token counts.

Any cost in a report is a list-rate estimate using separately supplied pricing
metadata and reported input/output tokens. It is not actual billed cost.
Only attempts with complete provider-reported input/output usage are included;
usage incurred by failed attempts may be omitted. Without both token counts and
pricing metadata, cost is unknown. Partial token coverage is labeled partial.
No current prices are embedded in the package.

Provider adapters can apply different system prompts, sampling defaults, and
response handling. A live protocol must record those adapter settings and keep
the case request/output contract constant wherever the existing boundaries
allow. Results compare configurations as exercised through those boundaries,
not isolated model weights.

## Metadata discovery boundary

Provider/model facts are captured in a typed, allowlisted snapshot, with unknown
facts left null. The snapshot records provider/model identity when returned,
retrieval time, metadata source/method, and only normalized capability/limit/
sampling/reasoning fields. It is provenance and call-construction evidence,
not a production model-profile system. Experiment policy remains separate and
records deliberate overrides such as output limits or temperature. Effective
invocation descriptors preserve both policy and the resulting intended values,
including whether each value is explicit, provider-default, or unset.

Stage 1A provides evaluation-only provider metadata resolvers. Cloudflare
resolution uses one exact account catalog search and one schema GET per unique
model. Gemini resolution uses the existing non-generating `models.get` path,
once per unique model. A campaign resolves each provider/model identity once
and reuses its snapshot across workloads; the six frozen candidates contain
four unique identities, so the planned metadata budget is seven GET requests
(six Cloudflare and one Gemini). Retrieval timestamps are recorded in snapshots.

Cloudflare normalization uses only returned catalog and schema fields.
`context_window`, request `max_tokens` bounds, and absent input/output limits
remain unmapped; a schema-declared `max_tokens` default is recorded only as
`default_output_tokens`, not as a runtime guarantee. Gemini maps only the
observed model name, generation method, token limits, temperature fields, and
thinking flag. Unreported facts remain null.

Metadata resolution happens before inference and is excluded from inference
latency. A resolution failure has a sanitized, typed boundary distinct from
inference failure and prevents a snapshot—and therefore a live invocation
descriptor—from being produced for that candidate. It does not create an
`EvaluationAttempt`. These resolvers are Stage 1A evaluation support, not a
production metadata service, cache, or model-profile system.

## Stage gates and call budgets

| Stage | Scope | Planned live inference calls |
| --- | --- | ---: |
| 0 | Offline datasets, checks, runner, reports, and fake-invoker tests | 0 |
| 1 | One frozen case per candidate/workload compatibility smoke | 6 |
| 2 | All 6 cases × 3 candidates × 2 workloads | 36 total |
| 3 | After Stage 2, retain at most 2 finalists per workload; optional 6 cases × 2 finalists × 2 additional repetitions × 2 workloads | 48 additional; 84 total maximum on this path |

If Stage 1 uses the same frozen cases and protocol and succeeds, Stage 2 needs
30 additional calls. These counts exclude health probes, retrieval, retries,
and fallback. Stage 0 executes none of these planned live calls. Stage 3 is
targeted and conditional, not an automatic 108-call repetition plan.

## Experiment protocol

Use a unique, safe `run_id` and repository commit for each evaluation campaign;
reuse them across stages contributing to the same result set. Use a new `run_id`
for an independent campaign. Execute only the selected workload/candidates/
cases. Stage 1 and Stage 2 use repetition 0. Stage 2 must schedule only pairs
without an existing result record; Stage 1 successes contribute to the Stage 2
comparison, while recorded failures remain failure evidence and are not retried.
Stage 3 additional runs use repetitions 1 and 2 within the same campaign.
Attempt IDs are derived from run ID, dataset name/version/hash, case, candidate,
and repetition, not loop order.

The runner preflights scheduled IDs against the target JSONL, rejects existing
IDs before invocation, and checks again before append. One record is expected
for each completed ordinary success/failure invocation. Cancellation or
process termination may leave a scheduled attempt without a result record.
Duplicate protection is bounded, single-process Stage 0 behavior; concurrent
processes writing the same JSONL file are outside its guarantees.

Failures are persisted without raw exception bodies, and the runner continues
to the next independently scheduled pair. It never substitutes another
candidate for a failed attempt.

Before any live run, freeze the selected Stage 1 cases and output contracts,
confirm candidate availability and adapter compatibility, record non-secret
sampling/output-limit settings, and review the expected call budget. Live
provider tests remain outside ordinary CI.
