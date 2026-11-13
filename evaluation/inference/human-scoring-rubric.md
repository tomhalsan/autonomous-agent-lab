# Blinded Inference Output Scoring Worksheet

Score each available answer from **1 (poor) to 5 (excellent)** using the
dimension anchors below. Judge only the supplied case, its fixed input/evidence,
and the answer. Do not infer which provider or model produced it. Leave a
dimension blank only when it is genuinely not applicable; note why.

Use the opaque `attempt_id` to identify a row. Do not add provider/model
identity to the worksheet. Keep short notes about specific claims or omissions,
not guesses about model identity.

## Shared anchors

- **1 — Fails:** materially wrong, unsupported, or does not address the task.
- **2 — Weak:** important errors or omissions; only a small part is useful.
- **3 — Adequate:** mostly useful, with a notable omission or minor issue.
- **4 — Strong:** correct and useful; only minor defects.
- **5 — Excellent:** fully meets the case requirements, clear and appropriately
  concise, with no material defect.

## General Worker dimensions

| Dimension | Apply the shared anchors to |
| --- | --- |
| `correctness` | Accuracy against the supplied code, facts, or stable technical topic; do not reward unsupported additions. |
| `instruction_following` | Compliance with requested scope, length, caveats, bullet count, and output format. |
| `concision_usefulness` | Directness, clarity, and whether the answer is no longer than useful. |
| `format_validity` | Readability and structure, especially where a deterministic checker does not fully cover the requested format. |

## Research synthesis dimensions

| Dimension | Apply the shared anchors to |
| --- | --- |
| `supported_claim_coverage` | Coverage of the important claims actually supported by the fixed excerpts. |
| `unsupported_claims` | Absence of unsupported additions; score 5 when none are present and lower as unsupported claims become more material/frequent. |
| `conflict_handling` | Fairly represents conflicting evidence without manufacturing consensus. |
| `uncertainty` | Clearly states limits when the evidence is insufficient or silent. |
| `claim_source_attribution` | Attaches each material claim to the correct source. |
| `citation_fidelity` | Uses only supplied sources and cites the claims those sources support. |
| `concision_usefulness` | Direct, readable synthesis without unnecessary elaboration or extra findings. |

## Score record

Record JSONL or tabular scores keyed by `attempt_id`, for example:

```json
{"attempt_id":"attempt-opaque-id","dimensions":{"correctness":4,"instruction_following":5},"notes":"One minor omission."}
```

The identifier is for later joining only. Identity masking is not cryptographic
blinding; remove identity from the review artifact before scoring and avoid
sharing the unblinded result file with reviewers.
