"""Loading, validating, and hashing the frozen inference evaluation datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from evaluation.inference.models import (
    EvaluationCase,
    EvaluationDataset,
    SourceExcerpt,
    Workload,
)

DATASET_DIR = Path(__file__).parent


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def dataset_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_dataset(name: Workload) -> tuple[dict[str, object], str]:
    path = DATASET_DIR / f"{name}_dataset.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return cast(dict[str, object], raw), dataset_hash(raw)


def load_dataset(name: Workload) -> EvaluationDataset:
    raw, sha256 = _read_dataset(name)
    if raw.get("name") != name:
        raise ValueError(f"Expected dataset name {name!r}, got {raw.get('name')!r}")
    version = raw.get("version")
    raw_cases = raw.get("cases")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Dataset version must be a non-empty string")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Dataset cases must be a non-empty list")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise TypeError("Each dataset case must be an object")
        case_id = raw_case.get("id")
        category = raw_case.get("category")
        instruction = raw_case.get("instruction")
        checks = raw_case.get("checks")
        rubric_items = raw_case.get("human_rubric_items")
        if not all(
            isinstance(value, str) and value.strip() for value in (case_id, category, instruction)
        ):
            raise ValueError("Each case requires non-empty id, category, and instruction")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen_ids.add(cast(str, case_id))
        if not isinstance(checks, list) or not isinstance(rubric_items, list):
            raise TypeError(f"Case {case_id!r} requires checks and human_rubric_items lists")
        if not all(isinstance(item, dict) for item in checks):
            raise ValueError(f"Case {case_id!r} checks must be objects")
        if not all(isinstance(item, str) and item.strip() for item in rubric_items):
            raise ValueError(f"Case {case_id!r} rubric items must be non-empty strings")

        input_value = raw_case.get("input")
        question_value = raw_case.get("question")
        sources_value = raw_case.get("sources", [])
        if name == "general":
            if not isinstance(input_value, str) or not input_value.strip():
                raise ValueError(f"General case {case_id!r} requires input")
            if question_value is not None or sources_value:
                raise ValueError(f"General case {case_id!r} cannot contain research evidence")
            question = None
            sources: tuple[SourceExcerpt, ...] = ()
        else:
            if not isinstance(question_value, str) or not question_value.strip():
                raise ValueError(f"Research case {case_id!r} requires a question")
            if not isinstance(sources_value, list) or not sources_value:
                raise ValueError(f"Research case {case_id!r} requires fixed sources")
            if not all(isinstance(source, dict) for source in sources_value):
                raise ValueError(f"Research case {case_id!r} sources must be objects")
            if any(
                not all(
                    isinstance(source.get(key), str) and source[key].strip()
                    for key in ("id", "url", "excerpt")
                )
                for source in sources_value
            ):
                raise ValueError(f"Research case {case_id!r} contains an incomplete source")
            sources = tuple(
                SourceExcerpt(
                    id=source["id"],
                    url=source["url"],
                    excerpt=source["excerpt"],
                )
                for source in sources_value
            )
            if len({source.id for source in sources}) != len(sources):
                raise ValueError(f"Research case {case_id!r} has duplicate source ids")
            question = question_value
            input_value = None

        cases.append(
            EvaluationCase(
                id=cast(str, case_id),
                workload=name,
                category=cast(str, category),
                instruction=cast(str, instruction),
                checks=tuple(cast(dict[str, object], item) for item in checks),
                human_rubric_items=tuple(cast(list[str], rubric_items)),
                input=cast(str | None, input_value),
                question=cast(str | None, question),
                sources=sources,
            )
        )

    return EvaluationDataset(name=name, version=version, cases=tuple(cases), sha256=sha256)
