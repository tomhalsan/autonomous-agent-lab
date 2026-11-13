"""Opt-in runner for injected inference boundaries; it never creates providers."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from agent_contracts import InferenceError, NonFallbackInferenceError

from evaluation.inference.checks import evaluate_checks, validate_check_specs
from evaluation.inference.datasets import canonical_json
from evaluation.inference.models import (
    Candidate,
    EffectiveInvocationDescriptor,
    EvaluationAttempt,
    EvaluationCase,
    EvaluationDataset,
    ExperimentPolicy,
    ProviderOutput,
)


class EvaluationInvoker(Protocol):
    async def invoke(
        self, descriptor: EffectiveInvocationDescriptor, case: EvaluationCase
    ) -> ProviderOutput:
        """Invoke exactly the supplied effective descriptor through an injected boundary."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validated_timestamp(timestamp_factory: Callable[[], str]) -> str:
    value = timestamp_factory()
    if not isinstance(value, str):
        raise TypeError("timestamp_factory must return an ISO timestamp string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("timestamp_factory must return an ISO timestamp string") from error
    if parsed.tzinfo is None:
        raise ValueError("timestamp_factory must return a timezone-aware ISO timestamp")
    return value


def _failure_category(error: Exception) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, (TypeError, ValueError)):
        return "invalid_response"
    if not isinstance(error, InferenceError):
        return "execution_error"

    reason = error.reason
    status_code = error.status_code
    if type(status_code) is not int or not 100 <= status_code <= 599:
        status_code = None

    if status_code == 429:
        return "rate_limited"
    if status_code in (401, 403):
        return "authentication_or_authorization"
    if reason in {
        "blocked_safety",
        "blocked_recitation",
        "blocked_blocklist",
        "blocked_prohibited_content",
        "blocked_spii",
    } and isinstance(error, NonFallbackInferenceError):
        return "safety_block"
    if status_code is not None and 400 <= status_code <= 499:
        return "http_4xx"
    if status_code is not None and 500 <= status_code <= 599:
        return "http_5xx"
    if reason == "rate_limited":
        return "rate_limited"
    if reason == "timeout":
        return "timeout"
    if reason == "connection_error":
        return "connection_error"
    if reason == "malformed_response":
        return "invalid_response"
    return "execution_error"


def scheduled_attempt_id(
    *,
    run_id: str,
    dataset: EvaluationDataset,
    case_id: str,
    descriptor: EffectiveInvocationDescriptor,
    repetition: int,
) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", run_id):
        raise ValueError("run_id must be a 1-64 character safe identifier")
    if type(repetition) is not int or repetition < 0:
        raise ValueError("repetition must be a non-negative integer")
    identity = {
        "run_id": run_id,
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "dataset_sha256": dataset.sha256,
        "workload": descriptor.candidate.workload,
        "case_id": case_id,
        "candidate_id": descriptor.candidate.id,
        "repetition": repetition,
    }
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    return f"attempt-{digest[:32]}"


def _read_attempt_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    attempt_ids: set[str] = set()
    with path.open(encoding="utf-8") as result_file:
        for line_number, line in enumerate(result_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL record at line {line_number}") from error
            if not isinstance(record, dict):
                raise TypeError(f"JSONL record at line {line_number} must be an object")
            if record.get("schema_version") != 2:
                raise ValueError(f"Unsupported JSONL schema at line {line_number}")
            attempt_id = record.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                raise ValueError(f"Missing attempt_id at line {line_number}")
            if attempt_id in attempt_ids:
                raise ValueError(f"Duplicate attempt_id already persisted: {attempt_id}")
            attempt_ids.add(attempt_id)
    return attempt_ids


def _append_record(path: Path, attempt: EvaluationAttempt) -> None:
    if attempt.attempt_id in _read_attempt_ids(path):
        raise ValueError(f"Attempt already persisted: {attempt.attempt_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as result_file:
        result_file.write(canonical_json(attempt.to_dict()))
        result_file.write("\n")
        result_file.flush()


async def run_evaluation(
    *,
    dataset: EvaluationDataset,
    candidates: Sequence[Candidate],
    descriptors: Sequence[EffectiveInvocationDescriptor],
    experiment_policy: ExperimentPolicy,
    invoker: EvaluationInvoker,
    output_path: Path,
    run_id: str,
    repository_commit: str,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    timestamp_factory: Callable[[], str] = _timestamp,
) -> list[EvaluationAttempt]:
    """Run each requested case/candidate pair once, persist each result immediately."""
    if experiment_policy.dataset_name != dataset.name:
        raise ValueError("experiment policy dataset_name must match the dataset")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", run_id):
        raise ValueError("run_id must be a 1-64 character safe identifier")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", repository_commit):
        raise ValueError("repository_commit must be a 7-64 character hexadecimal commit")
    if not candidates or not descriptors:
        raise ValueError("at least one candidate and descriptor must be selected")
    if any(candidate.workload != dataset.name for candidate in candidates):
        raise ValueError("all candidates must match the selected dataset workload")
    if len({candidate.id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate ids must be unique within a run")
    descriptor_by_id = {item.candidate.id: item for item in descriptors}
    if len(descriptor_by_id) != len(descriptors):
        raise ValueError("descriptor candidate ids must be unique within a run")
    if set(descriptor_by_id) != {item.id for item in candidates}:
        raise ValueError("descriptors must match the selected candidate identities")
    if any(descriptor_by_id[item.id].candidate != item for item in candidates):
        raise ValueError("descriptor candidate identity does not match scheduled candidate")
    ordered_descriptors = tuple(descriptor_by_id[item.id] for item in candidates)

    if experiment_policy.case_ids is None:
        selected_cases = dataset.cases
    else:
        selected_cases = tuple(dataset.case(case_id) for case_id in experiment_policy.case_ids)
    if not selected_cases:
        raise ValueError("at least one case must be selected")
    for case in selected_cases:
        if case.workload != dataset.name:
            raise ValueError(f"Case {case.id!r} does not match dataset {dataset.name!r}")
        validate_check_specs(case)

    schedule = [
        (
            case,
            descriptor,
            scheduled_attempt_id(
                run_id=run_id,
                dataset=dataset,
                case_id=case.id,
                descriptor=descriptor,
                repetition=experiment_policy.repetition,
            ),
        )
        for case in selected_cases
        for descriptor in ordered_descriptors
    ]
    scheduled_ids = [item[2] for item in schedule]
    if len(set(scheduled_ids)) != len(scheduled_ids):
        raise ValueError("scheduled attempts contain duplicate identities")
    existing_ids = _read_attempt_ids(output_path)
    conflicts = existing_ids.intersection(scheduled_ids)
    if conflicts:
        raise ValueError(f"Attempt already persisted: {min(conflicts)}")

    records: list[EvaluationAttempt] = []
    for case, descriptor, attempt_id in schedule:
        if attempt_id in _read_attempt_ids(output_path):
            raise ValueError(f"Attempt already persisted: {attempt_id}")
        candidate = descriptor.candidate
        started_ns = clock_ns()
        provider_output: ProviderOutput | None = None
        failure_category: str | None = None
        try:
            raw_output = await invoker.invoke(descriptor, case)
            if not isinstance(raw_output, ProviderOutput):
                raise TypeError("Injected invoker returned an invalid output envelope")
            provider_output = raw_output
        except Exception as error:  # noqa: BLE001 - record a sanitized result for any adapter failure.
            failure_category = _failure_category(error)
        latency_ms = max(0, clock_ns() - started_ns) / 1_000_000

        observed_provider = (
            provider_output.provider.strip().lower()
            if provider_output is not None and provider_output.provider is not None
            else None
        )
        observed_model = provider_output.model if provider_output is not None else None
        if provider_output is not None and (
            (observed_provider is not None and observed_provider != candidate.provider)
            or (observed_model is not None and observed_model != candidate.model)
        ):
            failure_category = "identity_mismatch"

        checks = evaluate_checks(case, provider_output.text) if provider_output is not None else ()
        status = (
            "succeeded" if provider_output is not None and failure_category is None else "failed"
        )
        attempt = EvaluationAttempt(
            schema_version=2,
            run_id=run_id,
            attempt_id=attempt_id,
            repository_commit=repository_commit,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            dataset_sha256=dataset.sha256,
            case_id=case.id,
            repetition=experiment_policy.repetition,
            invocation=descriptor,
            timestamp=_validated_timestamp(timestamp_factory),
            status=status,
            latency_ms=latency_ms,
            observed_provider=observed_provider,
            observed_model=observed_model,
            input_tokens=provider_output.input_tokens if provider_output else None,
            output_tokens=provider_output.output_tokens if provider_output else None,
            total_tokens=provider_output.total_tokens if provider_output else None,
            finish_reason=provider_output.finish_reason if provider_output else None,
            truncated=provider_output.truncated if provider_output else None,
            failure_category=failure_category,
            output=provider_output.text if provider_output else None,
            checks=checks,
        )
        _append_record(output_path, attempt)
        records.append(attempt)
    return records
