"""Pure preparation of the frozen Stage 1B compatibility-smoke schedule."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from evaluation.inference.candidates import CANDIDATES
from evaluation.inference.datasets import load_dataset
from evaluation.inference.models import (
    EffectiveInvocationDescriptor,
    EvaluationCase,
    EvaluationDataset,
    InvocationPolicy,
    ProviderModelMetadataSnapshot,
)
from evaluation.inference.runner import scheduled_attempt_id

MetadataKey = tuple[str, str]

_CASE_IDS = {
    "general": "general-debug-01",
    "research": "research-insufficient-04",
}
_METADATA_PROVENANCE = {
    "cloudflare": ("account_catalog", "account_catalog"),
    "gemini": ("provider_api", "model_get"),
}
STAGE1B_OUTPUT_LIMIT = 1024
STAGE1B_REPETITION = 0


@dataclass(frozen=True)
class PreparedStage1BAttempt:
    dataset: EvaluationDataset
    case: EvaluationCase
    descriptor: EffectiveInvocationDescriptor
    repetition: int
    run_id: str
    repository_commit: str
    attempt_id: str

    def to_sanitized_dict(self) -> dict[str, object]:
        metadata = self.descriptor.metadata
        candidate = self.descriptor.candidate
        return {
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "repository_commit": self.repository_commit,
            "workload": candidate.workload,
            "dataset_version": self.dataset.version,
            "dataset_sha256": self.dataset.sha256,
            "case_id": self.case.id,
            "candidate_id": candidate.id,
            "provider": candidate.provider,
            "model": candidate.model,
            "repetition": self.repetition,
            "metadata": {
                "source": metadata.metadata_source,
                "method": metadata.metadata_method,
                "retrieved_at": metadata.retrieved_at,
                "supports_text_generation": metadata.supports_text_generation,
                "context_limit_tokens": metadata.context_limit_tokens,
                "input_limit_tokens": metadata.input_limit_tokens,
                "max_output_tokens": metadata.max_output_tokens,
                "default_output_tokens": metadata.default_output_tokens,
                "temperature_supported": metadata.temperature_supported,
                "min_temperature": metadata.min_temperature,
                "default_temperature": metadata.default_temperature,
                "max_temperature": metadata.max_temperature,
                "reasoning_supported": metadata.reasoning_supported,
                "reasoning_default": metadata.reasoning_default,
            },
            "effective_output_limit_mode": self.descriptor.effective_output_limit_mode,
            "effective_output_limit": self.descriptor.effective_output_limit,
            "effective_temperature_mode": self.descriptor.effective_temperature_mode,
            "effective_temperature": self.descriptor.effective_temperature,
        }


def _validate_metadata(
    metadata_snapshots: Mapping[MetadataKey, ProviderModelMetadataSnapshot],
) -> None:
    expected_keys = {(candidate.provider, candidate.model) for candidate in CANDIDATES}
    if set(metadata_snapshots) != expected_keys:
        raise ValueError("metadata snapshots must match the four frozen provider/model identities")
    for (provider, model), snapshot in metadata_snapshots.items():
        if snapshot.provider != provider or snapshot.model != model:
            raise ValueError("metadata snapshot identity does not match its map key")
        if snapshot.supports_text_generation is not True:
            raise ValueError("text-generation capability must be confirmed before scheduling")
        if (snapshot.metadata_source, snapshot.metadata_method) != _METADATA_PROVENANCE[provider]:
            raise ValueError("metadata snapshot provenance is not valid for this provider")


def prepare_stage1b_schedule(
    metadata_snapshots: Mapping[MetadataKey, ProviderModelMetadataSnapshot],
    *,
    run_id: str,
    repository_commit: str,
) -> tuple[PreparedStage1BAttempt, ...]:
    """Build and validate the exact six-attempt schedule without invoking providers."""
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", run_id):
        raise ValueError("run_id must be a 1-64 character safe identifier")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", repository_commit):
        raise ValueError("repository_commit must be a 7-64 character hexadecimal commit")
    _validate_metadata(metadata_snapshots)

    datasets = {workload: load_dataset(workload) for workload in ("general", "research")}
    cases = {workload: datasets[workload].case(case_id) for workload, case_id in _CASE_IDS.items()}
    prepared: list[PreparedStage1BAttempt] = []
    for candidate in CANDIDATES:
        dataset = datasets[candidate.workload]
        case = cases[candidate.workload]
        snapshot = metadata_snapshots[(candidate.provider, candidate.model)]
        research_gemini = candidate.workload == "research" and candidate.provider == "gemini"
        policy = InvocationPolicy(
            output_limit_mode="explicit",
            output_limit=STAGE1B_OUTPUT_LIMIT,
            temperature_mode="explicit" if research_gemini else "unset",
            temperature=0.0 if research_gemini else None,
        )
        descriptor = EffectiveInvocationDescriptor(candidate, snapshot, policy)
        prepared.append(
            PreparedStage1BAttempt(
                dataset=dataset,
                case=case,
                descriptor=descriptor,
                repetition=STAGE1B_REPETITION,
                run_id=run_id,
                repository_commit=repository_commit,
                attempt_id=scheduled_attempt_id(
                    run_id=run_id,
                    dataset=dataset,
                    case_id=case.id,
                    descriptor=descriptor,
                    repetition=STAGE1B_REPETITION,
                ),
            )
        )

    expected_schedule = [
        (candidate.workload, candidate.id, _CASE_IDS[candidate.workload])
        for candidate in CANDIDATES
    ]
    actual_schedule = [
        (
            item.descriptor.candidate.workload,
            item.descriptor.candidate.id,
            item.case.id,
        )
        for item in prepared
    ]
    attempt_ids = [item.attempt_id for item in prepared]
    if actual_schedule != expected_schedule or len(set(attempt_ids)) != 6:
        raise ValueError("Stage 1B schedule does not match the six unique frozen attempts")
    return tuple(prepared)
