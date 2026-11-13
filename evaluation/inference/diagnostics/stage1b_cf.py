"""Evaluation-only preparation and execution for the four-call Cloudflare diagnostic campaign."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_contracts import InferenceResponse
from research_agent.domain.models import SynthesisResult
from research_agent.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as ResearchCloudflareInferenceProvider,
)
from worker.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as WorkerCloudflareInferenceProvider,
)

from evaluation.inference.candidates import CANDIDATES
from evaluation.inference.datasets import load_dataset
from evaluation.inference.models import (
    Candidate,
    EffectiveInvocationDescriptor,
    EvaluationAttempt,
    EvaluationCase,
    EvaluationDataset,
    ExperimentPolicy,
    InvocationPolicy,
    ProviderModelMetadataSnapshot,
    ProviderOutput,
)
from evaluation.inference.runner import EvaluationInvoker, run_evaluation, scheduled_attempt_id
from evaluation.inference.stage1a.metadata_resolvers import CloudflareMetadataResolver

DIAGNOSTIC_RUN_ID = "stage1b-cf-diagnostic-20261113-01"
DIAGNOSTIC_OUTPUT_LIMIT = 1024
DIAGNOSTIC_REPETITION = 0

_CASE_IDS = {
    "general": "general-debug-01",
    "research": "research-insufficient-04",
}

CLOUDFLARE_DIAGNOSTIC_CANDIDATES: tuple[Candidate, ...] = tuple(
    candidate for candidate in CANDIDATES if candidate.provider == "cloudflare"
)

EXPECTED_CLOUDFLARE_MODELS: frozenset[str] = frozenset(
    candidate.model for candidate in CLOUDFLARE_DIAGNOSTIC_CANDIDATES
)


def _required_credential(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _known_truncation(finish_reason: str | None) -> bool | None:
    if finish_reason is None:
        return None
    if finish_reason.strip().lower() in {
        "length",
        "max_tokens",
        "token_limit",
        "max_tokens_exceeded",
    }:
        return True
    return None


def _usage_count(usage: object, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    if type(value) is int and value >= 0:
        return value
    return None


def _worker_output(response: InferenceResponse) -> ProviderOutput:
    return ProviderOutput(
        text=response.text,
        provider=response.provider_name,
        model=response.model_name,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        total_tokens=response.total_tokens,
        finish_reason=response.finish_reason,
        truncated=_known_truncation(response.finish_reason),
    )


def _research_output(response: SynthesisResult) -> ProviderOutput:
    return ProviderOutput(
        text=response.content,
        provider=response.provider,
        model=response.model,
        input_tokens=_usage_count(response.usage, "prompt_tokens"),
        output_tokens=_usage_count(response.usage, "completion_tokens"),
        total_tokens=_usage_count(response.usage, "total_tokens"),
        finish_reason=response.finish_reason,
        truncated=response.truncated,
    )


@dataclass(frozen=True)
class PreparedStage1BCFAttempt:
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


def _validate_diagnostic_metadata(
    metadata_snapshots: Mapping[tuple[str, str], ProviderModelMetadataSnapshot],
) -> None:
    expected_keys = {("cloudflare", model) for model in EXPECTED_CLOUDFLARE_MODELS}
    if set(metadata_snapshots) != expected_keys:
        raise ValueError(
            "metadata snapshots must match the three unique Cloudflare diagnostic model identities"
        )
    for (provider, model), snapshot in metadata_snapshots.items():
        if provider != "cloudflare" or snapshot.provider != "cloudflare" or snapshot.model != model:
            raise ValueError("metadata snapshot identity does not match its map key")
        if snapshot.supports_text_generation is not True:
            raise ValueError("text-generation capability must be confirmed before scheduling")
        if (snapshot.metadata_source, snapshot.metadata_method) != (
            "account_catalog",
            "account_catalog",
        ):
            raise ValueError("metadata snapshot provenance is not valid for Cloudflare")


async def resolve_stage1b_cf_diagnostic_metadata(
    resolver: CloudflareMetadataResolver,
) -> dict[tuple[str, str], ProviderModelMetadataSnapshot]:
    """Resolve fresh metadata for the three unique Cloudflare diagnostic models (budget: 6 GETs)."""
    snapshots: dict[tuple[str, str], ProviderModelMetadataSnapshot] = {}
    for candidate in CLOUDFLARE_DIAGNOSTIC_CANDIDATES:
        key = ("cloudflare", candidate.model)
        if key in snapshots:
            continue
        snapshot = await resolver.resolve(candidate)
        snapshots[key] = snapshot
    return snapshots


def prepare_stage1b_cf_diagnostic_schedule(
    metadata_snapshots: Mapping[tuple[str, str], ProviderModelMetadataSnapshot],
    *,
    run_id: str = DIAGNOSTIC_RUN_ID,
    repository_commit: str,
) -> tuple[PreparedStage1BCFAttempt, ...]:
    """Build and validate the exact four-attempt diagnostic schedule."""
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", run_id):
        raise ValueError("run_id must be a 1-64 character safe identifier")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", repository_commit):
        raise ValueError("repository_commit must be a 7-64 character hexadecimal commit")
    _validate_diagnostic_metadata(metadata_snapshots)

    datasets = {workload: load_dataset(workload) for workload in ("general", "research")}
    cases = {workload: datasets[workload].case(case_id) for workload, case_id in _CASE_IDS.items()}
    prepared: list[PreparedStage1BCFAttempt] = []
    for candidate in CLOUDFLARE_DIAGNOSTIC_CANDIDATES:
        dataset = datasets[candidate.workload]
        case = cases[candidate.workload]
        snapshot = metadata_snapshots[(candidate.provider, candidate.model)]
        policy = InvocationPolicy(
            output_limit_mode="explicit",
            output_limit=DIAGNOSTIC_OUTPUT_LIMIT,
            temperature_mode="unset",
            temperature=None,
        )
        descriptor = EffectiveInvocationDescriptor(candidate, snapshot, policy)
        prepared.append(
            PreparedStage1BCFAttempt(
                dataset=dataset,
                case=case,
                descriptor=descriptor,
                repetition=DIAGNOSTIC_REPETITION,
                run_id=run_id,
                repository_commit=repository_commit,
                attempt_id=scheduled_attempt_id(
                    run_id=run_id,
                    dataset=dataset,
                    case_id=case.id,
                    descriptor=descriptor,
                    repetition=DIAGNOSTIC_REPETITION,
                ),
            )
        )

    expected_schedule = [
        (candidate.workload, candidate.id, _CASE_IDS[candidate.workload])
        for candidate in CLOUDFLARE_DIAGNOSTIC_CANDIDATES
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
    if actual_schedule != expected_schedule or len(set(attempt_ids)) != 4:
        raise ValueError("Diagnostic schedule does not match the four unique Cloudflare attempts")
    return tuple(prepared)


def _validate_descriptor(
    item: PreparedStage1BCFAttempt,
    frozen_dataset: EvaluationDataset,
) -> None:
    descriptor = item.descriptor
    candidate = descriptor.candidate
    if candidate not in CLOUDFLARE_DIAGNOSTIC_CANDIDATES:
        raise ValueError("candidate is not part of the Cloudflare diagnostic matrix")
    if candidate.provider != "cloudflare":
        raise ValueError("only Cloudflare candidates are allowed in this diagnostic")
    if item.case.workload != candidate.workload or item.dataset.name != candidate.workload:
        raise ValueError("descriptor, dataset, and case workloads must match")
    if item.dataset != frozen_dataset:
        raise ValueError("dataset differs from the frozen evaluation dataset")
    expected_case_id = _CASE_IDS[candidate.workload]
    if item.case != item.dataset.case(expected_case_id):
        raise ValueError("case is not the frozen diagnostic case")
    if item.repetition != DIAGNOSTIC_REPETITION:
        raise ValueError("Cloudflare diagnostic requires repetition 0")
    if item.attempt_id != scheduled_attempt_id(
        run_id=item.run_id,
        dataset=item.dataset,
        case_id=item.case.id,
        descriptor=descriptor,
        repetition=item.repetition,
    ):
        raise ValueError("attempt id does not match the prepared descriptor")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", item.repository_commit):
        raise ValueError("repository_commit must be a 7-64 character hexadecimal commit")
    metadata = descriptor.metadata
    if (
        metadata.provider != "cloudflare"
        or metadata.model != candidate.model
        or metadata.supports_text_generation is not True
    ):
        raise ValueError("descriptor metadata must confirm Cloudflare text-generation candidate")
    if (
        descriptor.effective_output_limit_mode != "explicit"
        or descriptor.effective_output_limit != DIAGNOSTIC_OUTPUT_LIMIT
    ):
        raise ValueError("Cloudflare diagnostic requires explicit 1024 output limit")
    if candidate.adapter != "cloudflare-workers-ai":
        raise ValueError("candidate adapter must be cloudflare-workers-ai")
    if descriptor.effective_temperature_mode != "unset" or descriptor.effective_temperature is not None:
        raise ValueError("Cloudflare diagnostic requires unset temperature")


class Stage1BCFDiagnosticInvoker(EvaluationInvoker):
    """Direct, single-call Cloudflare-only evaluation invoker for the four diagnostic attempts."""

    def __init__(
        self,
        schedule: Sequence[PreparedStage1BCFAttempt],
        *,
        cloudflare_account_id: str,
        cloudflare_api_token: str,
        cloudflare_client: Any | None = None,
    ) -> None:
        self._cloudflare_account_id = _required_credential(
            cloudflare_account_id, "cloudflare_account_id"
        )
        self._cloudflare_api_token = _required_credential(
            cloudflare_api_token, "cloudflare_api_token"
        )
        self._cloudflare_client = cloudflare_client
        items = tuple(schedule)
        if len(items) != len(CLOUDFLARE_DIAGNOSTIC_CANDIDATES):
            raise ValueError(
                "Stage1BCFDiagnosticInvoker requires the complete four-attempt diagnostic schedule"
            )
        if len({item.descriptor.candidate.id for item in items}) != len(items):
            raise ValueError("Diagnostic candidate descriptors must be unique")
        frozen_datasets = {workload: load_dataset(workload) for workload in ("general", "research")}
        for item in items:
            _validate_descriptor(item, frozen_datasets[item.descriptor.candidate.workload])
        run_ids = {item.run_id for item in items}
        commits = {item.repository_commit for item in items}
        if len(run_ids) != 1 or len(commits) != 1:
            raise ValueError("all diagnostic attempts must share one run id and repository commit")

        self._prepared = {item.descriptor.candidate.id: item for item in items}
        self._general_adapters: dict[str, WorkerCloudflareInferenceProvider] = {}
        self._research_adapters: dict[str, ResearchCloudflareInferenceProvider] = {}
        for item in items:
            descriptor = item.descriptor
            candidate = descriptor.candidate
            output_limit = descriptor.effective_output_limit
            if candidate.workload == "general":
                self._general_adapters[candidate.id] = WorkerCloudflareInferenceProvider(
                    account_id=self._cloudflare_account_id,
                    api_token=self._cloudflare_api_token,
                    model=candidate.model,
                    client=self._cloudflare_client,
                    max_tokens=output_limit,
                )
            else:
                self._research_adapters[candidate.id] = ResearchCloudflareInferenceProvider(
                    account_id=self._cloudflare_account_id,
                    api_token=self._cloudflare_api_token,
                    model=candidate.model,
                    client=self._cloudflare_client,
                    max_tokens=output_limit,
                )

    def __repr__(self) -> str:
        return "Stage1BCFDiagnosticInvoker()"

    async def invoke(
        self,
        descriptor: EffectiveInvocationDescriptor,
        case: EvaluationCase,
    ) -> ProviderOutput:
        candidate_id = descriptor.candidate.id
        prepared = self._prepared.get(candidate_id)
        if prepared is None or prepared.descriptor != descriptor or prepared.case != case:
            raise ValueError("invocation does not match the prepared diagnostic schedule")

        if candidate_id in self._general_adapters:
            response = await self._general_adapters[candidate_id].generate(case.general_prompt())
            return _worker_output(response)

        if case.question is None:
            raise ValueError("Research case must contain a question")
        response = await self._research_adapters[candidate_id].synthesize(
            case.question,
            case.research_evidence(),
        )
        return _research_output(response)


async def run_stage1b_cf_diagnostic(
    schedule: Sequence[PreparedStage1BCFAttempt],
    invoker: EvaluationInvoker,
    *,
    output_path: Path,
) -> list[EvaluationAttempt]:
    """Execute the four-attempt diagnostic campaign and append results to output_path."""
    attempts: list[EvaluationAttempt] = []
    items = tuple(schedule)
    if len(items) != 4:
        raise ValueError("Diagnostic execution requires exactly four scheduled attempts")
    for workload in ("general", "research"):
        workload_schedule = [
            item for item in items if item.descriptor.candidate.workload == workload
        ]
        if not workload_schedule:
            continue
        dataset = workload_schedule[0].dataset
        workload_attempts = await run_evaluation(
            dataset=dataset,
            candidates=[item.descriptor.candidate for item in workload_schedule],
            descriptors=[item.descriptor for item in workload_schedule],
            experiment_policy=ExperimentPolicy(
                dataset_name=workload,
                repetition=DIAGNOSTIC_REPETITION,
                case_ids=(workload_schedule[0].case.id,),
            ),
            invoker=invoker,
            output_path=output_path,
            run_id=items[0].run_id,
            repository_commit=items[0].repository_commit,
        )
        attempts.extend(workload_attempts)
    return attempts
