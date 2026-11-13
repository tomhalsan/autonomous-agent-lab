"""Direct, single-call Stage 1B adapter bridge for the Stage 0 evaluation runner."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

from agent_contracts import InferenceResponse
from research_agent.domain.models import SynthesisResult
from research_agent.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as ResearchCloudflareInferenceProvider,
)
from research_agent.infrastructure.gemini_inference import (
    GeminiInferenceProvider as ResearchGeminiInferenceProvider,
)
from worker.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as WorkerCloudflareInferenceProvider,
)
from worker.infrastructure.gemini_inference import (
    GeminiInferenceProvider as WorkerGeminiInferenceProvider,
)

from evaluation.inference.candidates import CANDIDATES
from evaluation.inference.datasets import load_dataset
from evaluation.inference.models import (
    EffectiveInvocationDescriptor,
    EvaluationCase,
    EvaluationDataset,
    ProviderOutput,
)
from evaluation.inference.runner import EvaluationInvoker, scheduled_attempt_id
from evaluation.inference.stage1b.preparation import (
    STAGE1B_OUTPUT_LIMIT,
    STAGE1B_REPETITION,
    PreparedStage1BAttempt,
)


class _GeneralAdapter(Protocol):
    async def generate(self, prompt: str) -> InferenceResponse: ...


class _ResearchAdapter(Protocol):
    async def synthesize(self, question: str, evidence: str) -> SynthesisResult: ...


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


def _usage_count(usage: object, *keys: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in keys:
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


def _research_output(response: SynthesisResult, provider: str) -> ProviderOutput:
    if provider == "cloudflare":
        usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    else:
        usage_keys = ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
    return ProviderOutput(
        text=response.content,
        provider=response.provider,
        model=response.model,
        input_tokens=_usage_count(response.usage, usage_keys[0]),
        output_tokens=_usage_count(response.usage, usage_keys[1]),
        total_tokens=_usage_count(response.usage, usage_keys[2]),
        finish_reason=response.finish_reason,
        truncated=response.truncated,
    )


def _validate_descriptor(
    item: PreparedStage1BAttempt,
    frozen_dataset: EvaluationDataset,
) -> None:
    descriptor = item.descriptor
    candidate = descriptor.candidate
    if candidate not in CANDIDATES:
        raise ValueError("candidate is not part of the frozen Stage 1B matrix")
    if item.case.workload != candidate.workload or item.dataset.name != candidate.workload:
        raise ValueError("descriptor, dataset, and case workloads must match")
    if item.dataset != frozen_dataset:
        raise ValueError("dataset differs from the frozen Stage 1B dataset")
    expected_case_id = (
        "general-debug-01" if candidate.workload == "general" else "research-insufficient-04"
    )
    if item.case != item.dataset.case(expected_case_id):
        raise ValueError("case is not the frozen Stage 1B smoke case")
    if item.repetition != STAGE1B_REPETITION:
        raise ValueError("Stage 1B requires repetition 0")
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
        metadata.provider != candidate.provider
        or metadata.model != candidate.model
        or metadata.supports_text_generation is not True
    ):
        raise ValueError("descriptor metadata must confirm the exact text-generation candidate")
    if (
        descriptor.effective_output_limit_mode != "explicit"
        or descriptor.effective_output_limit != STAGE1B_OUTPUT_LIMIT
    ):
        raise ValueError("Stage 1B adapters require the frozen explicit output limit")
    if candidate.adapter != (
        "cloudflare-workers-ai" if candidate.provider == "cloudflare" else "google-gemini"
    ):
        raise ValueError("candidate provider and adapter do not match")

    temperature_mode = descriptor.effective_temperature_mode
    temperature = descriptor.effective_temperature
    if candidate.provider == "cloudflare":
        if temperature_mode != "unset" or temperature is not None:
            raise ValueError("Cloudflare Stage 1B adapter does not accept an explicit temperature")
    elif candidate.workload == "general":
        if temperature_mode != "unset" or temperature is not None:
            raise ValueError("General Gemini Stage 1B requires unset temperature")
    elif temperature_mode != "explicit" or temperature != 0.0:
        raise ValueError("Research Gemini adapter requires its existing explicit 0.0 temperature")


class Stage1BInvoker(EvaluationInvoker):
    """Invoke only the six frozen direct adapters, once per prepared descriptor."""

    def __init__(
        self,
        schedule: Sequence[PreparedStage1BAttempt],
        *,
        cloudflare_account_id: str,
        cloudflare_api_token: str,
        gemini_api_key: str,
        cloudflare_client: Any | None = None,
        gemini_client: Any | None = None,
    ) -> None:
        self._cloudflare_account_id = _required_credential(
            cloudflare_account_id, "cloudflare_account_id"
        )
        self._cloudflare_api_token = _required_credential(
            cloudflare_api_token, "cloudflare_api_token"
        )
        self._gemini_api_key = _required_credential(gemini_api_key, "gemini_api_key")
        self._cloudflare_client = cloudflare_client
        self._gemini_client = gemini_client
        items = tuple(schedule)
        if len(items) != len(CANDIDATES):
            raise ValueError("Stage1BInvoker requires the complete six-attempt smoke schedule")
        if len({item.descriptor.candidate.id for item in items}) != len(items):
            raise ValueError("Stage 1B candidate descriptors must be unique")
        frozen_datasets = {workload: load_dataset(workload) for workload in ("general", "research")}
        for item in items:
            _validate_descriptor(item, frozen_datasets[item.descriptor.candidate.workload])
        run_ids = {item.run_id for item in items}
        commits = {item.repository_commit for item in items}
        if len(run_ids) != 1 or len(commits) != 1:
            raise ValueError("all Stage 1B attempts must share one run id and repository commit")

        self._prepared = {item.descriptor.candidate.id: item for item in items}
        self._general_adapters: dict[str, _GeneralAdapter] = {}
        self._research_adapters: dict[str, _ResearchAdapter] = {}
        for item in items:
            descriptor = item.descriptor
            candidate = descriptor.candidate
            output_limit = descriptor.effective_output_limit
            if candidate.workload == "general":
                if candidate.provider == "cloudflare":
                    adapter = WorkerCloudflareInferenceProvider(
                        account_id=self._cloudflare_account_id,
                        api_token=self._cloudflare_api_token,
                        model=candidate.model,
                        client=self._cloudflare_client,
                        max_tokens=output_limit,
                    )
                else:
                    temperature = (
                        descriptor.effective_temperature
                        if descriptor.effective_temperature_mode == "explicit"
                        else None
                    )
                    adapter = WorkerGeminiInferenceProvider(
                        api_key=self._gemini_api_key,
                        model=candidate.model,
                        client=self._gemini_client,
                        max_tokens=output_limit,
                        temperature=temperature,
                    )
                self._general_adapters[candidate.id] = adapter
            elif candidate.provider == "cloudflare":
                adapter = ResearchCloudflareInferenceProvider(
                    account_id=self._cloudflare_account_id,
                    api_token=self._cloudflare_api_token,
                    model=candidate.model,
                    client=self._cloudflare_client,
                    max_tokens=output_limit,
                )
                self._research_adapters[candidate.id] = adapter
            else:
                adapter = ResearchGeminiInferenceProvider(
                    api_key=self._gemini_api_key,
                    model=candidate.model,
                    client=self._gemini_client,
                    max_tokens=output_limit,
                )
                self._research_adapters[candidate.id] = adapter

    def __repr__(self) -> str:
        return "Stage1BInvoker()"

    async def invoke(
        self,
        descriptor: EffectiveInvocationDescriptor,
        case: EvaluationCase,
    ) -> ProviderOutput:
        candidate_id = descriptor.candidate.id
        prepared = self._prepared.get(candidate_id)
        if prepared is None or prepared.descriptor != descriptor or prepared.case != case:
            raise ValueError("invocation does not match the prepared Stage 1B schedule")

        if candidate_id in self._general_adapters:
            response = await self._general_adapters[candidate_id].generate(case.general_prompt())
            return _worker_output(response)

        if case.question is None:
            raise ValueError("Research case must contain a question")
        response = await self._research_adapters[candidate_id].synthesize(
            case.question,
            case.research_evidence(),
        )
        return _research_output(response, descriptor.candidate.provider)
