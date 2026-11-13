from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.inference.candidates import CANDIDATES
from evaluation.inference.datasets import load_dataset
from evaluation.inference.models import (
    Candidate,
    EffectiveInvocationDescriptor,
    EvaluationAttempt,
    EvaluationCase,
    ExperimentPolicy,
    InvocationPolicy,
    ProviderModelMetadataSnapshot,
    ProviderOutput,
)
from evaluation.inference.reporting import read_attempts
from evaluation.inference.runner import run_evaluation, scheduled_attempt_id

_COMMIT = "3ee0bc92d33a98ac72fedb84bb1f929df6301a26"
_TIMESTAMP = "2025-01-01T00:00:00Z"


def _candidate(workload: str = "general", index: int = 0) -> Candidate:
    return tuple(candidate for candidate in CANDIDATES if candidate.workload == workload)[index]


def _metadata(
    candidate: Candidate,
    *,
    provider: str | None = None,
    model: str | None = None,
    default_output_tokens: int | None = 256,
    default_temperature: float | None = 0.4,
) -> ProviderModelMetadataSnapshot:
    return ProviderModelMetadataSnapshot(
        provider=candidate.provider if provider is None else provider,
        model=candidate.model if model is None else model,
        retrieved_at=_TIMESTAMP,
        metadata_source="offline_fixture",
        metadata_method="fixture",
        supports_text_generation=True,
        context_limit_tokens=4096,
        input_limit_tokens=None,
        max_output_tokens=1024,
        default_output_tokens=default_output_tokens,
        temperature_supported=True,
        min_temperature=0.0,
        default_temperature=default_temperature,
        max_temperature=1.0,
        reasoning_supported=None,
        reasoning_default=None,
    )


def _descriptor(
    candidate: Candidate,
    *,
    metadata: ProviderModelMetadataSnapshot | None = None,
    policy: InvocationPolicy | None = None,
) -> EffectiveInvocationDescriptor:
    return EffectiveInvocationDescriptor(
        candidate=candidate,
        metadata=metadata or _metadata(candidate),
        policy=policy or InvocationPolicy(),
    )


def _run_policy(
    dataset: str,
    *,
    case_ids: tuple[str, ...] | None = ("general-json-06",),
    repetition: int = 0,
) -> ExperimentPolicy:
    return ExperimentPolicy(
        dataset_name=dataset,
        case_ids=case_ids,
        repetition=repetition,
    )


class FakeInvoker:
    def __init__(self, *, fail: bool = False, no_usage: bool = False) -> None:
        self.fail = fail
        self.no_usage = no_usage
        self.calls: list[tuple[str, str]] = []

    async def invoke(
        self,
        descriptor: EffectiveInvocationDescriptor,
        case: EvaluationCase,
    ) -> ProviderOutput:
        candidate = descriptor.candidate
        self.calls.append((candidate.id, case.id))
        if self.fail:
            raise RuntimeError("private diagnostic body must not be persisted")
        return ProviderOutput(
            text='{"status":"completed","count":4}',
            provider=candidate.provider,
            model=candidate.model,
            input_tokens=None if self.no_usage else 12,
            output_tokens=None if self.no_usage else 8,
            finish_reason="stop",
            truncated=False,
        )


async def _run(
    *,
    candidates: tuple[Candidate, ...],
    descriptors: tuple[EffectiveInvocationDescriptor, ...],
    output_path: Path,
    run_id: str,
    dataset_name: str = "general",
    case_ids: tuple[str, ...] | None = ("general-json-06",),
    repetition: int = 0,
    invoker: FakeInvoker | None = None,
) -> list[EvaluationAttempt]:
    dataset = load_dataset(dataset_name)
    return await run_evaluation(
        dataset=dataset,
        candidates=candidates,
        descriptors=descriptors,
        experiment_policy=_run_policy(
            dataset_name,
            case_ids=case_ids,
            repetition=repetition,
        ),
        invoker=invoker or FakeInvoker(),
        output_path=output_path,
        run_id=run_id,
        repository_commit=_COMMIT,
        clock_ns=iter(range(0, 1000, 10)).__next__,
        timestamp_factory=lambda: _TIMESTAMP,
    )


@pytest.mark.anyio
async def test_attempt_ids_are_stable_across_candidate_loop_order(tmp_path: Path) -> None:
    candidates = tuple(candidate for candidate in CANDIDATES if candidate.workload == "general")
    descriptors = tuple(_descriptor(candidate) for candidate in candidates)

    first = await _run(
        candidates=candidates,
        descriptors=descriptors,
        output_path=tmp_path / "first.jsonl",
        run_id="stable-run",
    )
    second = await _run(
        candidates=tuple(reversed(candidates)),
        descriptors=tuple(reversed(descriptors)),
        output_path=tmp_path / "second.jsonl",
        run_id="stable-run",
    )

    assert {attempt.candidate_id: attempt.attempt_id for attempt in first} == {
        attempt.candidate_id: attempt.attempt_id for attempt in second
    }


def test_scheduled_attempt_id_includes_each_identity_dimension() -> None:
    general = load_dataset("general")
    candidate = _candidate()
    descriptor = _descriptor(candidate)
    base = scheduled_attempt_id(
        run_id="identity-run",
        dataset=general,
        case_id="general-debug-01",
        descriptor=descriptor,
        repetition=0,
    )
    name_changed = replace(general, name="research")
    version_changed = replace(general, version="1.0.2")
    digest_changed = replace(general, sha256="f" * 64)
    other_candidate = _descriptor(replace(candidate, id="other-candidate"))
    other_workload = _descriptor(replace(candidate, workload="research"))
    identities = {
        base,
        scheduled_attempt_id(
            run_id="other-run",
            dataset=general,
            case_id="general-debug-01",
            descriptor=descriptor,
            repetition=0,
        ),
        scheduled_attempt_id(
            run_id="identity-run",
            dataset=name_changed,
            case_id="general-debug-01",
            descriptor=descriptor,
            repetition=0,
        ),
        scheduled_attempt_id(
            run_id="identity-run",
            dataset=version_changed,
            case_id="general-debug-01",
            descriptor=descriptor,
            repetition=0,
        ),
        scheduled_attempt_id(
            run_id="identity-run",
            dataset=digest_changed,
            case_id="general-debug-01",
            descriptor=descriptor,
            repetition=0,
        ),
        scheduled_attempt_id(
            run_id="identity-run",
            dataset=general,
            case_id="general-explanation-04",
            descriptor=descriptor,
            repetition=0,
        ),
        scheduled_attempt_id(
            run_id="identity-run",
            dataset=general,
            case_id="general-debug-01",
            descriptor=other_candidate,
            repetition=0,
        ),
        scheduled_attempt_id(
            run_id="identity-run",
            dataset=general,
            case_id="general-debug-01",
            descriptor=other_workload,
            repetition=0,
        ),
        scheduled_attempt_id(
            run_id="identity-run",
            dataset=general,
            case_id="general-debug-01",
            descriptor=descriptor,
            repetition=1,
        ),
    }

    assert len(identities) == 9
    assert base == scheduled_attempt_id(
        run_id="identity-run",
        dataset=general,
        case_id="general-debug-01",
        descriptor=descriptor,
        repetition=0,
    )


def test_invalid_repetition_is_rejected_before_invocation() -> None:
    invoker = FakeInvoker()

    with pytest.raises(ValueError, match="non-negative integer"):
        ExperimentPolicy(dataset_name="general", repetition=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        ExperimentPolicy(dataset_name="general", repetition=True)

    assert invoker.calls == []


@pytest.mark.anyio
async def test_duplicate_attempt_in_existing_jsonl_fails_before_invocation(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    descriptor = _descriptor(candidate)
    output_path = tmp_path / "results.jsonl"
    first = FakeInvoker()
    await _run(
        candidates=(candidate,),
        descriptors=(descriptor,),
        output_path=output_path,
        run_id="duplicate-run",
        invoker=first,
    )
    duplicate = FakeInvoker()

    with pytest.raises(ValueError, match="Attempt already persisted"):
        await _run(
            candidates=(candidate,),
            descriptors=(descriptor,),
            output_path=output_path,
            run_id="duplicate-run",
            invoker=duplicate,
        )

    assert len(first.calls) == 1
    assert duplicate.calls == []
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.anyio
async def test_separate_calls_append_distinct_candidates_cases_and_repetitions(
    tmp_path: Path,
) -> None:
    candidate_a, candidate_b = tuple(
        candidate for candidate in CANDIDATES if candidate.workload == "general"
    )[:2]
    output_path = tmp_path / "appended.jsonl"
    scheduled = (
        (candidate_a, "general-debug-01", 0),
        (candidate_b, "general-debug-01", 0),
        (candidate_a, "general-explanation-04", 0),
        (candidate_a, "general-debug-01", 1),
    )

    records = []
    for candidate, case_id, repetition in scheduled:
        records.extend(
            await _run(
                candidates=(candidate,),
                descriptors=(_descriptor(candidate),),
                output_path=output_path,
                run_id="append-run",
                case_ids=(case_id,),
                repetition=repetition,
            )
        )

    assert len(records) == 4
    assert len({record.attempt_id for record in records}) == 4
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 4
    assert [record.repetition for record in records] == [0, 0, 0, 1]


@pytest.mark.parametrize(
    ("provider", "model", "identity_field"),
    [
        ("different-provider", None, "provider"),
        (None, "different-model", "model"),
    ],
)
def test_metadata_candidate_identity_mismatch_is_rejected(
    provider: str | None,
    model: str | None,
    identity_field: str,
) -> None:
    candidate = _candidate()
    mismatch = _metadata(candidate, provider=provider, model=model)

    with pytest.raises(ValueError, match=f"metadata {identity_field} does not match"):
        _descriptor(candidate, metadata=mismatch)


@pytest.mark.anyio
async def test_descriptor_must_match_scheduled_candidate_before_invocation(
    tmp_path: Path,
) -> None:
    candidate_a, candidate_b = tuple(
        candidate for candidate in CANDIDATES if candidate.workload == "general"
    )[:2]
    invoker = FakeInvoker()

    with pytest.raises(ValueError, match="descriptors must match"):
        await _run(
            candidates=(candidate_b,),
            descriptors=(_descriptor(candidate_a),),
            output_path=tmp_path / "mismatch.jsonl",
            run_id="descriptor-mismatch",
            invoker=invoker,
        )

    assert invoker.calls == []


@pytest.mark.anyio
async def test_per_candidate_effective_settings_persist_and_unknowns_stay_null(
    tmp_path: Path,
) -> None:
    candidate_a, candidate_b = tuple(
        candidate for candidate in CANDIDATES if candidate.workload == "general"
    )[:2]
    known_a = _descriptor(
        candidate_a,
        policy=InvocationPolicy(
            output_limit_mode="provider_default",
            temperature_mode="provider_default",
        ),
    )
    known_b = _descriptor(
        candidate_b,
        metadata=_metadata(
            candidate_b,
            default_output_tokens=128,
            default_temperature=0.2,
        ),
        policy=InvocationPolicy(
            output_limit_mode="explicit",
            output_limit=64,
            temperature_mode="explicit",
            temperature=0.0,
        ),
    )
    unknown_metadata = ProviderModelMetadataSnapshot(
        provider=None,
        model=None,
        retrieved_at=_TIMESTAMP,
        metadata_source="offline_fixture",
        metadata_method="fixture",
    )
    unknown = _descriptor(
        candidate_a,
        metadata=unknown_metadata,
        policy=InvocationPolicy(
            output_limit_mode="provider_default",
            temperature_mode="provider_default",
        ),
    )
    assert unknown.effective_output_limit is None
    assert unknown.effective_temperature is None

    attempts = await _run(
        candidates=(candidate_a, candidate_b),
        descriptors=(known_a, known_b),
        output_path=tmp_path / "descriptors.jsonl",
        run_id="descriptor-run",
    )

    assert attempts[0].invocation.effective_output_limit == 256
    assert attempts[0].invocation.effective_temperature == 0.4
    assert attempts[1].invocation.effective_output_limit == 64
    assert attempts[1].invocation.effective_temperature == 0.0
    assert attempts[0].invocation.effective_output_limit_mode == "provider_default"
    assert attempts[1].invocation.policy.output_limit_mode == "explicit"
    assert attempts[1].invocation.metadata.retrieved_at == _TIMESTAMP
    assert attempts[1].invocation.metadata.metadata_source == "offline_fixture"
    assert attempts[1].invocation.metadata.metadata_method == "fixture"
    persisted = [
        json.loads(line) for line in (tmp_path / "descriptors.jsonl").read_text().splitlines()
    ]
    assert persisted[0]["invocation"]["candidate"]["id"] == candidate_a.id
    assert persisted[1]["invocation"]["candidate"]["id"] == candidate_b.id
    assert persisted[0]["invocation"]["metadata"]["context_limit_tokens"] == 4096
    assert persisted[1]["invocation"]["effective_temperature"] == 0.0


@pytest.mark.anyio
async def test_failure_retains_descriptor_and_sanitized_category(tmp_path: Path) -> None:
    candidate = _candidate()
    descriptor = _descriptor(
        candidate,
        policy=InvocationPolicy(
            output_limit_mode="explicit",
            output_limit=64,
            temperature_mode="explicit",
            temperature=0.0,
        ),
    )
    invoker = FakeInvoker(fail=True)

    attempts = await _run(
        candidates=(candidate,),
        descriptors=(descriptor,),
        output_path=tmp_path / "failure.jsonl",
        run_id="failure-run",
        invoker=invoker,
    )

    assert attempts[0].status == "failed"
    assert attempts[0].failure_category == "execution_error"
    assert attempts[0].invocation == descriptor
    assert attempts[0].output is None
    assert invoker.calls == [(candidate.id, "general-json-06")]
    assert "private diagnostic body" not in (tmp_path / "failure.jsonl").read_text()


@pytest.mark.anyio
async def test_result_jsonl_round_trips_typed_descriptor_and_token_unknowns(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    descriptor = _descriptor(
        candidate,
        metadata=ProviderModelMetadataSnapshot(
            provider=None,
            model=None,
            retrieved_at=_TIMESTAMP,
            metadata_source="offline_fixture",
            metadata_method="fixture",
        ),
    )
    output_path = tmp_path / "roundtrip.jsonl"
    attempts = await _run(
        candidates=(candidate,),
        descriptors=(descriptor,),
        output_path=output_path,
        run_id="roundtrip",
        invoker=FakeInvoker(no_usage=True),
    )

    assert read_attempts(output_path) == attempts
    data = json.loads(output_path.read_text(encoding="utf-8"))
    metadata = data["invocation"]["metadata"]
    assert metadata["provider"] is None
    assert metadata["model"] is None
    assert data["input_tokens"] is None
    assert data["output_tokens"] is None
    assert not any(isinstance(value, dict) for value in metadata.values())
    forbidden_fields = {
        "api_key",
        "api_token",
        "access_token",
        "credentials",
        "raw_response",
        "raw_exception",
    }

    def nested_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return {key for key in value if isinstance(key, str)}.union(
                *(nested_keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(nested_keys(item) for item in value))
        return set()

    assert nested_keys(data).isdisjoint(forbidden_fields)


def test_arbitrary_metadata_fields_are_rejected_and_schema_is_allowlisted() -> None:
    candidate = _candidate()
    with pytest.raises(TypeError):
        ProviderModelMetadataSnapshot(
            provider=candidate.provider,
            model=candidate.model,
            retrieved_at=_TIMESTAMP,
            metadata_source="offline_fixture",
            metadata_method="fixture",
            arbitrary_metadata={"api_token": "must-not-persist"},  # type: ignore[call-arg]
        )

    assert "arbitrary_metadata" not in ProviderModelMetadataSnapshot.__dataclass_fields__
    assert "raw_response" not in ProviderModelMetadataSnapshot.__dataclass_fields__
    assert "credential" not in ProviderModelMetadataSnapshot.__dataclass_fields__


def test_evaluation_package_has_no_provider_or_network_imports() -> None:
    forbidden_roots = {
        "aiohttp",
        "agent_infrastructure",
        "google",
        "httpx",
        "os",
        "requests",
        "research_agent",
        "socket",
        "urllib",
        "worker",
    }
    forbidden_constructors = {
        "AsyncClient",
        "CloudflareHealthProbe",
        "CloudflareInferenceProvider",
        "GeminiClient",
        "GeminiHealthProbe",
        "GeminiInferenceProvider",
    }
    package = Path(__file__).parents[1] / "evaluation" / "inference"

    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", maxsplit=1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".", maxsplit=1)[0]}
            else:
                roots = set()
            assert roots.isdisjoint(forbidden_roots), f"forbidden import in {path}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_constructors, (
                    f"forbidden provider/network construction in {path}"
                )
