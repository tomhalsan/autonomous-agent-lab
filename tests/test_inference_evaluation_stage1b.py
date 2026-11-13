import ast
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TypedDict

import httpx
import pytest

from evaluation.inference.candidates import CANDIDATES
from evaluation.inference.models import (
    EffectiveInvocationDescriptor,
    ExperimentPolicy,
    ProviderModelMetadataSnapshot,
)
from evaluation.inference.runner import run_evaluation
from evaluation.inference.stage1b import Stage1BInvoker, prepare_stage1b_schedule
from evaluation.inference.stage1b.preparation import PreparedStage1BAttempt

CF_TOKEN = "stage1b-cloudflare-secret-sentinel"
GEMINI_KEY = "stage1b-gemini-secret-sentinel"
GENERAL_TEXT = (
    "The empty list causes IndexError because index 0 does not exist. "
    "Add `if not items: return None` before `items[0]`."
)
RESEARCH_TEXT = (
    "The evidence is insufficient to determine the outage's cause; "
    "it does not mention a battery defect [S1]."
)


class _CapturedRequest(TypedDict):
    provider: str
    method: str
    path: str
    payload: Any


def _metadata_snapshots() -> dict[tuple[str, str], ProviderModelMetadataSnapshot]:
    snapshots: dict[tuple[str, str], ProviderModelMetadataSnapshot] = {}
    for candidate in CANDIDATES:
        key = (candidate.provider, candidate.model)
        if key in snapshots:
            continue
        if candidate.provider == "cloudflare":
            snapshots[key] = ProviderModelMetadataSnapshot(
                provider=candidate.provider,
                model=candidate.model,
                retrieved_at="2026-11-13T08:00:00+00:00",
                metadata_source="account_catalog",
                metadata_method="account_catalog",
                supports_text_generation=True,
                default_output_tokens=(
                    256 if candidate.model == "@cf/openai/gpt-oss-20b" else None
                ),
                temperature_supported=True,
                min_temperature=0.0,
                default_temperature=1.0,
                max_temperature=5.0 if candidate.model == "@cf/openai/gpt-oss-20b" else 2.0,
                reasoning_supported=True,
            )
        else:
            snapshots[key] = ProviderModelMetadataSnapshot(
                provider=candidate.provider,
                model=candidate.model,
                retrieved_at="2026-11-13T08:00:00+00:00",
                metadata_source="provider_api",
                metadata_method="model_get",
                supports_text_generation=True,
                input_limit_tokens=1_048_576,
                max_output_tokens=65_536,
                temperature_supported=True,
                default_temperature=1.0,
                max_temperature=2.0,
                reasoning_supported=True,
            )
    return snapshots


def _schedule() -> tuple[PreparedStage1BAttempt, ...]:
    return prepare_stage1b_schedule(
        _metadata_snapshots(),
        run_id="stage1b-20261113-0817",
        repository_commit="98f8d9b",
    )


def _cloudflare_result(
    text: str,
    *,
    usage: dict[str, int] | None = None,
    finish_reason: str = "stop",
) -> httpx.Response:
    result: dict[str, object] = {"response": text, "finish_reason": finish_reason}
    if usage is not None:
        result["usage"] = usage
    return httpx.Response(200, json={"success": True, "result": result})


def _gemini_result(
    text: str,
    *,
    usage: dict[str, int] | None = None,
    finish_reason: str = "STOP",
) -> httpx.Response:
    body: dict[str, object] = {
        "candidates": [
            {
                "content": {"parts": [{"text": text}]},
                "finishReason": finish_reason,
            }
        ]
    }
    if usage is not None:
        body["usageMetadata"] = usage
    return httpx.Response(200, json=body)


def _clients(
    cloudflare_response: Callable[[_CapturedRequest], httpx.Response] | None = None,
    gemini_response: Callable[[_CapturedRequest], httpx.Response] | None = None,
) -> tuple[httpx.AsyncClient, httpx.AsyncClient, list[_CapturedRequest]]:
    requests: list[_CapturedRequest] = []

    def cloudflare_handler(request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or "/ai/run/" not in request.url.path:
            raise AssertionError("unexpected Cloudflare request")
        payload = json.loads(request.content)
        request_record: _CapturedRequest = {
            "provider": "cloudflare",
            "method": request.method,
            "path": request.url.path,
            "payload": payload,
        }
        requests.append(request_record)
        if cloudflare_response is not None:
            return cloudflare_response(request_record)
        messages = payload["messages"]
        user_text = messages[1]["content"]
        text = RESEARCH_TEXT if "Research Question:\n" in user_text else GENERAL_TEXT
        return _cloudflare_result(
            text,
            usage={"prompt_tokens": 11, "completion_tokens": 13, "total_tokens": 24},
        )

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or not request.url.path.endswith(":generateContent"):
            raise AssertionError("unexpected Gemini request")
        payload = json.loads(request.content)
        request_record: _CapturedRequest = {
            "provider": "gemini",
            "method": request.method,
            "path": request.url.path,
            "payload": payload,
        }
        requests.append(request_record)
        if gemini_response is not None:
            return gemini_response(request_record)
        text = payload["contents"][0]["parts"][0]["text"]
        output = RESEARCH_TEXT if "Research Question:\n" in text else GENERAL_TEXT
        return _gemini_result(
            output,
            usage={
                "promptTokenCount": 17,
                "candidatesTokenCount": 19,
                "totalTokenCount": 36,
            },
        )

    return (
        httpx.AsyncClient(transport=httpx.MockTransport(cloudflare_handler)),
        httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler)),
        requests,
    )


def _invoker(
    schedule: tuple[PreparedStage1BAttempt, ...],
    cloudflare_client: httpx.AsyncClient,
    gemini_client: httpx.AsyncClient,
) -> Stage1BInvoker:
    return Stage1BInvoker(
        schedule,
        cloudflare_account_id="stage1b-account-sentinel",
        cloudflare_api_token=CF_TOKEN,
        gemini_api_key=GEMINI_KEY,
        cloudflare_client=cloudflare_client,
        gemini_client=gemini_client,
    )


def _item(
    schedule: tuple[PreparedStage1BAttempt, ...],
    candidate_id: str,
) -> PreparedStage1BAttempt:
    return next(item for item in schedule if item.descriptor.candidate.id == candidate_id)


def test_schedule_prepares_exact_six_frozen_attempts_and_reuses_snapshots() -> None:
    schedule = _schedule()

    assert len(schedule) == 6
    assert [item.descriptor.candidate for item in schedule] == list(CANDIDATES)
    assert [item.case.id for item in schedule] == [
        "general-debug-01",
        "general-debug-01",
        "general-debug-01",
        "research-insufficient-04",
        "research-insufficient-04",
        "research-insufficient-04",
    ]
    assert all(item.repetition == 0 for item in schedule)
    assert len({item.attempt_id for item in schedule}) == 6
    assert all(item.descriptor.effective_output_limit == 1024 for item in schedule)
    assert [item.descriptor.effective_temperature_mode for item in schedule] == [
        "unset",
        "unset",
        "unset",
        "unset",
        "unset",
        "explicit",
    ]
    assert schedule[-1].descriptor.effective_temperature == 0.0
    assert schedule[1].descriptor.metadata is schedule[4].descriptor.metadata
    assert schedule[2].descriptor.metadata is schedule[5].descriptor.metadata
    assert schedule[0].case is schedule[2].case
    assert schedule[3].case is schedule[5].case
    assert "secret" not in json.dumps(schedule[0].to_sanitized_dict())


def test_schedule_rejects_missing_or_unconfirmed_metadata() -> None:
    snapshots = _metadata_snapshots()
    snapshots.pop(("gemini", "gemini-3.5-flash-lite"))
    with pytest.raises(ValueError, match="four frozen"):
        prepare_stage1b_schedule(
            snapshots,
            run_id="stage1b-test",
            repository_commit="98f8d9b",
        )

    snapshots = _metadata_snapshots()
    key = ("cloudflare", "@cf/openai/gpt-oss-20b")
    snapshots[key] = replace(snapshots[key], supports_text_generation=None)
    with pytest.raises(ValueError, match="must be confirmed"):
        prepare_stage1b_schedule(
            snapshots,
            run_id="stage1b-test",
            repository_commit="98f8d9b",
        )


@pytest.mark.anyio
async def test_general_cloudflare_uses_one_exact_messages_request() -> None:
    schedule = _schedule()
    cloudflare_client, gemini_client, requests = _clients()
    async with cloudflare_client, gemini_client:
        invoker = _invoker(schedule, cloudflare_client, gemini_client)
        item = schedule[0]
        result = await invoker.invoke(item.descriptor, item.case)

    assert len(requests) == 1
    request = requests[0]
    assert request["provider"] == "cloudflare"
    assert request["path"].endswith("/ai/run/@cf/google/gemma-4-26b-a4b-it")
    payload = request["payload"]
    assert payload["max_tokens"] == 1024
    assert "temperature" not in payload
    assert payload["messages"][1]["content"] == item.case.general_prompt()
    assert result.text == GENERAL_TEXT
    assert (result.provider, result.model) == ("cloudflare", item.descriptor.candidate.model)
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (11, 13, 24)
    assert result.finish_reason == "stop"
    assert result.truncated is None
    assert CF_TOKEN not in json.dumps(asdict(result))


@pytest.mark.anyio
async def test_general_gemini_uses_one_exact_generate_content_request() -> None:
    schedule = _schedule()
    cloudflare_client, gemini_client, requests = _clients()
    async with cloudflare_client, gemini_client:
        invoker = _invoker(schedule, cloudflare_client, gemini_client)
        item = schedule[2]
        result = await invoker.invoke(item.descriptor, item.case)

    assert len(requests) == 1
    request = requests[0]
    assert request["provider"] == "gemini"
    assert request["path"].endswith("/models/gemini-3.5-flash-lite:generateContent")
    payload = request["payload"]
    assert payload["generationConfig"] == {"maxOutputTokens": 1024}
    assert payload["contents"][0]["parts"][0]["text"] == item.case.general_prompt()
    assert result.text == GENERAL_TEXT
    assert (result.provider, result.model) == ("gemini", item.descriptor.candidate.model)
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (17, 19, 36)


@pytest.mark.anyio
async def test_research_cloudflare_uses_fixed_question_and_evidence_once() -> None:
    schedule = _schedule()
    cloudflare_client, gemini_client, requests = _clients()
    async with cloudflare_client, gemini_client:
        invoker = _invoker(schedule, cloudflare_client, gemini_client)
        item = schedule[3]
        result = await invoker.invoke(item.descriptor, item.case)

    assert len(requests) == 1
    request = requests[0]
    assert request["provider"] == "cloudflare"
    assert request["path"].endswith("/ai/run/@cf/qwen/qwen3.8-27b")
    payload = request["payload"]
    assert payload["max_tokens"] == 1024
    assert "temperature" not in payload
    prompt = payload["messages"][1]["content"]
    assert item.case.question in prompt
    assert item.case.research_evidence() in prompt
    assert result.text == RESEARCH_TEXT
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (11, 13, 24)
    assert result.truncated is False


@pytest.mark.anyio
async def test_research_gemini_uses_fixed_evidence_and_explicit_zero_temperature() -> None:
    schedule = _schedule()
    cloudflare_client, gemini_client, requests = _clients()
    async with cloudflare_client, gemini_client:
        invoker = _invoker(schedule, cloudflare_client, gemini_client)
        item = schedule[5]
        result = await invoker.invoke(item.descriptor, item.case)

    assert len(requests) == 1
    request = requests[0]
    assert request["provider"] == "gemini"
    assert request["path"].endswith("/models/gemini-3.5-flash-lite:generateContent")
    payload = request["payload"]
    assert payload["generationConfig"] == {
        "maxOutputTokens": 1024,
        "temperature": 0.0,
    }
    prompt = payload["contents"][0]["parts"][0]["text"]
    assert item.case.question in prompt
    assert item.case.research_evidence() in prompt
    assert result.text == RESEARCH_TEXT
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (17, 19, 36)
    assert result.truncated is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("candidate_id", "provider"),
    [
        ("general-cloudflare-gemma-4-26b-a4b-it", "cloudflare"),
        ("general-gemini-3.5-flash-lite", "gemini"),
    ],
)
async def test_missing_usage_stays_null(
    candidate_id: str,
    provider: str,
) -> None:
    schedule = _schedule()
    no_usage = lambda _request: (
        _cloudflare_result(GENERAL_TEXT)
        if provider == "cloudflare"
        else _gemini_result(GENERAL_TEXT)
    )
    cloudflare_client, gemini_client, requests = _clients(
        cloudflare_response=no_usage if provider == "cloudflare" else None,
        gemini_response=no_usage if provider == "gemini" else None,
    )
    async with cloudflare_client, gemini_client:
        invoker = _invoker(schedule, cloudflare_client, gemini_client)
        item = _item(schedule, candidate_id)
        result = await invoker.invoke(item.descriptor, item.case)

    assert len(requests) == 1
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.total_tokens is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("candidate_id", "provider", "finish_reason"),
    [
        ("general-cloudflare-gemma-4-26b-a4b-it", "cloudflare", "max_tokens"),
        ("general-gemini-3.5-flash-lite", "gemini", "MAX_TOKENS"),
    ],
)
async def test_known_token_limit_finish_reason_maps_truncation(
    candidate_id: str,
    provider: str,
    finish_reason: str,
) -> None:
    schedule = _schedule()

    def response(_request: _CapturedRequest) -> httpx.Response:
        if provider == "cloudflare":
            return _cloudflare_result(GENERAL_TEXT, finish_reason=finish_reason)
        return _gemini_result(GENERAL_TEXT, finish_reason=finish_reason)

    cloudflare_client, gemini_client, requests = _clients(
        cloudflare_response=response if provider == "cloudflare" else None,
        gemini_response=response if provider == "gemini" else None,
    )
    async with cloudflare_client, gemini_client:
        invoker = _invoker(schedule, cloudflare_client, gemini_client)
        item = _item(schedule, candidate_id)
        result = await invoker.invoke(item.descriptor, item.case)

    assert len(requests) == 1
    assert result.truncated is True


@pytest.mark.anyio
async def test_invoker_rejects_mismatches_and_unrepresentable_settings_pre_request() -> None:
    schedule = _schedule()
    cloudflare_client, gemini_client, requests = _clients()
    async with cloudflare_client, gemini_client:
        invoker = _invoker(schedule, cloudflare_client, gemini_client)
        item = schedule[0]
        with pytest.raises(ValueError, match="prepared Stage 1B schedule"):
            await invoker.invoke(item.descriptor, replace(item.case, instruction="changed"))
        with pytest.raises(ValueError, match="prepared Stage 1B schedule"):
            await invoker.invoke(item.descriptor, schedule[3].case)

    bad_candidate = replace(
        schedule[0].descriptor.candidate,
        adapter="google-gemini",
    )
    bad_descriptor = replace(schedule[0].descriptor, candidate=bad_candidate)
    bad_item = replace(schedule[0], descriptor=bad_descriptor)
    invalid_schedule = (bad_item, *schedule[1:])
    with pytest.raises(ValueError, match="frozen Stage 1B matrix"):
        Stage1BInvoker(
            invalid_schedule,
            cloudflare_account_id="account",
            cloudflare_api_token=CF_TOKEN,
            gemini_api_key=GEMINI_KEY,
        )

    unset_output_limit = replace(
        schedule[0].descriptor.policy,
        output_limit_mode="unset",
        output_limit=None,
    )
    unset_output_descriptor = EffectiveInvocationDescriptor(
        schedule[0].descriptor.candidate,
        schedule[0].descriptor.metadata,
        unset_output_limit,
    )
    unset_output_schedule = (
        replace(schedule[0], descriptor=unset_output_descriptor),
        *schedule[1:],
    )
    with pytest.raises(ValueError, match="frozen explicit output limit"):
        Stage1BInvoker(
            unset_output_schedule,
            cloudflare_account_id="account",
            cloudflare_api_token=CF_TOKEN,
            gemini_api_key=GEMINI_KEY,
        )

    cf_item = schedule[0]
    explicit_cf_temperature = replace(
        cf_item.descriptor.policy,
        temperature_mode="explicit",
        temperature=0.5,
    )
    unsupported_descriptor = EffectiveInvocationDescriptor(
        cf_item.descriptor.candidate,
        cf_item.descriptor.metadata,
        explicit_cf_temperature,
    )
    unsupported_schedule = (
        replace(cf_item, descriptor=unsupported_descriptor),
        *schedule[1:],
    )
    with pytest.raises(ValueError, match="does not accept an explicit temperature"):
        Stage1BInvoker(
            unsupported_schedule,
            cloudflare_account_id="account",
            cloudflare_api_token=CF_TOKEN,
            gemini_api_key=GEMINI_KEY,
        )
    assert requests == []


@pytest.mark.anyio
async def test_failed_call_is_not_retried_or_fallen_back_and_secrets_are_not_persisted(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    schedule = _schedule()

    def failed_response(_request: _CapturedRequest) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "success": False,
                "errors": [{"message": f"provider rejected {CF_TOKEN}"}],
            },
        )

    cloudflare_client, gemini_client, requests = _clients(cloudflare_response=failed_response)
    output_path = tmp_path / "attempts.jsonl"
    async with cloudflare_client, gemini_client:
        invoker = _invoker(schedule, cloudflare_client, gemini_client)
        item = schedule[0]
        attempts = await run_evaluation(
            dataset=item.dataset,
            candidates=[item.descriptor.candidate],
            descriptors=[item.descriptor],
            experiment_policy=ExperimentPolicy(
                dataset_name="general",
                repetition=0,
                case_ids=(item.case.id,),
            ),
            invoker=invoker,
            output_path=output_path,
            run_id=item.run_id,
            repository_commit=item.repository_commit,
        )

    serialized = output_path.read_text(encoding="utf-8")
    assert len(requests) == 1
    assert requests[0]["path"].endswith("/ai/run/@cf/google/gemma-4-26b-a4b-it")
    assert attempts[0].status == "failed"
    assert attempts[0].failure_category == "execution_error"
    assert attempts[0].attempt_id == item.attempt_id
    assert CF_TOKEN not in serialized
    assert GEMINI_KEY not in serialized
    assert CF_TOKEN not in caplog.text
    assert CF_TOKEN not in repr(invoker)
    assert CF_TOKEN not in json.dumps(asdict(attempts[0].invocation))


@pytest.mark.anyio
async def test_mocked_six_attempt_runner_integration_has_no_extra_requests(
    tmp_path: Path,
) -> None:
    schedule = _schedule()
    cloudflare_client, gemini_client, requests = _clients()
    output_path = tmp_path / "attempts.jsonl"
    attempts = []
    async with cloudflare_client, gemini_client:
        invoker = _invoker(schedule, cloudflare_client, gemini_client)
        for workload in ("general", "research"):
            workload_schedule = [
                item for item in schedule if item.descriptor.candidate.workload == workload
            ]
            dataset = workload_schedule[0].dataset
            attempts.extend(
                await run_evaluation(
                    dataset=dataset,
                    candidates=[item.descriptor.candidate for item in workload_schedule],
                    descriptors=[item.descriptor for item in workload_schedule],
                    experiment_policy=ExperimentPolicy(
                        dataset_name=workload,
                        repetition=0,
                        case_ids=(workload_schedule[0].case.id,),
                    ),
                    invoker=invoker,
                    output_path=output_path,
                    run_id=schedule[0].run_id,
                    repository_commit=schedule[0].repository_commit,
                )
            )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert len(requests) == len(attempts) == len(records) == 6
    assert len({record["attempt_id"] for record in records}) == 6
    assert {record["attempt_id"] for record in records} == {item.attempt_id for item in schedule}
    assert all(attempt.status == "succeeded" for attempt in attempts)
    assert all(
        attempt.observed_provider == attempt.requested_provider
        and attempt.observed_model == attempt.requested_model
        for attempt in attempts
    )
    assert all(attempt.repetition == 0 for attempt in attempts)
    general_prompts = [
        request["payload"]["messages"][1]["content"]
        if request["provider"] == "cloudflare"
        else request["payload"]["contents"][0]["parts"][0]["text"]
        for request in requests[:3]
    ]
    research_prompts = [
        request["payload"]["messages"][1]["content"]
        if request["provider"] == "cloudflare"
        else request["payload"]["contents"][0]["parts"][0]["text"]
        for request in requests[3:]
    ]
    assert len(set(general_prompts)) == 1
    assert all("Research Question:\n" in prompt for prompt in research_prompts)
    assert len(set(research_prompts)) == 1
    assert all(
        schedule[3].case.question in prompt and schedule[3].case.research_evidence() in prompt
        for prompt in research_prompts
    )
    assert all(request["method"] == "POST" for request in requests)
    assert all(
        "/ai/run/" in request["path"] or request["path"].endswith(":generateContent")
        for request in requests
    )
    assert not any(
        "models/search" in request["path"] or "models/schema" in request["path"]
        for request in requests
    )
    assert CF_TOKEN not in output_path.read_text(encoding="utf-8")
    assert GEMINI_KEY not in output_path.read_text(encoding="utf-8")


def test_stage1b_modules_have_no_orchestration_or_retrieval_imports() -> None:
    package_dir = Path(__file__).parents[1] / "evaluation/inference/stage1b"
    forbidden_roots = {"tavily", "redis", "orchestrator"}
    forbidden_prefixes = (
        "worker.application",
        "research_agent.application",
    )
    for path in package_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            for name in imported:
                root = name.split(".", maxsplit=1)[0]
                assert root not in forbidden_roots
                assert not name.startswith(forbidden_prefixes)
