import ast
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict

import httpx
import pytest

from evaluation.inference.candidates import CANDIDATES
from evaluation.inference.diagnostics import (
    CLOUDFLARE_DIAGNOSTIC_CANDIDATES,
    DIAGNOSTIC_OUTPUT_LIMIT,
    DIAGNOSTIC_REPETITION,
    DIAGNOSTIC_RUN_ID,
    EXPECTED_CLOUDFLARE_MODELS,
    PreparedStage1BCFAttempt,
    Stage1BCFDiagnosticInvoker,
    prepare_stage1b_cf_diagnostic_schedule,
    resolve_stage1b_cf_diagnostic_metadata,
    run_stage1b_cf_diagnostic,
)
from evaluation.inference.models import ProviderModelMetadataSnapshot
from evaluation.inference.stage1a.metadata_resolvers import CloudflareMetadataResolver
from evaluation.inference.stage1b import prepare_stage1b_schedule

CF_TOKEN = "diagnostic-cf-secret-sentinel-xyz"
CF_ACCOUNT_ID = "diagnostic-cf-account-sentinel-123"

ORIGINAL_STAGE1B_ATTEMPTS_SHA256 = (
    "38b1de6ae690576887335760824a0e3a61d314d9a17c286b46f1f09a08363d93"
)
ORIGINAL_STAGE1B_BLINDED_SHA256 = (
    "3c7733c80b570ebdb94ebb5a98cbdadbb9d226ada62e10eb3eaa8099209b2b6f"
)

GENERAL_TEXT = (
    "The empty list causes IndexError because index 0 does not exist. "
    "Add `if not items: return None` before `items[0]`."
)
RESEARCH_TEXT = (
    "The evidence is insufficient to determine the outage's cause; "
    "it does not mention a battery defect [S1]."
)


class _CapturedRequest(TypedDict):
    method: str
    path: str
    query: str
    payload: Any


def _cf_metadata_snapshots() -> dict[tuple[str, str], ProviderModelMetadataSnapshot]:
    snapshots: dict[tuple[str, str], ProviderModelMetadataSnapshot] = {}
    for model in sorted(EXPECTED_CLOUDFLARE_MODELS):
        snapshots[("cloudflare", model)] = ProviderModelMetadataSnapshot(
            provider="cloudflare",
            model=model,
            retrieved_at="2026-11-13T10:00:00+00:00",
            metadata_source="account_catalog",
            metadata_method="account_catalog",
            supports_text_generation=True,
            default_output_tokens=256 if model == "@cf/openai/gpt-oss-20b" else None,
            temperature_supported=True,
            min_temperature=0.0,
            default_temperature=1.0,
            max_temperature=5.0 if model == "@cf/openai/gpt-oss-20b" else 2.0,
            reasoning_supported=True,
        )
    return snapshots


def _diagnostic_schedule() -> tuple[PreparedStage1BCFAttempt, ...]:
    return prepare_stage1b_cf_diagnostic_schedule(
        _cf_metadata_snapshots(),
        run_id=DIAGNOSTIC_RUN_ID,
        repository_commit="cc3b894",
    )


def _cf_result(
    text: str,
    *,
    usage: dict[str, int] | None = None,
    finish_reason: str = "stop",
) -> httpx.Response:
    result: dict[str, object] = {"response": text, "finish_reason": finish_reason}
    if usage is not None:
        result["usage"] = usage
    return httpx.Response(200, json={"success": True, "result": result})


def _mock_cf_client(
    response_fn: Callable[[_CapturedRequest], httpx.Response] | None = None,
) -> tuple[httpx.AsyncClient, list[_CapturedRequest]]:
    requests: list[_CapturedRequest] = []

    def handler(request: httpx.Request) -> httpx.Response:
        req_record: _CapturedRequest = {
            "method": request.method,
            "path": request.url.path,
            "query": request.url.query.decode("utf-8"),
            "payload": json.loads(request.content) if request.content else None,
        }
        requests.append(req_record)
        if response_fn is not None:
            return response_fn(req_record)

        if request.method == "POST" and "/ai/run/" in request.url.path:
            payload = req_record["payload"]
            messages = payload.get("messages", [])
            user_text = messages[1]["content"] if len(messages) > 1 else ""
            text = RESEARCH_TEXT if "Research Question:\n" in user_text else GENERAL_TEXT
            return _cf_result(
                text,
                usage={"prompt_tokens": 12, "completion_tokens": 14, "total_tokens": 26},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), requests


# --- Proof 1: Diagnostic schedule contains exactly four attempts ---
def test_diagnostic_schedule_contains_exact_four_attempts() -> None:
    schedule = _diagnostic_schedule()
    assert len(schedule) == 4
    assert [item.descriptor.candidate for item in schedule] == list(
        CLOUDFLARE_DIAGNOSTIC_CANDIDATES
    )
    assert [item.case.id for item in schedule] == [
        "general-debug-01",
        "general-debug-01",
        "research-insufficient-04",
        "research-insufficient-04",
    ]
    assert all(item.repetition == DIAGNOSTIC_REPETITION for item in schedule)
    assert all(item.run_id == DIAGNOSTIC_RUN_ID for item in schedule)
    assert all(item.descriptor.effective_output_limit == DIAGNOSTIC_OUTPUT_LIMIT for item in schedule)
    assert all(item.descriptor.effective_temperature_mode == "unset" for item in schedule)
    assert all(item.descriptor.effective_temperature is None for item in schedule)


# --- Proof 2: Only the three expected Cloudflare model identities are accepted ---
def test_only_three_expected_cloudflare_models_accepted() -> None:
    assert EXPECTED_CLOUDFLARE_MODELS == frozenset({
        "@cf/google/gemma-4-26b-a4b-it",
        "@cf/openai/gpt-oss-20b",
        "@cf/qwen/qwen3.8-27b",
    })
    snapshots = _cf_metadata_snapshots()

    # Missing a model
    missing = {k: v for k, v in snapshots.items() if k[1] != "@cf/openai/gpt-oss-20b"}
    with pytest.raises(ValueError, match="metadata snapshots must match"):
        prepare_stage1b_cf_diagnostic_schedule(missing, repository_commit="cc3b894")

    # Extra model
    extra = dict(snapshots)
    extra[("cloudflare", "@cf/meta/llama-3.1-8b-instruct")] = ProviderModelMetadataSnapshot(
        provider="cloudflare",
        model="@cf/meta/llama-3.1-8b-instruct",
        retrieved_at="2026-11-13T10:00:00+00:00",
        metadata_source="account_catalog",
        metadata_method="account_catalog",
        supports_text_generation=True,
    )
    with pytest.raises(ValueError, match="metadata snapshots must match"):
        prepare_stage1b_cf_diagnostic_schedule(extra, repository_commit="cc3b894")


# --- Proof 3: No Gemini metadata or inference dependency exists ---
def test_no_gemini_dependency_in_diagnostic() -> None:
    module_path = (
        Path(__file__).parent.parent / "evaluation" / "inference" / "diagnostics" / "stage1b_cf.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)

    assert not any("gemini" in name.lower() for name in imported_names)

    # Passing Gemini snapshot is rejected
    snapshots = dict(_cf_metadata_snapshots())
    snapshots[("gemini", "gemini-3.5-flash-lite")] = ProviderModelMetadataSnapshot(
        provider="gemini",
        model="gemini-3.5-flash-lite",
        retrieved_at="2026-11-13T10:00:00+00:00",
        metadata_source="provider_api",
        metadata_method="model_get",
        supports_text_generation=True,
    )
    with pytest.raises(ValueError, match="metadata snapshots must match"):
        prepare_stage1b_cf_diagnostic_schedule(snapshots, repository_commit="cc3b894")


# --- Proof 4 & 5: Prompts and evidence are identical between candidate pairs ---
def test_general_prompt_and_research_evidence_are_identical() -> None:
    schedule = _diagnostic_schedule()
    general_items = [i for i in schedule if i.descriptor.candidate.workload == "general"]
    research_items = [i for i in schedule if i.descriptor.candidate.workload == "research"]

    assert len(general_items) == 2
    assert len(research_items) == 2

    # General prompt identical
    assert general_items[0].case.general_prompt() == general_items[1].case.general_prompt()
    assert general_items[0].case.id == general_items[1].case.id == "general-debug-01"

    # Research question & evidence identical
    assert research_items[0].case.question == research_items[1].case.question
    assert (
        research_items[0].case.research_evidence()
        == research_items[1].case.research_evidence()
    )
    assert research_items[0].case.id == research_items[1].case.id == "research-insufficient-04"


# --- Proof 6, 7, 8, 9, 10: Mock four-call proof (POST, path, max_tokens, no temp) ---
@pytest.mark.anyio
async def test_mocked_four_call_diagnostic_execution(tmp_path: Path) -> None:
    schedule = _diagnostic_schedule()
    client, captured_requests = _mock_cf_client()
    output_path = tmp_path / "attempts.jsonl"

    async with client:
        invoker = Stage1BCFDiagnosticInvoker(
            schedule,
            cloudflare_account_id=CF_ACCOUNT_ID,
            cloudflare_api_token=CF_TOKEN,
            cloudflare_client=client,
        )
        attempts = await run_stage1b_cf_diagnostic(schedule, invoker, output_path=output_path)

    # Proof 8 & 9: exactly 4 POST requests
    assert len(captured_requests) == 4
    assert len(attempts) == 4
    assert all(req["method"] == "POST" for req in captured_requests)

    # Proof 10: exact model in /ai/run/... path
    expected_paths = [
        f"/{CF_ACCOUNT_ID}/ai/run/@cf/google/gemma-4-26b-a4b-it",
        f"/{CF_ACCOUNT_ID}/ai/run/@cf/openai/gpt-oss-20b",
        f"/{CF_ACCOUNT_ID}/ai/run/@cf/qwen/qwen3.8-27b",
        f"/{CF_ACCOUNT_ID}/ai/run/@cf/openai/gpt-oss-20b",
    ]
    assert [req["path"].replace("/client/v4/accounts", "") for req in captured_requests] == expected_paths

    # Proof 6: output limit 1024 reaches payload
    assert all(req["payload"]["max_tokens"] == 1024 for req in captured_requests)

    # Proof 7: temperature is omitted
    assert all("temperature" not in req["payload"] for req in captured_requests)

    # All succeeded with expected observed provider/model
    assert all(a.status == "succeeded" for a in attempts)
    assert [a.observed_model for a in attempts] == [
        "@cf/google/gemma-4-26b-a4b-it",
        "@cf/openai/gpt-oss-20b",
        "@cf/qwen/qwen3.8-27b",
        "@cf/openai/gpt-oss-20b",
    ]


# --- Proof 11: No retries ---
@pytest.mark.anyio
async def test_no_retries_on_failure(tmp_path: Path) -> None:
    schedule = _diagnostic_schedule()

    def error_response(req: _CapturedRequest) -> httpx.Response:
        return httpx.Response(500, json={"success": False, "errors": [{"message": "fail"}]})

    client, captured_requests = _mock_cf_client(response_fn=error_response)
    output_path = tmp_path / "attempts.jsonl"

    async with client:
        invoker = Stage1BCFDiagnosticInvoker(
            schedule,
            cloudflare_account_id=CF_ACCOUNT_ID,
            cloudflare_api_token=CF_TOKEN,
            cloudflare_client=client,
        )
        attempts = await run_stage1b_cf_diagnostic(schedule, invoker, output_path=output_path)

    assert len(captured_requests) == 4  # Exactly 1 request per attempt, no retries
    assert len(attempts) == 4
    assert all(a.status == "failed" for a in attempts)
    assert all(a.failure_category == "http_5xx" for a in attempts)


# --- Proof 12, 13, 14, 15: No fallback, Tavily, health probes, or orchestration ---
@pytest.mark.anyio
async def test_no_fallback_tavily_probes_or_orchestration(tmp_path: Path) -> None:
    schedule = _diagnostic_schedule()
    client, captured_requests = _mock_cf_client()
    output_path = tmp_path / "attempts.jsonl"

    async with client:
        invoker = Stage1BCFDiagnosticInvoker(
            schedule,
            cloudflare_account_id=CF_ACCOUNT_ID,
            cloudflare_api_token=CF_TOKEN,
            cloudflare_client=client,
        )
        await run_stage1b_cf_diagnostic(schedule, invoker, output_path=output_path)

    # Check request paths: none are health probes, Tavily, or orchestration
    for req in captured_requests:
        assert "/ai/run/" in req["path"]
        assert "probe" not in req["path"]
        assert "tavily" not in req["path"]
        assert "health" not in req["path"]


# --- Proof 16 & 17: Unique attempt IDs and no collision with original Stage 1B ---
def test_unique_attempt_ids_and_no_collision_with_original_stage1b() -> None:
    diag_schedule = _diagnostic_schedule()
    diag_ids = [item.attempt_id for item in diag_schedule]

    # Proof 16: four unique attempt IDs
    assert len(diag_ids) == 4
    assert len(set(diag_ids)) == 4

    # Build original Stage 1B schedule
    stage1b_snapshots: dict[tuple[str, str], ProviderModelMetadataSnapshot] = {}
    for candidate in CANDIDATES:
        key = (candidate.provider, candidate.model)
        if key in stage1b_snapshots:
            continue
        stage1b_snapshots[key] = ProviderModelMetadataSnapshot(
            provider=candidate.provider,
            model=candidate.model,
            retrieved_at="2026-11-13T08:00:00+00:00",
            metadata_source="account_catalog" if candidate.provider == "cloudflare" else "provider_api",
            metadata_method="account_catalog" if candidate.provider == "cloudflare" else "model_get",
            supports_text_generation=True,
            default_output_tokens=256 if candidate.model == "@cf/openai/gpt-oss-20b" else None,
            temperature_supported=True,
            min_temperature=0.0,
            default_temperature=1.0,
            max_temperature=5.0 if candidate.model == "@cf/openai/gpt-oss-20b" else 2.0,
            reasoning_supported=True,
        )
    orig_schedule = prepare_stage1b_schedule(
        stage1b_snapshots,
        run_id="stage1b-20261113-0817",
        repository_commit="98f8d9b",
    )
    orig_ids = [item.attempt_id for item in orig_schedule]

    # Proof 17: No collision
    assert set(diag_ids).isdisjoint(set(orig_ids))


# --- Proof 18: Safe failure classification preserved ---
@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "error_body", "expected_category"),
    [
        (500, {"success": False, "errors": [{"message": "Internal Server Error"}]}, "http_5xx"),
        (503, {"success": False, "errors": [{"message": "Service Unavailable"}]}, "http_5xx"),
        (400, {"success": False, "errors": [{"message": "Bad Request"}]}, "http_4xx"),
        (404, {"success": False, "errors": [{"message": "Not Found"}]}, "http_4xx"),
        (429, {"success": False, "errors": [{"message": "Rate Limited"}]}, "rate_limited"),
        (401, {"success": False, "errors": [{"message": "Unauthorized"}]}, "authentication_or_authorization"),
        (403, {"success": False, "errors": [{"message": "Forbidden"}]}, "authentication_or_authorization"),
    ],
)
async def test_safe_failure_classification(
    tmp_path: Path,
    status_code: int,
    error_body: dict[str, Any],
    expected_category: str,
) -> None:
    schedule = _diagnostic_schedule()

    def error_response(req: _CapturedRequest) -> httpx.Response:
        return httpx.Response(status_code, json=error_body)

    client, _ = _mock_cf_client(response_fn=error_response)
    output_path = tmp_path / "attempts.jsonl"

    async with client:
        invoker = Stage1BCFDiagnosticInvoker(
            schedule,
            cloudflare_account_id=CF_ACCOUNT_ID,
            cloudflare_api_token=CF_TOKEN,
            cloudflare_client=client,
        )
        attempts = await run_stage1b_cf_diagnostic(schedule, invoker, output_path=output_path)

    assert all(a.status == "failed" for a in attempts)
    assert all(a.failure_category == expected_category for a in attempts)


# --- Proof 19: Credential sentinels do not enter results, JSONL, logs, or repr ---
@pytest.mark.anyio
async def test_credentials_redacted_everywhere(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    schedule = _diagnostic_schedule()
    client, _ = _mock_cf_client()
    output_path = tmp_path / "attempts.jsonl"

    async with client:
        invoker = Stage1BCFDiagnosticInvoker(
            schedule,
            cloudflare_account_id=CF_ACCOUNT_ID,
            cloudflare_api_token=CF_TOKEN,
            cloudflare_client=client,
        )
        assert CF_TOKEN not in repr(invoker)
        assert CF_ACCOUNT_ID not in repr(invoker)

        attempts = await run_stage1b_cf_diagnostic(schedule, invoker, output_path=output_path)

    serialized = output_path.read_text(encoding="utf-8")
    assert CF_TOKEN not in serialized
    assert CF_TOKEN not in caplog.text
    for attempt in attempts:
        attempt_json = json.dumps(attempt.to_dict())
        assert CF_TOKEN not in attempt_json
        invocation_dict = asdict(attempt.invocation)
        assert CF_TOKEN not in json.dumps(invocation_dict)


# --- Proof 20: Original Stage 1B artifact hashes remain unchanged ---
def test_original_stage1b_artifact_hashes_unchanged() -> None:
    runs_dir = Path(__file__).parent.parent / "evaluation" / "inference" / "runs" / "stage1b-20261113-0817"
    attempts_path = runs_dir / "attempts.jsonl"
    blinded_path = runs_dir / "blinded-review.jsonl"

    assert attempts_path.exists()
    assert blinded_path.exists()

    attempts_sha = hashlib.sha256(attempts_path.read_bytes()).hexdigest()
    blinded_sha = hashlib.sha256(blinded_path.read_bytes()).hexdigest()

    assert attempts_sha == ORIGINAL_STAGE1B_ATTEMPTS_SHA256
    assert blinded_sha == ORIGINAL_STAGE1B_BLINDED_SHA256


# --- Metadata Resolution Budget Proof: exactly 6 GET requests (3 search + 3 schema) ---
@pytest.mark.anyio
async def test_metadata_resolution_budget_is_exact_six_gets() -> None:
    captured_gets: list[_CapturedRequest] = []

    def metadata_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        captured_gets.append({
            "method": request.method,
            "path": request.url.path,
            "query": request.url.query.decode("utf-8"),
            "payload": None,
        })
        model = request.url.params.get("search") or request.url.params.get("model")
        if "/ai/models/search" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "result": [
                        {
                            "name": model,
                            "task": {"name": "Text Generation"},
                            "properties": [{"property_id": "reasoning", "value": "true"}],
                        }
                    ],
                },
            )
        if "/ai/models/schema" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {
                        "input": {
                            "type": "object",
                            "properties": {
                                "messages": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                },
                                "max_tokens": {"type": "integer", "default": 256},
                                "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0, "default": 1.0},
                            },
                            "required": ["messages"],
                        }
                    },
                },
            )
        raise AssertionError(f"Unexpected metadata request: {request.url}")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(metadata_handler))
    resolver = CloudflareMetadataResolver(
        account_id=CF_ACCOUNT_ID,
        api_token=CF_TOKEN,
        client=mock_client,
    )

    snapshots = await resolve_stage1b_cf_diagnostic_metadata(resolver)

    # Exactly 3 unique models in snapshots
    assert len(snapshots) == 3
    assert set(snapshots.keys()) == {("cloudflare", m) for m in EXPECTED_CLOUDFLARE_MODELS}

    # Exactly 6 GET requests: 3 search + 3 schema
    assert len(captured_gets) == 6
    searches = [g for g in captured_gets if "/ai/models/search" in g["path"]]
    schemas = [g for g in captured_gets if "/ai/models/schema" in g["path"]]
    assert len(searches) == 3
    assert len(schemas) == 3

    # Schedule can be prepared with these snapshots
    schedule = prepare_stage1b_cf_diagnostic_schedule(snapshots, repository_commit="cc3b894")
    assert len(schedule) == 4
