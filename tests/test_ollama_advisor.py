"""Unit tests for the evaluation-only Ollama-backed RoutingAdvisor adapter."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from orchestrator.application.ports import RoutingAdvisor
from orchestrator.domain.advisor import (
    AdvisorExecutionError,
    AdvisorRecommendation,
    AdvisorTimeoutError,
    AdvisorUnavailableError,
)
from orchestrator.domain.assessment import (
    RoutingAssessment,
    RoutingClarity,
    RoutingSignal,
    RoutingSignalType,
    SignalStrength,
)
from orchestrator.domain.capability import AgentCapability

from evaluation.routing.ollama_advisor import OLLAMA_ROUTING_SCHEMA, OllamaRoutingAdvisor


def make_assessment(
    provisional_capability: AgentCapability = AgentCapability.GENERAL,
    clarity: RoutingClarity = RoutingClarity.AMBIGUOUS,
    reason: str = "Deterministic assessment flagged weak verb",
    signals: tuple[RoutingSignal, ...] | None = None,
) -> RoutingAssessment:
    """Create a sample RoutingAssessment with signals and reason."""
    if signals is None:
        signal = RoutingSignal(
            signal_type=RoutingSignalType.WEAK_RESEARCH_VERB,
            target_capability=AgentCapability.RESEARCH,
            strength=SignalStrength.WEAK,
            evidence="investigate",
        )
        signals = (signal,)
    return RoutingAssessment(
        provisional_capability=provisional_capability,
        clarity=clarity,
        signals=signals,
        reason=reason,
    )


def make_mock_client(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    text_content: str | None = None,
    capture_requests: list[httpx.Request] | None = None,
    side_effect: Exception | None = None,
) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with a deterministic MockTransport."""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture_requests is not None:
            capture_requests.append(request)
        if side_effect is not None:
            raise side_effect
        if json_data is not None:
            return httpx.Response(status_code=status_code, json=json_data)
        return httpx.Response(status_code=status_code, text=text_content or "")

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="http://localhost:11434")


@pytest.mark.anyio
async def test_1_valid_general_structured_response() -> None:
    async with make_mock_client(
        json_data={
            "response": json.dumps({
                "capability": "general",
                "reason": "Direct algorithm explanation requiring no live retrieval",
            })
        }
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)
        rec = await advisor.advise("Explain bubble sort", make_assessment())

        assert isinstance(rec, AdvisorRecommendation)
        assert rec.recommended_capability == AgentCapability.GENERAL
        assert rec.reason == "Direct algorithm explanation requiring no live retrieval"


@pytest.mark.anyio
async def test_2_valid_research_structured_response() -> None:
    async with make_mock_client(
        json_data={
            "response": json.dumps({
                "capability": "research",
                "reason": "Requires recent external market statistics and documentation",
            })
        }
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)
        rec = await advisor.advise("What are today's AWS spot prices?", make_assessment())

        assert isinstance(rec, AdvisorRecommendation)
        assert rec.recommended_capability == AgentCapability.RESEARCH
        assert rec.reason == "Requires recent external market statistics and documentation"


@pytest.mark.anyio
async def test_3_request_contains_stream_false() -> None:
    captured: list[httpx.Request] = []
    async with make_mock_client(
        json_data={"response": json.dumps({"capability": "general", "reason": "Reason"})},
        capture_requests=captured,
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)
        await advisor.advise("Test prompt", make_assessment())

        assert len(captured) == 1
        body = json.loads(captured[0].content)
        assert body["stream"] is False


@pytest.mark.anyio
async def test_4_request_contains_json_schema_format() -> None:
    captured: list[httpx.Request] = []
    async with make_mock_client(
        json_data={"response": json.dumps({"capability": "general", "reason": "Reason"})},
        capture_requests=captured,
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)
        await advisor.advise("Test prompt", make_assessment())

        assert len(captured) == 1
        body = json.loads(captured[0].content)
        assert body["format"] == OLLAMA_ROUTING_SCHEMA
        assert body["format"]["required"] == ["capability", "reason"]
        assert body["format"]["properties"]["capability"]["enum"] == ["general", "research"]
        assert body["format"]["additionalProperties"] is False


@pytest.mark.anyio
async def test_5_request_uses_deterministic_temperature_zero() -> None:
    captured: list[httpx.Request] = []
    async with make_mock_client(
        json_data={"response": json.dumps({"capability": "general", "reason": "Reason"})},
        capture_requests=captured,
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)
        await advisor.advise("Test prompt", make_assessment())

        assert len(captured) == 1
        body = json.loads(captured[0].content)
        assert "options" in body
        assert body["options"]["temperature"] == 0.0


@pytest.mark.anyio
async def test_6_prompt_does_not_contain_provisional_reason_or_signals() -> None:
    captured: list[httpx.Request] = []
    async with make_mock_client(
        json_data={"response": json.dumps({"capability": "general", "reason": "Reason"})},
        capture_requests=captured,
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        sentinel_reason = "SENTINEL_DETERMINISTIC_REASON_XYZ_98765"
        sentinel_evidence = "SENTINEL_SIGNAL_EVIDENCE_ABC_12345"
        sentinel_prompt = "User query prompt for unique testing"

        signal = RoutingSignal(
            signal_type=RoutingSignalType.WEAK_RESEARCH_VERB,
            target_capability=AgentCapability.RESEARCH,
            strength=SignalStrength.WEAK,
            evidence=sentinel_evidence,
        )
        assessment = make_assessment(
            provisional_capability=AgentCapability.RESEARCH,
            reason=sentinel_reason,
            signals=(signal,),
        )
        await advisor.advise(sentinel_prompt, assessment)

        assert len(captured) == 1
        body = json.loads(captured[0].content)
        sent_prompt = body["prompt"]

        assert sentinel_prompt in sent_prompt
        assert sentinel_reason not in sent_prompt
        assert sentinel_evidence not in sent_prompt
        assert "weak_research_verb" not in sent_prompt
        assert "provisional_capability" not in sent_prompt
        assert "expected_capability" not in sent_prompt
        assert "ground_truth" not in sent_prompt


@pytest.mark.anyio
async def test_7_markdown_fenced_model_json_is_rejected() -> None:
    fenced_output = '```json\n{"capability": "general", "reason": "Valid reason"}\n```'
    async with make_mock_client(json_data={"response": fenced_output}) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        with pytest.raises(AdvisorExecutionError, match="markdown code fences"):
            await advisor.advise("Test prompt", make_assessment())


@pytest.mark.anyio
async def test_8_conversational_prose_wrapping_json_is_rejected() -> None:
    prose_output = 'Here is the result: {"capability": "general", "reason": "Valid reason"}'
    async with make_mock_client(json_data={"response": prose_output}) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        with pytest.raises(AdvisorExecutionError, match="not valid JSON"):
            await advisor.advise("Test prompt", make_assessment())


@pytest.mark.anyio
async def test_9_malformed_json_is_rejected_without_exception_chaining() -> None:
    secret_marker = "SENSITIVE_SYNTAX_ERROR_BODY_XYZ_111"
    async with make_mock_client(
        json_data={"response": f'{{"capability": "general", "reason": {secret_marker}'}
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        with pytest.raises(AdvisorExecutionError) as exc_info:
            await advisor.advise("Test prompt", make_assessment())

        assert "not valid JSON" in str(exc_info.value)
        assert secret_marker not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True


@pytest.mark.anyio
async def test_10_missing_capability_is_rejected() -> None:
    async with make_mock_client(json_data={"response": '{"reason": "Only a reason"}'}) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        with pytest.raises(AdvisorExecutionError, match="missing required 'capability' key"):
            await advisor.advise("Test prompt", make_assessment())


@pytest.mark.anyio
async def test_11_missing_reason_is_rejected() -> None:
    async with make_mock_client(json_data={"response": '{"capability": "general"}'}) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        with pytest.raises(AdvisorExecutionError, match="missing required 'reason' key"):
            await advisor.advise("Test prompt", make_assessment())


@pytest.mark.anyio
async def test_12_unsupported_capability_is_rejected() -> None:
    async with make_mock_client(
        json_data={"response": '{"capability": "coding", "reason": "Valid reason"}'}
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        with pytest.raises(AdvisorExecutionError, match="Unsupported capability value"):
            await advisor.advise("Test prompt", make_assessment())


@pytest.mark.anyio
async def test_13_empty_reason_is_rejected() -> None:
    async with make_mock_client(
        json_data={"response": '{"capability": "general", "reason": "   "}'}
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        with pytest.raises(AdvisorExecutionError, match="must be a non-empty string"):
            await advisor.advise("Test prompt", make_assessment())


@pytest.mark.anyio
async def test_14_connection_failure_maps_to_advisor_unavailable() -> None:
    req = httpx.Request("POST", "http://localhost:11434/api/generate")
    async with make_mock_client(
        side_effect=httpx.ConnectError("Connection refused", request=req)
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        with pytest.raises(AdvisorUnavailableError, match="unavailable or connection failed"):
            await advisor.advise("Test prompt", make_assessment())


@pytest.mark.anyio
async def test_15_timeout_maps_to_advisor_timeout() -> None:
    req = httpx.Request("POST", "http://localhost:11434/api/generate")
    async with make_mock_client(
        side_effect=httpx.ReadTimeout("Read deadline exceeded", request=req)
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        with pytest.raises(AdvisorTimeoutError, match="request timed out"):
            await advisor.advise("Test prompt", make_assessment())


@pytest.mark.anyio
async def test_16_non_success_http_response_mapping() -> None:
    # 503 Service Unavailable -> AdvisorUnavailableError
    async with make_mock_client(status_code=503, text_content="Service Unavailable") as client_503:
        advisor_503 = OllamaRoutingAdvisor(model="test-model", client=client_503)
        with pytest.raises(AdvisorUnavailableError, match="HTTP status 503"):
            await advisor_503.advise("Test prompt", make_assessment())

    # 404 Model Not Found -> AdvisorExecutionError
    async with make_mock_client(
        status_code=404, text_content="model 'test' not found"
    ) as client_404:
        advisor_404 = OllamaRoutingAdvisor(model="test-model", client=client_404)
        with pytest.raises(AdvisorExecutionError, match="HTTP status 404"):
            await advisor_404.advise("Test prompt", make_assessment())

    # 500 Internal Error -> AdvisorExecutionError
    async with make_mock_client(
        status_code=500, text_content="Ollama internal error"
    ) as client_500:
        advisor_500 = OllamaRoutingAdvisor(model="test-model", client=client_500)
        with pytest.raises(AdvisorExecutionError, match="HTTP status 500"):
            await advisor_500.advise("Test prompt", make_assessment())


@pytest.mark.anyio
async def test_17_arbitrary_raw_model_output_is_not_leaked_in_exceptions() -> None:
    secret_marker = "SENSITIVE_LEAKED_PAYLOAD_99999"
    async with make_mock_client(
        json_data={
            "response": f'{{"capability": "unsupported_capability_{secret_marker}", "reason": "Reason {secret_marker}"}}'
        }
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        with pytest.raises(AdvisorExecutionError) as exc_info:
            await advisor.advise("Test prompt", make_assessment())

        assert secret_marker not in str(exc_info.value)
        assert exc_info.value.__cause__ is None


def test_18_adapter_satisfies_routing_advisor_protocol() -> None:
    advisor = OllamaRoutingAdvisor(model="test-model")
    assert isinstance(advisor, RoutingAdvisor)


@pytest.mark.anyio
async def test_19_exact_client_side_response_keys_enforced() -> None:
    """Extra keys (e.g., dispatch_to, confidence, agent) are rejected without echoing them."""
    extra_field_payloads = [
        {"capability": "general", "reason": "Valid reason", "dispatch_to": "agent.worker"},
        {"capability": "general", "reason": "Valid reason", "confidence": 0.95},
        {"capability": "general", "reason": "Valid reason", "agent": "worker-1"},
        {"capability": "general", "reason": "Valid reason", "command": "execute"},
        {"capability": "general", "reason": "Valid reason", "extra_sensitive_field_secret": "xyz"},
    ]

    for payload in extra_field_payloads:
        async with make_mock_client(json_data={"response": json.dumps(payload)}) as client:
            advisor = OllamaRoutingAdvisor(model="test-model", client=client)

            with pytest.raises(AdvisorExecutionError) as exc_info:
                await advisor.advise("Test prompt", make_assessment())

            err_msg = str(exc_info.value)
            assert "unexpected fields" in err_msg
            assert "extra_sensitive_field_secret" not in err_msg
            assert "dispatch_to" not in err_msg
            assert "confidence" not in err_msg


@pytest.mark.anyio
async def test_20_malformed_outer_provider_json_suppresses_exception_chaining() -> None:
    """Outer provider JSON decode errors do not retain raw payload in cause or context."""
    secret_outer_marker = "SECRET_MALFORMED_OUTER_PAYLOAD_TOKEN_777"
    async with make_mock_client(text_content=f"Not json: {secret_outer_marker}") as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)

        with pytest.raises(AdvisorExecutionError) as exc_info:
            await advisor.advise("Test prompt", make_assessment())

        assert "Malformed JSON in Ollama outer response" in str(exc_info.value)
        assert secret_outer_marker not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True


def test_21_constructor_configuration_validation() -> None:
    """Eager validation of model, base_url, timeout, and keep_alive."""
    # Valid configurations pass cleanly
    advisor = OllamaRoutingAdvisor(model="gemma3:1b")
    assert advisor.model == "gemma3:1b"
    assert advisor.base_url == "http://localhost:11434"
    assert advisor.timeout == 30.0
    assert advisor.keep_alive is None

    advisor_custom = OllamaRoutingAdvisor(
        model="llama3.2:3b",
        base_url="https://ollama.local:8443/",
        timeout=15.5,
        keep_alive="5m",
    )
    assert advisor_custom.model == "llama3.2:3b"
    assert advisor_custom.base_url == "https://ollama.local:8443"
    assert advisor_custom.timeout == 15.5
    assert advisor_custom.keep_alive == "5m"

    # Invalid models
    for invalid_model in ["", "   ", 123, None]:
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            OllamaRoutingAdvisor(model=invalid_model)  # type: ignore[arg-type]

    # Invalid base_url
    for invalid_url in ["", "   ", 123, None]:
        with pytest.raises(ValueError, match="base_url must be a non-empty string"):
            OllamaRoutingAdvisor(model="m", base_url=invalid_url)  # type: ignore[arg-type]

    # Unsupported URL scheme
    for bad_scheme_url in ["ftp://localhost:11434", "file:///path", "ws://localhost:11434", "localhost:11434"]:
        with pytest.raises(ValueError, match="base_url scheme must be 'http' or 'https'"):
            OllamaRoutingAdvisor(model="m", base_url=bad_scheme_url)

    # Missing host in base_url
    for no_host_url in ["http://", "https://", "http:///path"]:
        with pytest.raises(ValueError, match="base_url must contain a valid host"):
            OllamaRoutingAdvisor(model="m", base_url=no_host_url)

    # Invalid timeout values
    for bad_timeout in [0, -5, -0.1, True, False, "10", float("inf"), float("-inf"), float("nan")]:
        with pytest.raises(ValueError):
            OllamaRoutingAdvisor(model="m", timeout=bad_timeout)  # type: ignore[arg-type]

    # Invalid keep_alive values
    for bad_keep_alive in [-1, -100, "", "   ", 1.5, True, False]:
        with pytest.raises(ValueError):
            OllamaRoutingAdvisor(model="m", keep_alive=bad_keep_alive)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_22_injected_client_is_not_closed_by_advisor() -> None:
    """An injected httpx.AsyncClient remains open across advise calls (caller-owned)."""
    async with make_mock_client(
        json_data={"response": json.dumps({"capability": "general", "reason": "Reason"})}
    ) as client:
        advisor = OllamaRoutingAdvisor(model="test-model", client=client)
        assert not client.is_closed

        rec = await advisor.advise("Test prompt 1", make_assessment())
        assert rec.recommended_capability == AgentCapability.GENERAL
        assert not client.is_closed

        rec2 = await advisor.advise("Test prompt 2", make_assessment())
        assert rec2.recommended_capability == AgentCapability.GENERAL
        assert not client.is_closed
