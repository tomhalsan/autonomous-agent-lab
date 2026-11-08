from agent_contracts import InferenceHealthStatus, InferenceProviderHealth

from research_agent.domain.models import SynthesisResult


class StubInferenceProvider:
    """Deterministic stub inference provider / synthesizer.

    Does not introduce an LLM, external API, or fabricated research claims.
    Produces deterministic and testable synthesis preserving grounded evidence.
    """

    def __init__(
        self,
        provider_name: str = "stub",
        model_name: str = "stub",
        failure_trigger: str | None = None,
        health_status: InferenceHealthStatus = InferenceHealthStatus.HEALTHY,
        health_reason: str = "",
        health_latency_ms: float | None = 1.0,
        health_retry_after_seconds: float | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.failure_trigger = failure_trigger
        self.health_status = health_status
        self.health_reason = health_reason
        self.health_latency_ms = health_latency_ms
        self.health_retry_after_seconds = health_retry_after_seconds

    def set_health(
        self,
        status: InferenceHealthStatus,
        reason: str = "",
        latency_ms: float | None = 1.0,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.health_status = status
        self.health_reason = reason
        self.health_latency_ms = latency_ms
        self.health_retry_after_seconds = retry_after_seconds

    async def check_health(self) -> InferenceProviderHealth:
        status = self.health_status
        reason = self.health_reason
        if (
            status
            in (
                InferenceHealthStatus.RATE_LIMITED,
                InferenceHealthStatus.QUOTA_EXHAUSTED,
                InferenceHealthStatus.UNAVAILABLE,
            )
            and not reason
        ):
            reason = f"Simulated {status.value} for stub provider"

        return InferenceProviderHealth(
            provider_name=self.provider_name,
            status=status,
            model=self.model_name,
            latency_ms=self.health_latency_ms,
            retry_after_seconds=self.health_retry_after_seconds,
            reason=reason,
        )

    async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
        if self.failure_trigger and self.failure_trigger in question:
            raise RuntimeError(f"Simulated synthesis failure for question: {question}")

        stripped_evidence = evidence.strip()
        if stripped_evidence:
            content = (
                f"Synthesized research for: {question}\n"
                f"Grounded evidence:\n{stripped_evidence}"
            )
        else:
            content = (
                f"Synthesized research for: {question}\n"
                "No grounded evidence available."
            )

        return SynthesisResult(
            question=question,
            content=content,
            evidence=evidence,
            provider=self.provider_name,
            model=self.model_name,
        )


StubSynthesizer = StubInferenceProvider

