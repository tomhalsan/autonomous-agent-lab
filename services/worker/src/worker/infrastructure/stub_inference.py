from agent_contracts import InferenceHealthStatus, InferenceProviderHealth


class StubInferenceProvider:
    """Deterministic stub inference provider for general worker tasks."""

    def __init__(
        self,
        provider_name: str = "stub",
        model_name: str = "stub",
        failure_trigger: str | None = None,
        custom_response: str | None = None,
        health_status: InferenceHealthStatus = InferenceHealthStatus.HEALTHY,
        health_reason: str = "",
        health_latency_ms: float | None = 1.0,
        health_retry_after_seconds: float | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.failure_trigger = failure_trigger
        self.custom_response = custom_response
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

    async def generate(self, prompt: str) -> str:
        if self.failure_trigger and self.failure_trigger in prompt:
            raise RuntimeError(f"Simulated inference failure for prompt: {prompt}")

        if self.custom_response is not None:
            return self.custom_response

        return f"Generated response for: {prompt}"

