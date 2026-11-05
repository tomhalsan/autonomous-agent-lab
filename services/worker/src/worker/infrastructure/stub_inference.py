class StubInferenceProvider:
    """Deterministic stub inference provider for general worker tasks."""

    def __init__(
        self,
        provider_name: str = "stub",
        model_name: str = "stub",
        failure_trigger: str | None = None,
        custom_response: str | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.failure_trigger = failure_trigger
        self.custom_response = custom_response

    async def generate(self, prompt: str) -> str:
        if self.failure_trigger and self.failure_trigger in prompt:
            raise RuntimeError(f"Simulated inference failure for prompt: {prompt}")

        if self.custom_response is not None:
            return self.custom_response

        return f"Generated response for: {prompt}"

