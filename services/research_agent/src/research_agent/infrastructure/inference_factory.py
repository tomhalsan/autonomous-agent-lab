import os

from research_agent.application.ports import InferenceProvider
from research_agent.infrastructure.stub_inference import StubInferenceProvider


def create_inference_provider(
    name: str | None = None,
) -> InferenceProvider:
    """Instantiate an InferenceProvider based on the provided name or INFERENCE_PROVIDER env var.

    Defaults to 'stub' if no name is provided and INFERENCE_PROVIDER is unset.
    Raises ValueError on unsupported provider names.
    """
    raw_name = name if name is not None else os.getenv("INFERENCE_PROVIDER", "stub")
    normalized = raw_name.strip().lower()

    if normalized == "stub":
        return StubInferenceProvider()

    raise ValueError(
        f"Unsupported inference provider: {raw_name!r}. Supported providers: ['stub']"
    )

