import os
from typing import Any

from worker.application.ports import RuntimeInferenceProvider
from worker.infrastructure.cloudflare_inference import CloudflareInferenceProvider
from worker.infrastructure.gemini_inference import GeminiInferenceProvider
from worker.infrastructure.stub_inference import StubInferenceProvider


def create_inference_provider(
    name: str | None = None,
    account_id: str | None = None,
    api_token: str | None = None,
    model: str | None = None,
    client: Any | None = None,
    max_tokens: int | None = None,
) -> RuntimeInferenceProvider:
    """Instantiate an InferenceProvider based on the provided name or INFERENCE_PROVIDER env var.

    Defaults to 'stub' if no name is provided and INFERENCE_PROVIDER is unset.
    Raises ValueError on unsupported provider names or missing configuration.
    """
    raw_name = name if name is not None else os.getenv("INFERENCE_PROVIDER", "stub")
    normalized = raw_name.strip().lower()

    if normalized == "stub":
        return StubInferenceProvider()

    if normalized == "cloudflare":
        resolved_account = (
            account_id
            if account_id is not None
            else os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        )
        if not resolved_account or not resolved_account.strip():
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID environment variable is required when INFERENCE_PROVIDER is 'cloudflare'"
            )

        resolved_token = (
            api_token
            if api_token is not None
            else os.getenv("CLOUDFLARE_AI_API_TOKEN", "")
        )
        if not resolved_token or not resolved_token.strip():
            raise ValueError(
                "CLOUDFLARE_AI_API_TOKEN environment variable is required when INFERENCE_PROVIDER is 'cloudflare'"
            )

        resolved_model = (
            model if model is not None else os.getenv("CLOUDFLARE_AI_MODEL", None)
        )

        return CloudflareInferenceProvider(
            account_id=resolved_account.strip(),
            api_token=resolved_token.strip(),
            model=resolved_model.strip() if resolved_model else None,
            client=client,
            max_tokens=max_tokens,
        )

    if normalized == "gemini":
        resolved_model = (
            model if model is not None else os.getenv("GEMINI_MODEL", None)
        )
        return GeminiInferenceProvider(
            api_key=api_token,
            model=resolved_model.strip() if resolved_model else None,
            client=client,
            max_tokens=max_tokens,
        )

    raise ValueError(
        f"Unsupported inference provider: {raw_name!r}. Supported providers: ['stub', 'cloudflare', 'gemini']"
    )

