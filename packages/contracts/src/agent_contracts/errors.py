"""Provider-neutral inference failure semantics (Issue #16 Increment 1)."""

from __future__ import annotations


class InferenceError(RuntimeError):
    """Base exception for inference provider execution failures.

    Derives from RuntimeError to maintain backward compatibility across existing
    application exception boundaries and test suites.
    """

    def __init__(
        self,
        message: str,
        provider_name: str | None = None,
        reason: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_name = provider_name
        self.reason = reason
        self.status_code = status_code


class FallbackEligibleInferenceError(InferenceError):
    """Inference failure where execution may proceed to another independent provider.

    Indicates that the failure is provider-specific or transient and does not preclude
    alternative providers from attempting the task. Does not imply or claim that the
    failed provider itself will recover.

    Examples:
    - Connection or network failure
    - Request or transport timeout
    - Rate limit (HTTP 429)
    - Server outage or internal error (HTTP 500-599)
    - Malformed or structurally unusable provider response when caller request was valid
    """


class NonFallbackInferenceError(InferenceError):
    """Inference failure where fallback to another provider must not proceed.

    Indicates that the failure is intrinsic to the prompt, client request, authentication,
    policy, or configuration, and falling back to another provider would be invalid or unsafe.

    Examples:
    - Explicit policy or content safety block
    - Client protocol or request validation error (HTTP 400)
    - Authentication or authorization failure (HTTP 401, HTTP 403)
    - Endpoint not found / wrong path (HTTP 404)
    - Other permanent client errors (HTTP 4xx except 429)
    """
