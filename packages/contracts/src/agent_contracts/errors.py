"""Provider-neutral inference failure semantics (Issue #16 Increment 1 & 2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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


@dataclass(frozen=True)
class ProviderFailureSummary:
    """Sanitized metadata record of an individual provider attempt failure.

    Guarantees no raw Exception objects, tracebacks, response payloads, credentials,
    arbitrary messages, or request details are retained.
    """

    provider_name: str
    reason: str
    status_code: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        object.__setattr__(self, "provider_name", self.provider_name.strip().lower())

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        object.__setattr__(self, "reason", self.reason.strip())

        if self.status_code is not None:
            if type(self.status_code) is bool or not isinstance(self.status_code, int):
                raise TypeError(
                    f"status_code must be an int or None, got {type(self.status_code).__name__}"
                )
            if not (100 <= self.status_code <= 599):
                raise ValueError(
                    f"status_code must be a valid HTTP status (100-599), got {self.status_code}"
                )


class AllInferenceProvidersFailedError(InferenceError):
    """Raised when all candidate providers fail with fallback-eligible errors.

    The exception message is fixed, generic, and cannot be overridden by callers.
    Failure summaries are stored as an immutable tuple of sanitized ProviderFailureSummary objects.
    """

    MESSAGE = "All inference providers failed"

    def __init__(
        self,
        failures: Sequence[ProviderFailureSummary] = (),
    ) -> None:
        super().__init__(self.MESSAGE)
        normalized_failures: list[ProviderFailureSummary] = []
        for f in failures:
            if not isinstance(f, ProviderFailureSummary):
                raise TypeError(
                    f"failures must contain only ProviderFailureSummary items, got {type(f).__name__}"
                )
            normalized_failures.append(f)
        self.failures: tuple[ProviderFailureSummary, ...] = tuple(normalized_failures)
