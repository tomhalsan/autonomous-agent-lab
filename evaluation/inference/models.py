"""Provider-neutral data structures for inference evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal, cast

Workload = Literal["general", "research"]
SettingMode = Literal["provider_default", "unset", "explicit"]
MetadataSource = Literal["provider_api", "account_catalog", "offline_fixture"]
MetadataMethod = Literal["model_get", "model_list", "account_catalog", "fixture"]

_METADATA_SOURCES = ("provider_api", "account_catalog", "offline_fixture")
_METADATA_METHODS = ("model_get", "model_list", "account_catalog", "fixture")
_SETTING_MODES = ("provider_default", "unset", "explicit")


@dataclass(frozen=True)
class SourceExcerpt:
    id: str
    url: str
    excerpt: str


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    workload: Workload
    category: str
    instruction: str
    checks: tuple[dict[str, object], ...]
    human_rubric_items: tuple[str, ...]
    input: str | None = None
    question: str | None = None
    sources: tuple[SourceExcerpt, ...] = ()

    def research_evidence(self) -> str:
        """Render the fixed evidence fixture with stable source identifiers."""
        return "\n\n".join(
            f"[{source.id}] {source.url}\n{source.excerpt}" for source in self.sources
        )

    def general_prompt(self) -> str:
        if self.workload != "general" or self.input is None:
            raise ValueError(f"Case {self.id!r} is not a General case")
        return f"{self.instruction}\n\nInput:\n{self.input}"


@dataclass(frozen=True)
class EvaluationDataset:
    name: Workload
    version: str
    cases: tuple[EvaluationCase, ...]
    sha256: str

    def case(self, case_id: str) -> EvaluationCase:
        for item in self.cases:
            if item.id == case_id:
                return item
        raise KeyError(f"Unknown {self.name} evaluation case: {case_id}")


@dataclass(frozen=True)
class Candidate:
    id: str
    workload: Workload
    provider: str
    model: str
    adapter: str

    def __post_init__(self) -> None:
        if self.workload not in ("general", "research"):
            raise ValueError(f"Unsupported workload: {self.workload!r}")
        for field_name in ("id", "provider", "model", "adapter"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(
                self,
                field_name,
                value.strip().lower() if field_name == "provider" else value.strip(),
            )


@dataclass(frozen=True)
class ProviderModelMetadataSnapshot:
    """Allowlisted provider facts normalized from metadata or an offline fixture."""

    provider: str | None
    model: str | None
    retrieved_at: str
    metadata_source: MetadataSource
    metadata_method: MetadataMethod
    supports_text_generation: bool | None = None
    context_limit_tokens: int | None = None
    input_limit_tokens: int | None = None
    max_output_tokens: int | None = None
    default_output_tokens: int | None = None
    temperature_supported: bool | None = None
    min_temperature: float | None = None
    default_temperature: float | None = None
    max_temperature: float | None = None
    reasoning_supported: bool | None = None
    reasoning_default: bool | None = None

    def __post_init__(self) -> None:
        for field_name in ("provider", "model"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string or None")
            if value is not None:
                normalized = value.strip().lower() if field_name == "provider" else value.strip()
                object.__setattr__(self, field_name, normalized)
        if self.metadata_source not in _METADATA_SOURCES:
            raise ValueError(f"Unsupported metadata_source: {self.metadata_source!r}")
        if self.metadata_method not in _METADATA_METHODS:
            raise ValueError(f"Unsupported metadata_method: {self.metadata_method!r}")
        if not isinstance(self.retrieved_at, str) or not self.retrieved_at.strip():
            raise ValueError("retrieved_at must be a timezone-aware ISO timestamp")
        try:
            retrieved_at = datetime.fromisoformat(self.retrieved_at)
        except ValueError as error:
            raise ValueError("retrieved_at must be a timezone-aware ISO timestamp") from error
        if retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be a timezone-aware ISO timestamp")
        for field_name in (
            "supports_text_generation",
            "temperature_supported",
            "reasoning_supported",
            "reasoning_default",
        ):
            value = getattr(self, field_name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{field_name} must be a boolean or None")
        for field_name in (
            "context_limit_tokens",
            "input_limit_tokens",
            "max_output_tokens",
            "default_output_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or None")
        if (
            self.default_output_tokens is not None
            and self.max_output_tokens is not None
            and self.default_output_tokens > self.max_output_tokens
        ):
            raise ValueError("default_output_tokens cannot exceed max_output_tokens")
        for field_name in ("min_temperature", "default_temperature", "max_temperature"):
            value = getattr(self, field_name)
            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{field_name} must be a finite non-negative number or None")
                object.__setattr__(self, field_name, float(value))
        if (
            self.min_temperature is not None
            and self.max_temperature is not None
            and self.min_temperature > self.max_temperature
        ):
            raise ValueError("min_temperature cannot exceed max_temperature")
        for temperature in (self.min_temperature, self.default_temperature):
            if (
                temperature is not None
                and self.max_temperature is not None
                and temperature > self.max_temperature
            ):
                raise ValueError("temperature facts cannot exceed max_temperature")
        if (
            self.default_temperature is not None
            and self.min_temperature is not None
            and self.default_temperature < self.min_temperature
        ):
            raise ValueError("default_temperature cannot be below min_temperature")


@dataclass(frozen=True)
class ExperimentPolicy:
    """Run-level experiment selection; contains no provider/model facts."""

    dataset_name: Workload
    repetition: int
    case_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.dataset_name not in ("general", "research"):
            raise ValueError(f"Unsupported dataset_name: {self.dataset_name!r}")
        if type(self.repetition) is not int or self.repetition < 0:
            raise ValueError("repetition must be a non-negative integer")
        if self.case_ids is not None:
            if (
                not isinstance(self.case_ids, tuple)
                or not self.case_ids
                or any(not isinstance(item, str) or not item.strip() for item in self.case_ids)
            ):
                raise ValueError("case_ids must contain non-empty case identifiers")
            if len(set(self.case_ids)) != len(self.case_ids):
                raise ValueError("case_ids must be unique")


@dataclass(frozen=True)
class InvocationPolicy:
    """Explicit, candidate-specific experiment choices, separate from model facts."""

    output_limit_mode: SettingMode = "unset"
    output_limit: int | None = None
    temperature_mode: SettingMode = "unset"
    temperature: float | None = None

    def __post_init__(self) -> None:
        _validate_setting(self.output_limit_mode, self.output_limit, "output_limit")
        _validate_setting(self.temperature_mode, self.temperature, "temperature")
        if self.output_limit_mode == "explicit" and (
            type(self.output_limit) is not int or self.output_limit <= 0
        ):
            raise ValueError("an explicit output_limit must be a positive integer")
        if self.temperature_mode == "explicit" and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("an explicit temperature must be finite and non-negative")
        if self.temperature is not None:
            object.__setattr__(self, "temperature", float(self.temperature))


def _validate_setting(mode: SettingMode, value: object, name: str) -> None:
    if not isinstance(mode, str) or mode not in _SETTING_MODES:
        raise ValueError(f"Unsupported {name}_mode: {mode!r}")
    if mode == "explicit" and value is None:
        raise ValueError(f"{name} is required when mode is explicit")
    if mode != "explicit" and value is not None:
        raise ValueError(f"{name} must be None unless mode is explicit")


@dataclass(frozen=True)
class EffectiveInvocationDescriptor:
    """Candidate, metadata provenance, policy overrides, and effective settings."""

    candidate: Candidate
    metadata: ProviderModelMetadataSnapshot
    policy: InvocationPolicy
    effective_output_limit_mode: SettingMode = field(init=False)
    effective_output_limit: int | None = field(init=False)
    effective_temperature_mode: SettingMode = field(init=False)
    effective_temperature: float | None = field(init=False)

    def __post_init__(self) -> None:
        if self.metadata.provider is not None and self.metadata.provider != self.candidate.provider:
            raise ValueError("metadata provider does not match candidate provider")
        if self.metadata.model is not None and self.metadata.model != self.candidate.model:
            raise ValueError("metadata model does not match candidate model")
        if self.metadata.supports_text_generation is False:
            raise ValueError("metadata reports no text-generation capability")
        if self.policy.output_limit_mode == "explicit":
            assert self.policy.output_limit is not None
            if (
                self.metadata.max_output_tokens is not None
                and self.policy.output_limit > self.metadata.max_output_tokens
            ):
                raise ValueError("explicit output_limit exceeds metadata max_output_tokens")
        if self.policy.temperature_mode == "explicit":
            assert self.policy.temperature is not None
            if self.metadata.temperature_supported is False:
                raise ValueError("metadata reports temperature is unsupported")
            if (
                self.metadata.min_temperature is not None
                and self.policy.temperature < self.metadata.min_temperature
            ):
                raise ValueError("explicit temperature is below metadata min_temperature")
            if (
                self.metadata.max_temperature is not None
                and self.policy.temperature > self.metadata.max_temperature
            ):
                raise ValueError("explicit temperature exceeds metadata max_temperature")

        output_limit = self.policy.output_limit
        if self.policy.output_limit_mode == "provider_default":
            output_limit = self.metadata.default_output_tokens
        elif self.policy.output_limit_mode == "unset":
            output_limit = None
        temperature = self.policy.temperature
        if self.policy.temperature_mode == "provider_default":
            temperature = self.metadata.default_temperature
        elif self.policy.temperature_mode == "unset":
            temperature = None
        object.__setattr__(self, "effective_output_limit_mode", self.policy.output_limit_mode)
        object.__setattr__(self, "effective_output_limit", output_limit)
        object.__setattr__(self, "effective_temperature_mode", self.policy.temperature_mode)
        object.__setattr__(self, "effective_temperature", temperature)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> EffectiveInvocationDescriptor:
        candidate_value = value.get("candidate")
        metadata_value = value.get("metadata")
        policy_value = value.get("policy")
        if not all(
            isinstance(item, dict) for item in (candidate_value, metadata_value, policy_value)
        ):
            raise TypeError(
                "invocation descriptor requires candidate, metadata, and policy objects"
            )
        descriptor = cls(
            candidate=Candidate(**cast(dict[str, object], candidate_value)),  # type: ignore[arg-type]
            metadata=ProviderModelMetadataSnapshot(**cast(dict[str, object], metadata_value)),  # type: ignore[arg-type]
            policy=InvocationPolicy(**cast(dict[str, object], policy_value)),  # type: ignore[arg-type]
        )
        for field_name in (
            "effective_output_limit_mode",
            "effective_output_limit",
            "effective_temperature_mode",
            "effective_temperature",
        ):
            if value.get(field_name) != getattr(descriptor, field_name):
                raise ValueError(f"Persisted {field_name} does not match descriptor inputs")
        return descriptor


@dataclass(frozen=True)
class ProviderOutput:
    """Normalized output returned by an injected evaluation adapter."""

    text: str
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    truncated: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        for field_name in ("provider", "model", "finish_reason"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string or None")
            if value is not None:
                object.__setattr__(self, field_name, value.strip())
        for field_name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or None")
        if self.truncated is not None and type(self.truncated) is not bool:
            raise TypeError("truncated must be a bool or None")


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    passed: bool


@dataclass(frozen=True)
class EvaluationAttempt:
    schema_version: int
    run_id: str
    attempt_id: str
    repository_commit: str
    dataset_name: str
    dataset_version: str
    dataset_sha256: str
    case_id: str
    repetition: int
    invocation: EffectiveInvocationDescriptor
    timestamp: str
    status: Literal["succeeded", "failed"]
    latency_ms: float
    observed_provider: str | None = None
    observed_model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    truncated: bool | None = None
    failure_category: str | None = None
    output: str | None = None
    checks: tuple[CheckResult, ...] = ()

    def __post_init__(self) -> None:
        if type(self.repetition) is not int or self.repetition < 0:
            raise ValueError("repetition must be a non-negative integer")
        if self.dataset_name != self.invocation.candidate.workload:
            raise ValueError("attempt dataset_name must match invocation candidate workload")

    @property
    def workload(self) -> Workload:
        return self.invocation.candidate.workload

    @property
    def candidate_id(self) -> str:
        return self.invocation.candidate.id

    @property
    def requested_provider(self) -> str:
        return self.invocation.candidate.provider

    @property
    def requested_model(self) -> str:
        return self.invocation.candidate.model

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def attempt_from_dict(value: dict[str, object]) -> EvaluationAttempt:
    raw_invocation = value.get("invocation")
    if not isinstance(raw_invocation, dict):
        raise TypeError("invocation must be an object")
    raw_checks = value.get("checks", [])
    if not isinstance(raw_checks, list):
        raise TypeError("checks must be a list")
    checks: list[CheckResult] = []
    for raw_check in raw_checks:
        if not isinstance(raw_check, dict):
            raise TypeError("each check must be an object")
        check_id = raw_check.get("check_id")
        passed = raw_check.get("passed")
        if not isinstance(check_id, str) or not check_id.strip():
            raise TypeError("check_id must be a non-empty string")
        if type(passed) is not bool:
            raise TypeError("passed must be a boolean")
        checks.append(CheckResult(check_id=check_id, passed=passed))
    fields = dict(value)
    fields["invocation"] = EffectiveInvocationDescriptor.from_dict(raw_invocation)
    fields["checks"] = tuple(checks)
    return EvaluationAttempt(**fields)  # type: ignore[arg-type]
