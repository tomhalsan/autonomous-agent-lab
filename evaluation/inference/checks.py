"""Deterministic output-contract checks that do not claim semantic entailment."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from evaluation.inference.models import CheckResult, EvaluationCase

_CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")
_BULLET_PATTERN = re.compile(r"^\s*[-*]\s+(.+?)\s*$", re.MULTILINE)
_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")


def _words(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text)


def _check_spec(spec: Mapping[str, object], text: str) -> bool:
    kind = spec.get("kind")
    lowered = text.casefold()

    if kind == "nonempty":
        return bool(text.strip())
    if kind == "contains_all":
        values = spec.get("values", [])
        return isinstance(values, list) and all(
            isinstance(value, str) and value.casefold() in lowered for value in values
        )
    if kind == "contains_any":
        values = spec.get("values", [])
        return isinstance(values, list) and any(
            isinstance(value, str) and value.casefold() in lowered for value in values
        )
    if kind == "forbids_any":
        values = spec.get("values", [])
        return isinstance(values, list) and not any(
            isinstance(value, str) and value.casefold() in lowered for value in values
        )
    if kind == "max_words":
        value = spec.get("value")
        return type(value) is int and len(_words(text)) <= value
    if kind == "bullet_count":
        value = spec.get("value")
        return type(value) is int and len(_BULLET_PATTERN.findall(text)) == value
    if kind == "bullet_word_limit":
        value = spec.get("value")
        bullets = _BULLET_PATTERN.findall(text)
        return (
            type(value) is int
            and bool(bullets)
            and all(len(_words(bullet)) <= value for bullet in bullets)
        )
    if kind == "bullet_citation":
        bullets = _BULLET_PATTERN.findall(text)
        return bool(bullets) and all(_CITATION_PATTERN.search(bullet) for bullet in bullets)
    if kind == "json_object":
        keys = spec.get("keys")
        if not isinstance(keys, dict):
            return False
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict) or set(parsed) != set(keys):
            return False
        return all(
            _matches_json_type(parsed[key], expected_type) for key, expected_type in keys.items()
        )
    if kind == "json_values":
        expected = spec.get("values")
        if not isinstance(expected, dict):
            return False
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, dict) and all(
            parsed.get(key) == value and type(parsed.get(key)) is type(value)
            for key, value in expected.items()
        )
    if kind == "required_citations":
        expected = spec.get("values")
        return isinstance(expected, list) and set(expected).issubset(
            set(_CITATION_PATTERN.findall(text))
        )
    if kind == "citation_allowlist":
        allowed = spec.get("values")
        return isinstance(allowed, list) and set(_CITATION_PATTERN.findall(text)).issubset(
            set(allowed)
        )
    raise ValueError(f"Unsupported deterministic check kind: {kind!r}")


def _matches_json_type(value: object, expected: object) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return type(value) in (int, float)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def evaluate_checks(case: EvaluationCase, text: str) -> tuple[CheckResult, ...]:
    """Run fixture-defined checks; lexical matches are not semantic proof."""
    results = [CheckResult(check_id="nonempty-output", passed=bool(text.strip()))]
    for spec in case.checks:
        check_id = spec.get("id")
        if not isinstance(check_id, str) or not check_id.strip():
            raise ValueError(f"Case {case.id!r} contains a check without a non-empty id")
        results.append(CheckResult(check_id=check_id, passed=_check_spec(spec, text)))
    return tuple(results)


def validate_check_specs(case: EvaluationCase) -> None:
    """Fail early for unsupported or malformed frozen fixture checks."""
    supported = {
        "nonempty",
        "contains_all",
        "contains_any",
        "forbids_any",
        "max_words",
        "bullet_count",
        "bullet_word_limit",
        "bullet_citation",
        "json_object",
        "json_values",
        "required_citations",
        "citation_allowlist",
    }
    for spec in case.checks:
        kind = spec.get("kind")
        if kind not in supported:
            raise ValueError(f"Case {case.id!r} has unsupported check kind {kind!r}")
        if not isinstance(spec.get("id"), str):
            raise TypeError(f"Case {case.id!r} has a check without a string id")
