import json
from datetime import datetime
from typing import Any

from agent_contracts import (
    AgentCandidate,
    LivenessPolicy,
    evaluate_agent_record,
)
from agent_contracts.capabilities import DISALLOWED_METADATA_KEYS

from orchestrator.domain.capability import AgentCapability

from .ports import ResultReader, StateReader


class GetAgentStates:
    def __init__(self, reader: StateReader) -> None:
        self._reader = reader

    async def execute(self) -> list[dict[str, Any]]:
        return await self._reader.agents()


class GetRecentEvents:
    def __init__(self, reader: StateReader) -> None:
        self._reader = reader

    async def execute(self, limit: int = 50) -> list[dict[str, str]]:
        bounded_limit = min(max(limit, 1), 100)
        return await self._reader.events(limit=bounded_limit)


class GetTaskResult:
    def __init__(self, reader: ResultReader) -> None:
        self._reader = reader

    async def execute(self, task_id: str) -> dict[str, Any] | None:
        return await self._reader.get_result(task_id)


class FindCandidatesByCapability:
    """Query registered agents providing a required capability and eligible for task dispatch."""

    def __init__(
        self,
        reader: StateReader,
        policy: LivenessPolicy | None = None,
    ) -> None:
        self._reader = reader
        self._policy = policy or LivenessPolicy()

    async def execute(
        self,
        capability: AgentCapability | str,
        as_of: datetime | str | None = None,
    ) -> list[AgentCandidate]:
        if isinstance(capability, AgentCapability):
            target_cap = capability.value.lower()
        elif isinstance(capability, str):
            target_cap = capability.strip().lower()
            if not target_cap:
                raise ValueError("capability must not be empty")
        else:
            raise TypeError(
                f"capability must be an AgentCapability or str, got {type(capability).__name__}"
            )

        raw_agents = await self._reader.agents()
        candidates: list[AgentCandidate] = []

        for entry in raw_agents:
            if not isinstance(entry, dict):
                continue

            raw_agent_id = entry.get("agent_id")
            if not raw_agent_id or not str(raw_agent_id).strip():
                continue

            raw_capabilities = entry.get("capabilities", [])
            if isinstance(raw_capabilities, str):
                try:
                    raw_capabilities = json.loads(raw_capabilities)
                except (json.JSONDecodeError, TypeError):
                    raw_capabilities = [raw_capabilities]

            if not isinstance(raw_capabilities, list):
                continue

            normalized_capabilities: list[str] = [
                str(c).strip().lower()
                for c in raw_capabilities
                if isinstance(c, (str, AgentCapability)) and str(c).strip()
            ]

            # Step 1: Capability match check
            if target_cap not in normalized_capabilities:
                continue

            # Step 2: Availability evaluation using runtime liveness policy
            availability = evaluate_agent_record(
                entry,
                as_of=as_of,
                policy=self._policy,
            )

            # Step 3: Eligibility check (only READY agents with fresh heartbeats are eligible)
            if not availability.is_eligible:
                continue

            # Step 4: Extract and sanitize provider metadata
            raw_providers = entry.get("providers", {})
            if isinstance(raw_providers, str):
                try:
                    raw_providers = json.loads(raw_providers)
                except (json.JSONDecodeError, TypeError):
                    raw_providers = {}

            sanitized_providers: dict[str, str] = {}
            if isinstance(raw_providers, dict):
                for k, v in raw_providers.items():
                    k_str = str(k).strip()
                    k_lower = k_str.lower()
                    if any(disallowed in k_lower for disallowed in DISALLOWED_METADATA_KEYS):
                        continue
                    sanitized_providers[k_str] = str(v)

            raw_dispatch_target = entry.get("dispatch_target")

            candidates.append(
                AgentCandidate(
                    agent_id=str(raw_agent_id).strip(),
                    capabilities=normalized_capabilities,
                    agent_type=str(entry.get("agent_type", "agent")),
                    execution_state=availability.execution_state,
                    availability=availability,
                    providers=sanitized_providers,
                    dispatch_target=(
                        str(raw_dispatch_target).strip()
                        if raw_dispatch_target
                        else None
                    ),
                )
            )

        return candidates


QueryAgentsByCapability = FindCandidatesByCapability

