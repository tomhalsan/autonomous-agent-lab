import os

from research_agent.application.ports import ResearchExecutor
from research_agent.domain.evaluator import EvidenceEvaluator
from research_agent.infrastructure.mcp_tavily_executor import McpTavilyResearchExecutor
from research_agent.infrastructure.stub_executor import StubResearchExecutor
from research_agent.infrastructure.tavily_executor import TavilyResearchExecutor


def create_research_executor(
    name: str | None = None,
    api_key: str | None = None,
    evaluator: EvidenceEvaluator | None = None,
) -> ResearchExecutor:
    """Instantiate a ResearchExecutor based on the provided name or RESEARCH_EXECUTOR env var.

    Defaults to 'stub' if no name is provided and RESEARCH_EXECUTOR is unset.
    Raises ValueError on unsupported executor names or missing configuration.
    """
    raw_name = name if name is not None else os.getenv("RESEARCH_EXECUTOR", "stub")
    normalized = raw_name.strip().lower()

    if normalized == "stub":
        return StubResearchExecutor()

    if normalized == "tavily":
        resolved_key = (
            api_key if api_key is not None else os.getenv("TAVILY_API_KEY", "")
        )
        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "TAVILY_API_KEY environment variable is required when RESEARCH_EXECUTOR is 'tavily'"
            )
        return TavilyResearchExecutor(
            api_key=resolved_key.strip(),
            evaluator=evaluator,
        )

    if normalized == "tavily_mcp":
        url = os.getenv("TAVILY_MCP_URL", "https://mcp.tavily.com/mcp/").strip()
        resolved_key = (
            api_key if api_key is not None else os.getenv("TAVILY_API_KEY", "")
        ).strip()
        headers = {"Authorization": f"Bearer {resolved_key}"} if resolved_key else None
        return McpTavilyResearchExecutor(
            url=url,
            headers=headers,
            evaluator=evaluator,
        )

    raise ValueError(
        f"Unsupported research executor: {raw_name!r}. Supported executors: ['stub', 'tavily', 'tavily_mcp']"
    )
