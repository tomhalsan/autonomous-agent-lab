from research_agent.domain.models import SynthesisResult


class StubInferenceProvider:
    """Deterministic stub inference provider / synthesizer.

    Does not introduce an LLM, external API, or fabricated research claims.
    Produces deterministic and testable synthesis preserving grounded evidence.
    """

    def __init__(
        self,
        provider_name: str = "stub",
        model_name: str = "stub",
        failure_trigger: str | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.failure_trigger = failure_trigger

    async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
        if self.failure_trigger and self.failure_trigger in question:
            raise RuntimeError(f"Simulated synthesis failure for question: {question}")

        stripped_evidence = evidence.strip()
        if stripped_evidence:
            content = (
                f"Synthesized research for: {question}\n"
                f"Grounded evidence:\n{stripped_evidence}"
            )
        else:
            content = (
                f"Synthesized research for: {question}\n"
                "No grounded evidence available."
            )

        return SynthesisResult(
            question=question,
            content=content,
            evidence=evidence,
            provider=self.provider_name,
            model=self.model_name,
        )


StubSynthesizer = StubInferenceProvider

