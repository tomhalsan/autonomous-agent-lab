"""Frozen evaluation candidates; these are not production recommendations."""

from evaluation.inference.models import Candidate

CANDIDATES = (
    Candidate(
        id="general-cloudflare-gemma-4-26b-a4b-it",
        workload="general",
        provider="cloudflare",
        model="@cf/google/gemma-4-26b-a4b-it",
        adapter="cloudflare-workers-ai",
    ),
    Candidate(
        id="general-cloudflare-gpt-oss-20b",
        workload="general",
        provider="cloudflare",
        model="@cf/openai/gpt-oss-20b",
        adapter="cloudflare-workers-ai",
    ),
    Candidate(
        id="general-gemini-3.5-flash-lite",
        workload="general",
        provider="gemini",
        model="gemini-3.5-flash-lite",
        adapter="google-gemini",
    ),
    Candidate(
        id="research-cloudflare-qwen3.8-27b",
        workload="research",
        provider="cloudflare",
        model="@cf/qwen/qwen3.8-27b",
        adapter="cloudflare-workers-ai",
    ),
    Candidate(
        id="research-cloudflare-gpt-oss-20b",
        workload="research",
        provider="cloudflare",
        model="@cf/openai/gpt-oss-20b",
        adapter="cloudflare-workers-ai",
    ),
    Candidate(
        id="research-gemini-3.5-flash-lite",
        workload="research",
        provider="gemini",
        model="gemini-3.5-flash-lite",
        adapter="google-gemini",
    ),
)
