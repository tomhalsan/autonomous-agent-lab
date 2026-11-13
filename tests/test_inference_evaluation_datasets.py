from __future__ import annotations

from evaluation.inference.candidates import CANDIDATES
from evaluation.inference.datasets import load_dataset


def test_frozen_datasets_have_stable_schema_versions_and_hashes() -> None:
    general = load_dataset("general")
    research = load_dataset("research")

    assert general.name == "general"
    assert general.version == "1.0.1"
    assert len(general.cases) == 6
    assert general.sha256 == "cbeaee8c1b647b8518604a39c158384f72ccaf43264e4b90250912675df960e2"

    assert research.name == "research"
    assert research.version == "1.0.1"
    assert len(research.cases) == 6
    assert research.sha256 == "4097c8d994068e4a74d3897dd872fc7eef5ef63ae156010abbccd441149b1c14"
    assert all(case.sources for case in research.cases)
    assert all(
        source.url.startswith("https://sources.example/")
        for case in research.cases
        for source in case.sources
    )


def test_candidate_matrix_is_frozen_and_workload_specific() -> None:
    assert [
        (candidate.workload, candidate.provider, candidate.model) for candidate in CANDIDATES
    ] == [
        ("general", "cloudflare", "@cf/google/gemma-4-26b-a4b-it"),
        ("general", "cloudflare", "@cf/openai/gpt-oss-20b"),
        ("general", "gemini", "gemini-3.5-flash-lite"),
        ("research", "cloudflare", "@cf/qwen/qwen3.8-27b"),
        ("research", "cloudflare", "@cf/openai/gpt-oss-20b"),
        ("research", "gemini", "gemini-3.5-flash-lite"),
    ]
