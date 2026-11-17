"""The unpatched LangChain advisories listed in scripts/audit.sh are accepted *because this project does not use the
affected features*. This test keeps that true: if the code starts using one, the build fails and the accepted risk
has to be reconsidered (or the LangChain 1.x migration done) first."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# feature -> the accepted advisory it belongs to
AFFECTED_FEATURES = {
    "HTMLHeaderTextSplitter": "langchain-text-splitters: SSRF redirect bypass in split_text_from_url",
    "split_text_from_url": "langchain-text-splitters: SSRF redirect bypass",
    "get_num_tokens_from_messages": "langchain-core / langchain-openai: SSRF via image URL token counting",
    "image_url": "langchain-core / langchain-openai: SSRF via image URL token counting",
    "load_prompt": "langchain-core: path traversal in prompt loading",
    "load_prompt_from_config": "langchain-core: path traversal in prompt loading",
    "FilesystemFileSearchMiddleware": "langchain / langchain-anthropic: path traversal in file-search middleware",
    "create_agent": "langchain: agent middleware and config loaders (path traversal)",
    "TracingMiddleware": "langsmith: arbitrary file read in tracing middleware",
    "pull_prompt": "langsmith: unsafe deserialization of public prompt manifests",
    "hub.pull": "langsmith: unsafe deserialization of public prompt manifests",
}


def application_sources():
    return [
        path for path in ROOT.rglob("*.py")
        if not any(part.startswith(".") or part in {"tests", "node_modules"} for part in path.relative_to(ROOT).parts)
    ]


def test_code_does_not_use_features_that_have_accepted_unpatched_advisories():
    used = {}
    for path in application_sources():
        text = path.read_text()
        for feature, advisory in AFFECTED_FEATURES.items():
            if feature in text:
                used.setdefault(feature, []).append(f"{path.relative_to(ROOT)} ({advisory})")
    assert not used, (
        "This code now uses a feature with an accepted, unpatched advisory; migrate to LangChain 1.x or drop the "
        f"feature first (see scripts/audit.sh): {used}"
    )


def test_the_check_actually_looks_at_the_code():
    sources = {path.name for path in application_sources()}
    assert {"ingest.py", "query.py", "llm.py"} <= sources
