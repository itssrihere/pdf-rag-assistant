"""Chat-model factory: Claude (default) or OpenAI, selected by CHAT_PROVIDER. Embeddings are always OpenAI."""

import config

DEFAULT_MODELS = {"anthropic": "claude-haiku-4-5", "openai": "gpt-4o-mini"}
# Newer Claude models reject custom sampling parameters with an HTTP 400, so temperature is only sent to
# models that accept it (Haiku 4.5, Sonnet/Opus 4.6 and earlier).
NO_SAMPLING_PARAMS = (
    "claude-fable", "claude-mythos", "claude-opus-5", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5",
)
MAX_OUTPUT_TOKENS = 2048  # required by the Anthropic API; ample for a grounded answer


def make_chat_model(provider: str | None = None, model: str | None = None):
    """The model that writes answers and rewrites follow-up questions (temperature 0 where supported)."""
    provider = (provider or config.CHAT_PROVIDER).lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"CHAT_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, got {provider!r}")
    model = model or config.CHAT_MODEL or DEFAULT_MODELS[provider]
    if model.startswith("claude") != (provider == "anthropic"):
        raise ValueError(
            f"CHAT_MODEL {model!r} doesn't belong to CHAT_PROVIDER {provider!r}; "
            f"change both together, or unset CHAT_MODEL to use {DEFAULT_MODELS[provider]!r}"
        )

    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("Set ANTHROPIC_API_KEY (e.g. in .env) to chat with Claude, or set CHAT_PROVIDER=openai")
        from langchain_anthropic import ChatAnthropic

        kwargs = {} if model.startswith(NO_SAMPLING_PARAMS) else {"temperature": 0}
        return ChatAnthropic(model=model, api_key=config.ANTHROPIC_API_KEY, max_tokens=MAX_OUTPUT_TOKENS, **kwargs)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, openai_api_key=config.OPENAI_API_KEY, temperature=0)


def reply_text(message) -> str:
    """Text of a chat reply. Providers may return a string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
        if not isinstance(block, dict) or block.get("type") == "text"
    )
