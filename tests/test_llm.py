from types import SimpleNamespace

import pytest

import config
import llm


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setattr(config, "CHAT_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "CHAT_MODEL", None)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-anthropic-key")


def test_defaults_to_claude_haiku_with_temperature_zero(settings):
    model = llm.make_chat_model()
    assert type(model).__name__ == "ChatAnthropic"
    assert model.model == "claude-haiku-4-5" and model.temperature == 0
    assert model.max_tokens == llm.MAX_OUTPUT_TOKENS


def test_chat_model_setting_overrides_the_default(settings, monkeypatch):
    monkeypatch.setattr(config, "CHAT_MODEL", "claude-sonnet-4-6")
    assert llm.make_chat_model().model == "claude-sonnet-4-6"


@pytest.mark.parametrize("name", ["claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-4-8", "claude-fable-5-1"])
def test_models_that_reject_sampling_parameters_get_no_temperature(settings, name):
    assert llm.make_chat_model(model=name).temperature is None


def test_missing_anthropic_key_gives_a_clear_error(settings, monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY.*CHAT_PROVIDER=openai"):
        llm.make_chat_model()


def test_openai_provider_still_works_with_its_own_default(settings, monkeypatch):
    monkeypatch.setattr(config, "CHAT_PROVIDER", "openai")
    model = llm.make_chat_model()
    assert type(model).__name__ == "ChatOpenAI" and model.model_name == "gpt-4o-mini" and model.temperature == 0


def test_explicit_provider_and_model_arguments_win_over_config(settings):
    model = llm.make_chat_model("openai", "gpt-4o")
    assert type(model).__name__ == "ChatOpenAI" and model.model_name == "gpt-4o"


def test_unknown_provider_is_rejected(settings, monkeypatch):
    monkeypatch.setattr(config, "CHAT_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="CHAT_PROVIDER"):
        llm.make_chat_model()


def test_reply_text_handles_strings_and_content_blocks():
    assert llm.reply_text(SimpleNamespace(content="plain")) == "plain"
    blocks = [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]
    assert llm.reply_text(SimpleNamespace(content=blocks)) == "Hello world"


def test_model_from_the_wrong_provider_fails_early_with_a_clear_message(settings, monkeypatch):
    monkeypatch.setattr(config, "CHAT_PROVIDER", "openai")
    monkeypatch.setattr(config, "CHAT_MODEL", "claude-haiku-4-5")
    with pytest.raises(ValueError, match="doesn't belong to CHAT_PROVIDER 'openai'.*'gpt-4o-mini'"):
        llm.make_chat_model()
    monkeypatch.setattr(config, "CHAT_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "CHAT_MODEL", "gpt-4o-mini")
    with pytest.raises(ValueError, match="doesn't belong to CHAT_PROVIDER 'anthropic'"):
        llm.make_chat_model()
