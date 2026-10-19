"""
Covers what we wrote around the OpenAI SDK, not the SDK itself: the placeholder
key Ollama needs, the parameters we forward, and the refusal/emptiness
distinction.
"""

from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import SecretStr

from rag_architect.adapters import llm as llm_module
from rag_architect.adapters.llm import (
    NO_KEY_REQUIRED,
    LLMUnavailableError,
    OpenAICompatibleLLMClient,
)
from rag_architect.config import LLMSettings
from rag_architect.domain.ports import LLMClient


class StubCompletions:
    def __init__(self, response) -> None:
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class StubOpenAI:
    instances: ClassVar[list["StubOpenAI"]] = []

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.completions = StubCompletions(self.response)
        self.chat = SimpleNamespace(completions=self.completions)
        StubOpenAI.instances.append(self)

    response = None


def answer(content: str | None, finish_reason: str = "stop"):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


@pytest.fixture
def openai(monkeypatch):
    StubOpenAI.instances = []

    def factory(**kwargs):
        return StubOpenAI(**kwargs)

    monkeypatch.setattr(llm_module, "OpenAI", factory)
    return StubOpenAI


def test_satisfies_the_port(openai):
    StubOpenAI.response = answer("ok")

    client: LLMClient = OpenAICompatibleLLMClient(LLMSettings())

    assert isinstance(client, OpenAICompatibleLLMClient)


def test_a_placeholder_key_is_used_when_none_is_configured(openai):
    """Ollama needs no key, but the SDK refuses to build a client without one."""
    StubOpenAI.response = answer("ok")

    OpenAICompatibleLLMClient(LLMSettings())

    assert StubOpenAI.instances[0].init_kwargs["api_key"] == NO_KEY_REQUIRED


def test_a_configured_key_is_passed_through(openai):
    StubOpenAI.response = answer("ok")

    OpenAICompatibleLLMClient(LLMSettings(api_key=SecretStr("gsk-secret")))

    assert StubOpenAI.instances[0].init_kwargs["api_key"] == "gsk-secret"


def test_base_url_and_timeout_come_from_settings(openai):
    StubOpenAI.response = answer("ok")

    OpenAICompatibleLLMClient(
        LLMSettings(base_url="https://api.groq.com/openai/v1", timeout_s=12.0)
    )

    kwargs = StubOpenAI.instances[0].init_kwargs
    assert kwargs["base_url"] == "https://api.groq.com/openai/v1"
    assert kwargs["timeout"] == 12.0


def test_the_two_roles_are_sent_in_order(openai):
    StubOpenAI.response = answer("Turnover was 221,293 [page 3].")
    client = OpenAICompatibleLLMClient(LLMSettings())

    result = client.complete(
        system_prompt="Answer only from the excerpts.", user_prompt="Turnover?"
    )

    sent = StubOpenAI.instances[0].completions.calls[0]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][0]["content"] == "Answer only from the excerpts."
    assert sent["messages"][1]["content"] == "Turnover?"
    assert result == "Turnover was 221,293 [page 3]."


def test_sampling_parameters_come_from_settings(openai):
    StubOpenAI.response = answer("ok")
    client = OpenAICompatibleLLMClient(
        LLMSettings(temperature=0.0, max_tokens=256, model="qwen2.5:7b")
    )

    client.complete(system_prompt="s", user_prompt="u")

    sent = StubOpenAI.instances[0].completions.calls[0]
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 256
    assert sent["model"] == "qwen2.5:7b"


class TestFailures:
    def test_empty_content_is_an_error_not_a_refusal(self, openai):
        """A truncated completion must not be mistaken for a grounded refusal."""
        StubOpenAI.response = answer("", finish_reason="length")
        client = OpenAICompatibleLLMClient(LLMSettings())

        with pytest.raises(LLMUnavailableError, match="finish_reason=length"):
            client.complete(system_prompt="s", user_prompt="u")

    def test_none_content_is_an_error(self, openai):
        StubOpenAI.response = answer(None)
        client = OpenAICompatibleLLMClient(LLMSettings())

        with pytest.raises(LLMUnavailableError):
            client.complete(system_prompt="s", user_prompt="u")

    def test_whitespace_only_content_is_an_error(self, openai):
        StubOpenAI.response = answer("   \n  ")
        client = OpenAICompatibleLLMClient(LLMSettings())

        with pytest.raises(LLMUnavailableError):
            client.complete(system_prompt="s", user_prompt="u")

    def test_no_choices_is_an_error(self, openai):
        StubOpenAI.response = SimpleNamespace(choices=[])
        client = OpenAICompatibleLLMClient(LLMSettings())

        with pytest.raises(LLMUnavailableError, match="no choices"):
            client.complete(system_prompt="s", user_prompt="u")
