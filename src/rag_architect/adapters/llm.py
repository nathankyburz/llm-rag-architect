"""LLM client for any OpenAI-compatible chat completions endpoint."""

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from rag_architect.config import LLMSettings

# The OpenAI SDK refuses to build a client without a key. Ollama needs none, so
# a placeholder stands in; a real provider overrides it from the environment.
NO_KEY_REQUIRED = "not-needed"


class LLMUnavailableError(RuntimeError):
    """The provider answered, but with nothing usable."""


class OpenAICompatibleLLMClient:
    """
    Satisfies the LLMClient port against any endpoint speaking the OpenAI chat
    completions shape: Ollama locally, Groq in production. Switching providers is
    two environment variables and no code change.

    `max_tokens` rather than `max_completion_tokens`: the latter is OpenAI's own
    newer field, while the former is what compatible servers accept.
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._client = OpenAI(
            base_url=settings.base_url,
            api_key=(settings.api_key.get_secret_value() if settings.api_key else NO_KEY_REQUIRED),
            timeout=settings.timeout_s,
        )
        self._model = settings.model
        self._temperature = settings.temperature
        self._max_tokens = settings.max_tokens

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(role="user", content=user_prompt),
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        if not response.choices:
            msg = f"{self._model} returned no choices"
            raise LLMUnavailableError(msg)

        content = response.choices[0].message.content
        if not content or not content.strip():
            # A truncated or empty completion must not be mistaken for a refusal:
            # the answering service decides what a refusal looks like.
            reason = response.choices[0].finish_reason
            msg = f"{self._model} returned empty content (finish_reason={reason})"
            raise LLMUnavailableError(msg)
        return content
