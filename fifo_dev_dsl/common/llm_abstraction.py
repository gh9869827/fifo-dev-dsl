from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # Optional dependency: used by static type checkers only, not imported at runtime.
    from openai.types.chat import ChatCompletion

@dataclass(frozen=True)
class LlmRequest:
    """
    Request sent to an LLM for DSL generation.

    The request consists of exactly two messages:
    - a system prompt defining expected behavior, instructions, available tools, and
      any data sources
    - a user prompt containing the task, contextual information, and any retrieved data

    Args:
        system_prompt (str):
            System prompt content.

        user_prompt (str):
            User prompt content.

        max_new_tokens (int):
            Maximum number of tokens to generate.

        temperature (float):
            Sampling temperature (higher = more random). When 0.0, use greedy decoding.

        reasoning_effort (str | None):
            Reasoning effort level for reasoning models. Only applicable when using
            reasoning-capable models. Supported values depend on the backend
            implementation. Common values include "low", "medium", "high".
            When None, the parameter is not passed to the backend, allowing the
            model to use its default reasoning behavior. Defaults to None.
    """

    system_prompt: str
    user_prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.0
    reasoning_effort: str | None = None


class LlmBackend(Protocol):
    """
    Backend interface for generating DSL with an LLM.
    """

    def complete(self, req: LlmRequest) -> str:
        """
        Generate DSL output for the given request.

        Args:
            req (LlmRequest):
                LLM request containing system/user prompts and generation parameters.

        Returns:
            str:
                Model output (expected to be DSL code).
        """
        ... # pylint: disable=unnecessary-ellipsis


class AirlockBackend:
    """
    LLM backend that generates DSL by calling the Airlock model environment.

    This backend routes requests to a locally hosted Airlock model server and
    returns the raw DSL output produced by the model.

    Args:
        _container_name (str):
            Name of the container running the Airlock model server.

        _adapter (str):
            Adapter used by the model server to generate DSL.

        _host (str):
            Base URL of the Airlock model server
            (e.g. "http://127.0.0.1:8000").

        _model (Model):
            Base model to use for DSL generation (e.g., Phi4MiniInstruct,
            Phi4MultimodalInstruct). Defaults to Phi4MiniInstruct if not provided.
    """

    def __init__(self, *, container_name: str, adapter: str, host: str, model: "str | Model | None" = None) -> None:
        """
        Initialize an Airlock-backed LLM interface.

        This constructor lazily imports the Airlock SDK to allow the backend
        dependency to remain optional for users who only rely on other LLM
        backends.

        Args:
            container_name (str):
                Name of the container running the Airlock model server.

            adapter (str):
                Adapter to use when generating DSL.

            host (str):
                Base URL of the Airlock model server.

            model (str | Model | None):
                Base model to use for DSL generation. Can be a Model enum instance,
                a string matching a Model enum value (e.g., "Phi4MiniInstruct",
                "Phi4MultimodalInstruct"), or None to use the default.
                If not provided, defaults to Phi4MiniInstruct.
        """
        # pylint: disable=import-outside-toplevel
        from fifo_tool_airlock_model_env.common.models import (
            GenerationParameters,
            Message,
            Model,
        )
        from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server

        # Bind imported symbols to the instance to avoid hidden globals.
        self._generation_parameters_cls  = GenerationParameters
        self._message_cls  = Message
        self._model_enum  = Model
        self._call_airlock_model_server = call_airlock_model_server

        self._container_name = container_name
        self._adapter = adapter
        self._host = host
        
        # Store the model, defaulting to Phi4MiniInstruct
        if model is None:
            self._model = Model.Phi4MiniInstruct
        elif isinstance(model, Model):
            # Model enum instance passed directly
            self._model = model
        else:
            # String value - convert to Model enum using value-based parsing
            try:
                self._model = Model(model)
            except ValueError as e:
                valid_values = ", ".join(m.value for m in Model)
                raise ValueError(
                    f"Invalid model: {model!r}. Must be one of: {valid_values}"
                ) from e

    def complete(self, req: LlmRequest) -> str:
        """
        Generate DSL by forwarding the request to the Airlock model server.

        Args:
            req (LlmRequest):
                LLM request containing system and user prompts as well as
                generation parameters.

        Returns:
            str:
                DSL code generated by the model.
        """
        Message = self._message_cls
        GenerationParameters = self._generation_parameters_cls

        return self._call_airlock_model_server(
            model=self._model,
            adapter=self._adapter,
            messages=[
                Message.system(req.system_prompt),
                Message.user(req.user_prompt),
            ],
            parameters=GenerationParameters(
                max_new_tokens=req.max_new_tokens,
                do_sample=(req.temperature > 0.0),
            ),
            container_name=self._container_name,
            host=self._host,
        )


class OpenAICompatibleBackend:
    """
    LLM backend using an OpenAI-compatible API (e.g. vLLM, LM Studio, Ollama).
    This does not imply use of OpenAI-hosted services.

    Args:
        base_url (str):
            Base URL for the OpenAI-compatible server, including "/v1"
            (e.g. "http://127.0.0.1:8001/v1").

        model (str):
            Model name exposed by the server.

        api_key (str):
            API key used by the OpenAI client. Many local servers ignore this, but the
            client expects a value.

        timeout_s (float):
            Request timeout in seconds.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
    ) -> None:
        """
        Initialize an OpenAI-compatible LLM backend.

        Args:
            base_url (str):
                Base URL of the OpenAI-compatible server, including "/v1".

            model (str):
                Name of the model exposed by the server.

            api_key (str):
                API key passed to the OpenAI client. This value is often ignored by
                local servers but is required by the client interface.

            timeout_s (float):
                Request timeout in seconds.
        """
        # Local import keeps this dependency optional for users who only use an openai-compatible
        # backend.
        from openai import OpenAI  # pylint: disable=import-outside-toplevel

        self._model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
        )

    def complete(self, req: LlmRequest) -> str:
        """
        Generate DSL by forwarding the request to an OpenAI-compatible API.

        Args:
            req (LlmRequest):
                LLM request containing system and user prompts as well as
                generation parameters.

        Returns:
            str:
                DSL code generated by the model.
        """
        # Build kwargs for chat.completions.create
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.user_prompt},
            ],
            "temperature": req.temperature,
            "max_tokens": req.max_new_tokens,  # OpenAI API uses max_tokens
        }

        # Only include reasoning_effort if explicitly set (for reasoning models)
        if req.reasoning_effort is not None:
            create_kwargs["reasoning_effort"] = req.reasoning_effort

        resp = cast("ChatCompletion", self._client.chat.completions.create(**create_kwargs))

        content = (resp.choices[0].message.content or "").strip()
        return content
