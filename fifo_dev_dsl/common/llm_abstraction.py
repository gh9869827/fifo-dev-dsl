from __future__ import annotations

import argparse
import re
from enum import StrEnum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, cast, TYPE_CHECKING, Callable
from textwrap import dedent

if TYPE_CHECKING:  # pragma: no cover
    # Optional dependency: used by static type checkers only, not imported at runtime.
    from openai.types.chat import ChatCompletion
    from fifo_tool_airlock_model_env.common.models import Model

@dataclass(frozen=True)
class LlmRequest:
    """
    Request sent to an LLM.

    Depending on the backend configuration, it can be used either for DSL
    generation or general reasoning.

    The request consists of exactly two messages:
    - a system prompt defining expected behavior, instructions, available
      tools, and data sources
    - a user prompt containing the task, contextual information, and any
      retrieved or provided data

    Args:
        system_prompt (str):
            System prompt content.

        user_prompt (str):
            User prompt content.

        max_new_tokens (int):
            Maximum number of tokens to generate.

        temperature (float):
            Sampling temperature (higher = more random). When 0.0, use greedy
            decoding.

        reasoning_effort (str | None):
            Reasoning effort level for reasoning-capable models. Only applicable
            when using such models. Supported values depend on the backend
            implementation. Typical values include "low", "medium", and "high".
            When None, the parameter is not passed to the backend, allowing the
            model to use its default reasoning behavior. Defaults to None.
    """

    system_prompt: str
    user_prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.0
    reasoning_effort: str | None = None


class LlmBackend(ABC):
    """
    Backend interface for invoking an LLM.

    This abstract class defines a minimal, stateless request/response interface.

    The `complete()` method may perform either DSL generation or general
    reasoning, depending on the model and/or adapter invoked by the backend.
    """

    @abstractmethod
    def complete(self, req: LlmRequest) -> str:
        """
        Execute the given LLM request and return the model output.

        Args:
            req (LlmRequest):
                LLM request containing system and user prompts, along with
                generation parameters.

        Returns:
            str:
                Model output. The exact semantics (e.g. DSL code or natural
                language) depend on the LLM invoked by the backend.
        """
        raise NotImplementedError


class AirlockBackend(LlmBackend):
    """
    LLM backend that calls the Airlock model environment.

    This backend routes requests to a locally hosted Airlock model server and
    returns the output produced by the model.

    Typically, when both a base model and an adapter are provided, the backend
    returns structured DSL. When only a base model is provided, the backend is
    used to leverage the reasoning capabilities of the base (non-fine-tuned)
    foundation model. The backend does not enforce this behavior and returns
    the output produced by the invoked model.

    Behavior:
        - When both `base_model` and `adapter` are provided, the adapter
          fine-tuned on top of the base model is invoked.
        - When only `base_model` is provided, the base model is invoked directly.

    Args:
        _host (str):
            Base URL of the Airlock model server
            (e.g. "http://127.0.0.1:8000").

        _container_name (str):
            Name of the container running the Airlock model server.

        _base_model (Model):
            Base model to use (e.g., Phi4MiniInstruct, Phi4MultimodalInstruct).
            Defaults to Phi4MiniInstruct if not provided.

        _adapter (str | None):
            Optional adapter used by the model server.
            Defaults to None.
    """

    def __init__(self,
                 *,
                 host: str,
                 container_name: str,
                 base_model: str | Model | None = None,
                 adapter: str | None = None) -> None:
        """
        Initialize an Airlock-backed LLM interface.

        This constructor lazily imports the Airlock SDK to allow the backend
        dependency to remain optional for users who only rely on other LLM
        backends.

        Args:
            host (str):
                Base URL of the Airlock model server.

            container_name (str):
                Name of the container running the Airlock model server.

            base_model (str | Model | None):
                Base model to use. Can be a Model enum instance, a string matching a
                Model enum value (e.g. "Phi4MiniInstruct",
                "Phi4MultimodalInstruct"), or None to use the default.
                If not provided, defaults to Phi4MiniInstruct.

            adapter (str | None):
                Fine-tuned adapter to use. If None, only the base model is invoked.
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
        if base_model is None:
            self._model = Model.Phi4MiniInstruct
        elif isinstance(base_model, Model):
            # Model enum instance passed directly
            self._model = base_model
        else:
            # String value - convert to Model enum using value-based parsing
            try:
                self._model = Model(base_model)
            except ValueError as e:
                valid_values = ", ".join(m.value for m in Model)
                raise ValueError(
                    f"Invalid model: {base_model!r}. Must be one of: {valid_values}"
                ) from e

    def complete(self, req: LlmRequest) -> str:
        """
        Forward the request to the Airlock model server.

        Args:
            req (LlmRequest):
                LLM request containing system and user prompts as well as
                generation parameters.

        Returns:
            str:
                Model output. The exact semantics (e.g. DSL code or natural
                language) depend on the LLM invoked by the backend.
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


class OpenAICompatibleBackend(LlmBackend):
    """
    LLM backend using an OpenAI-compatible API (e.g. vLLM, LM Studio, Ollama).

    Note:
        This does not imply use of OpenAI-hosted services.

    This backend routes requests to an OpenAI-compatible server and returns
    the output produced by the model exposed at the given endpoint.

    The selected model may correspond either to a fine-tuned adapter intended
    for structured DSL generation or to a general-purpose foundation model
    intended for reasoning. The backend does not enforce a distinction and
    returns the output produced by the invoked model.

    Args:
        base_url (str):
            Base URL for the OpenAI-compatible server, including "/v1"
            (e.g. "http://127.0.0.1:8001/v1").

        model (str):
            Name or identifier of the model exposed by the server.

        api_key (str):
            API key used by the OpenAI client. Many local servers ignore this,
            but the client expects a value.

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
        Forward the request to an OpenAI-compatible API.

        Args:
            req (LlmRequest):
                LLM request containing system and user prompts as well as
                generation parameters.

        Returns:
            str:
                Model output. The exact semantics (e.g. DSL code or natural
                language) depend on the LLM invoked by the backend.
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


# ------------------------------
# Command line arguments parsing
# ------------------------------

class LlmBackendType(StrEnum):
    """
    Supported backend adapter types for routing LLM requests.

    This enum defines the backends supported for routing. Inheriting from 
    StrEnum provides direct compatibility with string-based interfaces 
    like argparse, JSON serialization, and configuration files.

    Members:
        AIRLOCK:
            Backend running an Airlock model environment.

        OPENAI_COMPATIBLE:
            Backend implementing the OpenAI-compatible API surface.
    """

    AIRLOCK = "airlock"
    """
    Backend running an Airlock model environment.
    """

    OPENAI_COMPATIBLE = "openai-compatible"
    """
    Backend implementing the OpenAI-compatible API surface (e.g., vLLM, Ollama).
    """


@dataclass(frozen=True)
class Backends:
    """
    Container for resolved backends from CLI.

    Attributes:
        dsl (LlmBackend):
            Backend used for DSL generation.

        reasoning (LlmBackend | None):
            Optional backend used for general reasoning.
    """
    dsl: LlmBackend
    reasoning: LlmBackend | None


def _make_dsl_parser(
    *,
    backend: LlmBackendType,
    default_adapter: str,
) -> argparse.ArgumentParser:
    """
    Create the parser for the DSL backend *options* for a specific LLM backend type (airlock vs
    openai-compatible).

        Airlock Backend:
        ================

        tool dsl=airlock
            [--host HOST]
            [--container CONTAINER]
            [--model MODEL]
            [--adapter ADAPTER]

        Defaults:
            --host      http://127.0.0.1:8000
            --container phi
            --model     Phi4MiniInstruct
            --adapter   {default_adapter}

        OpenAI Compatible Backend:
        ==========================

        tool dsl=openai-compatible
            --base-url URL
            [--adapter ADAPTER]
            [--api-key KEY]

        Requirements:
            --base-url  required

        Defaults:
            --adapter   {default_adapter}
            --api-key   EMPTY

    Args:
        backend (LlmBackendType):
            Backend selector value (e.g., "airlock" or "openai-compatible").

        default_adapter (str):
            Default adapter name used when --adapter is omitted.

    Returns:
        argparse.ArgumentParser:
            An ArgumentParser configured with the options for the selected LLM backend.

    Raises:
        ValueError:
            If `backend` is not a recognized DSL backend.
    """
    p = argparse.ArgumentParser(add_help=False)

    if backend == LlmBackendType.AIRLOCK:
        p.add_argument(
            "--host",
            default="http://127.0.0.1:8000",
            help='Airlock server URL. (default: http://127.0.0.1:8000)',
        )
        p.add_argument(
            "--container",
            default="phi",
            help="Airlock container name. (default: phi)",
        )
        p.add_argument(
            "--model",
            default="Phi4MiniInstruct",
            help="Airlock base model. (default: Phi4MiniInstruct)",
        )
        p.add_argument(
            "--adapter",
            default=default_adapter,
            help=(
                "Adapter identifier used for DSL generation. "
                f"(default: {default_adapter})"
            ),
        )
        return p

    if backend == LlmBackendType.OPENAI_COMPATIBLE:
        p.add_argument(
            "--base-url",
            required=True,
            help='Base URL for OpenAI-compatible server, including "/v1". (required)',
        )
        p.add_argument(
            "--adapter",
            default=default_adapter,
            help=(
                "Model identifier used for DSL generation. "
                f"(default: {default_adapter})"
            ),
        )
        p.add_argument(
            "--api-key",
            default="EMPTY",
            help='API key for OpenAI-compatible server. (default: "EMPTY")',
        )
        return p

    raise ValueError(f"Unknown DSL backend: {backend!r}")


def _make_reasoning_parser(*, backend: LlmBackendType) -> argparse.ArgumentParser:
    """
    Create the parser for the reasoning backend *options* for a specific LLM backend type
    (airlock vs openai-compatible).

        Airlock Backend:
        ================

        tool ... reasoning=airlock
            [--host HOST]
            [--container CONTAINER]
            [--model MODEL]

        Defaults:
            --host      http://127.0.0.1:8000
            --container phi
            --model     Phi4MiniInstruct

        OpenAI Compatible Backend:
        ==========================

        tool ... reasoning=openai-compatible
            --base-url URL
            --model MODEL
            [--api-key KEY]

        Requirements:
            --base-url  required
            --model     required

        Defaults:
            --api-key   EMPTY

    Args:
        backend (LlmBackendType):
            Backend selector value (e.g. "airlock" or "openai-compatible").

    Returns:
        argparse.ArgumentParser:
            An ArgumentParser configured with the options for the selected backend.

    Raises:
        ValueError:
            If `backend` is not a recognized reasoning backend.
    """
    p = argparse.ArgumentParser(add_help=False)

    if backend == LlmBackendType.AIRLOCK:
        p.add_argument(
            "--host",
            default="http://127.0.0.1:8000",
            help='Airlock server URL. (default: http://127.0.0.1:8000)',
        )
        p.add_argument(
            "--container",
            default="phi",
            help="Airlock container name. (default: phi)",
        )
        p.add_argument(
            "--model",
            default="Phi4MiniInstruct",
            help="Model identifier to use for reasoning. (default: Phi4MiniInstruct)",
        )
        return p

    if backend == LlmBackendType.OPENAI_COMPATIBLE:
        p.add_argument(
            "--base-url",
            required=True,
            help='Base URL for OpenAI-compatible server, including "/v1". (required)',
        )
        p.add_argument(
            "--model",
            required=True,
            help="Model identifier to use for reasoning. (required)",
        )
        p.add_argument(
            "--api-key",
            default="EMPTY",
            help='API key for OpenAI-compatible server. (default: "EMPTY")',
        )
        return p

    raise ValueError(f"Unknown reasoning backend: {backend!r}")


@dataclass(frozen=True)
class ParsedBackends:
    """
    Parsed backend configuration.

    Attributes:
        dsl_backend_type (LlmBackendType):
            DSL backend type: "airlock" or "openai-compatible".

        dsl_args (argparse.Namespace):
            Parsed args for the DSL backend.

        reasoning_backend_type (LlmBackendType | None):
            Reasoning backend type: "airlock" or "openai-compatible", or None if not provided.

        reasoning_args (argparse.Namespace | None):
            Parsed args for the reasoning backend, or None if not provided.

        extra_argv (list[str]):
            Unparsed command-line arguments remaining after all backend-related
            arguments have been consumed. These arguments may be application-
            specific and are intended to be parsed by a separate ArgumentParser.
    """
    dsl_backend_type: LlmBackendType
    dsl_args: argparse.Namespace
    reasoning_backend_type: LlmBackendType | None
    reasoning_args: argparse.Namespace | None
    extra_argv: list[str]


def _add_backend_cli_arguments(
    parser: argparse.ArgumentParser,
    default_adapter: str,
    require_reasoning: bool
) -> None:
    """
    Add usage/help text describing the key=value backend configuration syntax.

    Note:
        For the backend configuration, this tool uses an order-insensitive,
        selector-based style:

        - DSL only:
            tool dsl=<airlock|openai-compatible> [dsl-args...]

        - DSL + reasoning:
            tool dsl=<...> [dsl-args...] reasoning=<...> [reasoning-args...]

        Each selector (`dsl=...`, `reasoning=...`) appears at most once.
        All arguments following a selector apply to that backend until the
        next selector is encountered.

        Actual parsing is performed by `_parse_backend_args()`. This function
        documents the expected CLI shape for `-h` output only.
    """
    parser.formatter_class = argparse.RawTextHelpFormatter

    parser.epilog = dedent(
        f"""
        Backend configuration
        =====================

        DSL backend (required)
        ----------------------

          {parser.prog} dsl=airlock
              [--host HOST]
              [--container CONTAINER]
              [--model MODEL]
              [--adapter ADAPTER]

            Defaults:
              --host      http://127.0.0.1:8000
              --container phi
              --model     Phi4MiniInstruct
              --adapter   {default_adapter}


          {parser.prog} dsl=openai-compatible
              --base-url URL
              [--adapter ADAPTER]
              [--api-key KEY]

            Requirements:
              --base-url  required

            Defaults:
              --adapter   {default_adapter}
              --api-key   EMPTY
        """
    ).strip()

    if not require_reasoning:
        return

    parser.epilog += dedent(
        f"""

        Reasoning backend (optional)
        ----------------------------

          {parser.prog} ... reasoning=airlock
              [--host HOST]
              [--container CONTAINER]
              [--model MODEL]

            Defaults:
              --host      http://127.0.0.1:8000
              --container phi
              --model     Phi4MiniInstruct


          {parser.prog} ... reasoning=openai-compatible
              --base-url URL
              --model MODEL
              [--api-key KEY]

            Requirements:
              --base-url  required
              --model     required

            Defaults:
              --api-key   EMPTY
        """
    )

    parser.epilog = parser.epilog.strip()


def _parse_backend_args(
    argv: list[str],
    *,
    default_adapter: str,
    prog: str,
    require_reasoning: bool = False,
) -> ParsedBackends:
    """
    Parse backend configuration from argv using an assignment-token style.

    Supported (order-insensitive, each at most once):
      - dsl=<airlock|openai-compatible> [args...]
      - reasoning=<airlock|openai-compatible> [args...]

    Tokens appearing before the first assignment are returned in `extra_argv`.
    Tokens following an assignment belong to that section until another 
    assignment is encountered.
    """

    remaining = list(argv)

    dsl_backend_type: LlmBackendType | None = None
    dsl_args: argparse.Namespace | None = None
    reasoning_backend_type: LlmBackendType | None = None
    reasoning_args: argparse.Namespace | None = None

    extra_argv: list[str] = []

    assign_re = re.compile(r"^(dsl|reasoning)=([a-zA-Z-]+)$")

    current_cmd: str | None = None
    sections: dict[str, tuple[LlmBackendType, list[str]]] = {}

    i = 0
    while i < len(remaining):
        tok = remaining[i]
        m = assign_re.match(tok)

        if m:
            cmd = m.group(1)
            backend = m.group(2)

            try:
                backend_type = LlmBackendType(backend)
            except ValueError as e:
                raise SystemExit(f"{prog}: error: invalid backend spec '{tok}'") from e

            if cmd in sections:
                raise SystemExit(f"{prog}: error: duplicate command '{cmd}'")

            sections[cmd] = (backend_type, [])
            current_cmd = cmd
            i += 1
            continue

        if current_cmd is None:
            extra_argv.append(tok)
        else:
            backend_type, args_list = sections[current_cmd]
            args_list.append(tok)
            sections[current_cmd] = (backend_type, args_list)

        i += 1

    if "dsl" not in sections:
        raise SystemExit(f"{prog}: error: missing required 'dsl=...'")

    dsl_backend_type, dsl_section_argv = sections["dsl"]
    dsl_parser = _make_dsl_parser(backend=dsl_backend_type, default_adapter=default_adapter)
    dsl_parser.prog = f"{prog} dsl={dsl_backend_type}"
    dsl_args = dsl_parser.parse_args(dsl_section_argv)

    if "reasoning" in sections:
        reasoning_backend_type, reasoning_section_argv = sections["reasoning"]
        reasoning_parser = _make_reasoning_parser(backend=reasoning_backend_type)
        reasoning_parser.prog = f"{prog} reasoning={reasoning_backend_type}"
        reasoning_args = reasoning_parser.parse_args(reasoning_section_argv)

    if require_reasoning and reasoning_args is None:
        raise SystemExit(f"{prog}: error: missing required 'reasoning=...'")

    return ParsedBackends(
        dsl_backend_type=dsl_backend_type,
        dsl_args=dsl_args,
        reasoning_backend_type=reasoning_backend_type,
        reasoning_args=reasoning_args,
        extra_argv=extra_argv,
    )

# -----------------------------
# Backend instantiation
# -----------------------------

@dataclass(frozen=True)
class CreatedBackends:
    """
    Created backend instances.

    Attributes:
        dsl (LlmBackend):
            Backend used for DSL generation.

        reasoning (LlmBackend | None):
            Backend used for reasoning, or None if not configured.
    """
    dsl: LlmBackend
    reasoning: LlmBackend | None


def _create_backends_from_parsed(
    parsed: ParsedBackends,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> CreatedBackends:
    """
    Instantiate DSL and optional reasoning backends from parsed configuration.

    Args:
        parsed (ParsedBackends):
            Output of `_parse_backend_args()`.

        parser (argparse.ArgumentParser | None):
            Optional parser for consistent error formatting (in order to use parser.error()).

    Returns:
        CreatedBackends:
            DSL backend and optional reasoning backend.

    Raises:
        SystemExit:
            If a required argument is missing for the selected backend.
    """
    def _error(msg: str) -> None:
        if parser is not None:
            parser.error(msg)
        raise SystemExit(f"error: {msg}")

    # DSL backend
    dsl_backend: LlmBackend
    if parsed.dsl_backend_type == LlmBackendType.AIRLOCK:
        dsl_backend = AirlockBackend(
            container_name=parsed.dsl_args.container,
            host=parsed.dsl_args.host,
            base_model=parsed.dsl_args.model,
            adapter=parsed.dsl_args.adapter,  # DSL expects adapter (defaulted if omitted)
        )
    elif parsed.dsl_backend_type == LlmBackendType.OPENAI_COMPATIBLE:
        if not parsed.dsl_args.base_url:
            _error("'dsl openai-compatible' requires --base-url")
        dsl_backend = OpenAICompatibleBackend(
            base_url=parsed.dsl_args.base_url,
            model=parsed.dsl_args.adapter,    # DSL uses adapter identifier as the model name
            api_key=parsed.dsl_args.api_key,
        )
    else:
        _error(f"Unknown DSL backend type: {parsed.dsl_backend_type}")
        raise AssertionError("unreachable")

    # Reasoning backend (optional)
    reasoning_backend: LlmBackend | None = None
    if parsed.reasoning_args is not None and parsed.reasoning_backend_type is not None:
        if parsed.reasoning_backend_type == LlmBackendType.AIRLOCK:
            reasoning_backend = AirlockBackend(
                container_name=parsed.reasoning_args.container,
                host=parsed.reasoning_args.host,
                base_model=parsed.reasoning_args.model,
                adapter=None,  # reasoning uses base model directly
            )
        elif parsed.reasoning_backend_type == LlmBackendType.OPENAI_COMPATIBLE:
            if not parsed.reasoning_args.base_url:
                _error("'reasoning openai-compatible' requires --base-url")
            reasoning_backend = OpenAICompatibleBackend(
                base_url=parsed.reasoning_args.base_url,
                model=parsed.reasoning_args.model,  # reasoning uses base model name
                api_key=parsed.reasoning_args.api_key,
            )
        else:
            _error(f"Unknown reasoning backend type: {parsed.reasoning_backend_type}")
            raise AssertionError("unreachable")

    return CreatedBackends(dsl=dsl_backend, reasoning=reasoning_backend)


@dataclass(frozen=True)
class CliArgParsingResult:
    """
    Result of parsing the CLI.

    Attributes:
        backends (CreatedBackends):
            The instantiated backend objects.

        global_args (argparse.Namespace):
            Parsed program-level arguments (the args added to the global parser).

        parsed_backends (ParsedBackends):
            Parsed backend configuration returned by `_parse_backend_args`.
    """

    backends: CreatedBackends
    global_args: argparse.Namespace
    parsed_backends: ParsedBackends


def parse_cli_and_create_backends(
    argv: list[str],
    *,
    prog: str,
    description: str,
    default_adapter: str,
    require_reasoning: bool,
    add_global_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> CliArgParsingResult:
    """
    Orchestrates the multi-stage parsing of global and backend-specific arguments.

    The parsing pipeline follows these steps:
    1. Builds the global parser and applies optional extensions.
    2. Scan `argv`, identifying backend selectors (`dsl=...`, `reasoning=...`)
       while collecting arguments that appear before any backend selector.
    3. Parses backend-specific arguments using specialized sub-parsers.
    4. Parses global arguments using the primary parser.
    5. Instantiates the resulting backend objects.

    Args:
        argv (list[str]):
            Argument vector excluding the program name.

        prog (str):
            Program name used in help and error messages.

        description (str):
            High-level description for the global parser.

        default_adapter (str):
            Default adapter identifier used if the DSL backend configuration omits one.

        require_reasoning (bool):
            If True, validation will fail if a `reasoning=...` section is missing.

        add_global_arguments (Callable[[argparse.ArgumentParser], None] | None):
            Optional callback to add program-level flags (e.g., `--max-tokens`).
            Note: Global arguments MUST appear before any backend specifications.

    Returns:
        CliArgParsingResult:
            A container holding instantiated backends, parsed global arguments,
            and the raw parsed backend configuration.
    """
    global_parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
    )

    if add_global_arguments is not None:
        add_global_arguments(global_parser)

    # Backend help text (epilog only)
    _add_backend_cli_arguments(global_parser, default_adapter, require_reasoning)

    # If user asks for help, show it and exit
    if "-h" in argv or "--help" in argv:
        global_parser.parse_args(argv)  # prints help + exits via SystemExit
        raise AssertionError("unreachable")

    parsed_backends = _parse_backend_args(
        argv,
        default_adapter=default_adapter,
        prog=global_parser.prog,
        require_reasoning=require_reasoning,
    )

    global_args = global_parser.parse_args(parsed_backends.extra_argv)

    backends = _create_backends_from_parsed(
        parsed_backends,
        parser=global_parser,
    )

    return CliArgParsingResult(
        backends=backends,
        global_args=global_args,
        parsed_backends=parsed_backends,
    )
