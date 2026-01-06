# `fifo_dev_dsl.common.llm_abstraction` Module

This module provides a **project-level unified LLM backend abstraction** for the `fifo-dev-dsl` codebase, enabling tools and DSL modules to work with multiple LLM providers through a consistent interface.

It supports:
- **Airlock model environment**: Local [Airlock model environment](https://github.com/gh9869827/fifo-tool-airlock-model-env) for fine-tuned adapters
- **OpenAI-compatible APIs**: vLLM, LM Studio, Ollama, and other compatible servers

The abstraction separates **DSL generation** (using fine-tuned adapters) from **general reasoning** (using base models), allowing tools to leverage both capabilities when needed.

---

## 🎯 Purpose

The LLM backend abstraction provides:

1. **Project-level Unified Interface**: Work with different LLM providers using the same API
2. **Dual Backend Support**: Configure separate backends for DSL generation and reasoning
3. **CLI Integration**: Parse backend configuration from command-line arguments
4. **Flexible Configuration**: Support both Airlock and OpenAI-compatible servers

---

## 📚 Table of Contents

- 🎯 [Project Status & Audience](#-project-status--audience)
- 🔌 [Supported LLM Backends](#-supported-backends)
- 🖥️ [Command-Line Usage](#️-command-line-usage)
- 🐍 [Python API](#-python-api)
- 📖 [Examples](#-examples)

---

## 🎯 Project Status & Audience

🚧 **Work in Progress** — Part of the **`fifo-dev-dsl`** project, currently in **early development**. 🚧

This is a personal project developed and maintained by a solo developer.  
Contributions, ideas, and feedback are welcome, but development is driven by personal time and priorities.

`fifo-dev-dsl` is designed to support other `fifo-*` projects developed by the author.  
It is provided for **individual developers** interested in experimenting with DSL-driven natural language interpretation.

No official release or pre-release has been published yet. The code is provided for **preview and experimentation**.  
**Use at your own risk.**

## 🔌 Supported LLM Backends

Each LLM backend can be used to serve LLMs for either DSL generation or reasoning.
Structured DSL generation is typically performed by invoking a model configured
with a fine-tuned LoRA adapter specific to the DSL syntax, while reasoning is
typically performed by a foundation model.

### Airlock LLM Backend

Connects to a locally hosted [Airlock model environment](https://github.com/gh9869827/fifo-tool-airlock-model-env).

### OpenAI-Compatible LLM Backend

Connects to any OpenAI-compatible API server (vLLM, LM Studio, Ollama, etc.).

---

## 🖥️ Command-Line Usage

### Backend Specification Format

Backends are specified using a **selector-based syntax**:

```bash
tool [global-args...] dsl=<backend-type> [dsl-args...] [reasoning=<backend-type> [reasoning-args...]]
```

- **Order-sensitive**: Global args must appear before any backend selector
- **Each selector appears at most once**: `dsl=...` and `reasoning=...`
- **Arguments follow their selector**: All args after a selector apply to that backend until the next selector

### DSL Backend (Required)

The DSL backend is used for generating structured DSL syntax from natural language.

#### Airlock DSL Backend

```bash
tool dsl=airlock \
    [--container CONTAINER] \
    [--host HOST] \
    [--model MODEL] \
    [--adapter ADAPTER]
```

**Defaults:**
- `--container`: `phi`
- `--host`: `http://127.0.0.1:8000`
- `--model`: `Phi4MiniInstruct`
- `--adapter`: Depends on the tool

**Example:**
```bash
python evaluate_mini_date_converter_dsl_model.py \
    dsl=airlock \
        --container phi \
        --model Phi4MiniInstruct \
        --adapter mini-date-converter-dsl-adapter
```

#### OpenAI-Compatible DSL Backend

```bash
tool dsl=openai-compatible \
    --base-url URL \
    [--adapter ADAPTER] \
    [--api-key KEY]
```

**Requirements:**
- `--base-url`: Required (must include `/v1`)

**Defaults:**
- `--adapter`: Depends on the tool
- `--api-key`: `EMPTY`

**Example:**
```bash
python evaluate_mini_date_converter_dsl_model.py \
    dsl=openai-compatible \
        --base-url http://127.0.0.1:8001/v1 \
        --adapter mini-date-converter-dsl-adapter
```

### Reasoning Backend (Optional, Depending on the Tool)

The reasoning backend is used for general reasoning tasks. Its use may be required or optional,
depending on the tool.

#### Airlock Reasoning Backend

```bash
tool ... reasoning=airlock \
    [--container CONTAINER] \
    [--host HOST] \
    [--model MODEL]
```

**Defaults:**
- `--container`: `phi`
- `--host`: `http://127.0.0.1:8000`
- `--model`: `Phi4MiniInstruct`

**Example:**
```bash
python calculator_eval_performance.py \
    dsl=airlock \
        --adapter dia-intent-sequencer-calculator-adapter \
    reasoning=airlock \
        --model Phi4MiniInstruct
```

#### OpenAI-Compatible Reasoning Backend

```bash
tool ... reasoning=openai-compatible \
    --base-url URL \
    --model MODEL \
    [--api-key KEY]
```

**Requirements:**
- `--base-url`: Required (must include `/v1`)
- `--model`: Required

**Defaults:**
- `--api-key`: `EMPTY`

**Example:**
```bash
python calculator_eval_performance.py \
    dsl=openai-compatible \
        --base-url http://127.0.0.1:8001/v1 \
        --adapter calculator-adapter \
    reasoning=openai-compatible \
        --base-url http://127.0.0.1:8002/v1 \
        --model reasoning-model
```

### Mixed Backend Configuration

You can mix Airlock and OpenAI-compatible backends:

```bash
python tool.py \
    dsl=airlock \
        --adapter my-dsl-adapter \
    reasoning=openai-compatible \
        --base-url http://127.0.0.1:8001/v1 \
        --model reasoning-model
```

---

## 🐍 Python API

### Core Function: `parse_cli_and_create_backends`

This function provides a single entry point for handling command-line arguments and creating backend instances.

```python
from fifo_dev_dsl.common.llm_abstraction import parse_cli_and_create_backends

def main(argv: list[str]) -> None:
    res = parse_cli_and_create_backends(
        argv,
        prog="my_tool.py",
        description="Tool description",
        default_adapter="my-default-adapter",
        require_reasoning=False,  # True if reasoning is required
        add_global_arguments=add_global_args,  # Optional callback (see below).
    )

    # Access created backends
    dsl_backend = res.backends.dsl
    reasoning_backend = res.backends.reasoning  # May be None

    if reasoning_backend is not None:
        pass  # Use reasoning backend

    # Access parsed global arguments
    global_args = res.global_args
```

### Adding Global Arguments

Define a callback to add tool-specific global arguments.  
**These flags must appear before any backend selectors** on the command line.

```python
import argparse

def add_global_args(parser: argparse.ArgumentParser) -> None:
    # Global arguments must appear before any backend selectors
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
```

### Using Backends in Code

Once backends are created, use them with `LlmRequest`:

```python
from fifo_dev_dsl.common.llm_abstraction import LlmRequest

# Create a request
request = LlmRequest(
    system_prompt="You are a helpful assistant...",
    user_prompt="Schedule the task...",
    max_new_tokens=1024,
    temperature=0.0,
    reasoning_effort="medium"  # Optional, for reasoning models
)

# Execute the request
response = dsl_backend.complete(request)
print(response)
```

### Direct Backend Instantiation

You can also create backends directly without CLI parsing:

```python
from fifo_dev_dsl.common.llm_abstraction import AirlockBackend, OpenAICompatibleBackend

# Airlock backend
airlock = AirlockBackend(
    container_name="phi",
    host="http://127.0.0.1:8000",
    base_model="Phi4MiniInstruct",
    adapter="my-adapter"
)

# OpenAI-compatible backend
openai_compat = OpenAICompatibleBackend(
    base_url="http://127.0.0.1:8001/v1",
    model="my-model",
    api_key="EMPTY"
)
```

---

## 📖 Examples

### Example 1: DSL Evaluation Tool (Single Backend)

Below is a simplified excerpt showing the global-argument pattern; see the
[full file](../domain_specific/mini_date_converter_dsl/evaluate_mini_date_converter_dsl_model.py) for evaluation logic and branching.

```python
def add_global_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--template-base", type=int, choices=(1, 2))

def main(argv: list[str]) -> None:
    res = parse_cli_and_create_backends(
        argv,
        prog="evaluate_mini_date_converter_dsl_model.py",
        description="Evaluate mini date converter DSL model accuracy",
        default_adapter="mini-date-converter-dsl-adapter",
        require_reasoning=False,
        add_global_arguments=add_global_args,
    )
    
    # Use the DSL backend
    backend = res.backends.dsl
    max_tokens = res.global_args.max_new_tokens
    
    # Run evaluation...
```

**Usage:**
```bash
python evaluate_mini_date_converter_dsl_model.py \
    --max-new-tokens 2048 \
    dsl=airlock \
        --adapter mini-date-converter-dsl-adapter
```

### Example 2: Calculator with Dual Backends

Below is a second simplified excerpt; see the
[full file](../dia/demo/calculator_eval_performance.py) for evaluation logic and branching.


```python
def add_global_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--delta-flag", action="store_true")

def main(argv: list[str]) -> None:
    res = parse_cli_and_create_backends(
        argv,
        prog="calculator_eval_performance.py",
        description="Evaluate DIA calculator adapter",
        default_adapter="dia-intent-sequencer-calculator-adapter",
        require_reasoning=True,  # Reasoning backend is required
        add_global_arguments=add_global_args,
    )
    
    # Create runtime context with both backends
    runtime_context = LLMRuntimeContext(
        llm_backend_dsl=res.backends.dsl,
        llm_backend_reasoning=res.backends.reasoning,
        tools=[...],
        query_sources=[...]
    )
```

**Usage:**
```bash
python calculator_eval_performance.py \
    --random \
    dsl=airlock \
        --adapter dia-intent-sequencer-calculator-adapter \
    reasoning=openai-compatible \
        --model reasoning-model \
        --base-url http://127.0.0.1:8001/v1
```

### Example 3: Mixed Backend Configuration

```bash
# DSL via Airlock, reasoning via OpenAI-compatible
python tool.py \
    --temperature 0.0 \
    dsl=airlock \
        --adapter my-dsl-adapter \
        --container phi \
    reasoning=openai-compatible \
        --model reasoning-model \
        --base-url http://127.0.0.1:8001/v1 \
        --api-key my-key
```

---

## ✅ Key Design Principles

1. **Separation of Concerns**: DSL generation and reasoning use separate backends
2. **LLM Backend Agnostic**: Tools work with any supported LLM backend through a unified interface
3. **Flexible Configuration**: Support both CLI and direct instantiation, with an explicit CLI grammar
4. **Error Handling**: Clear error messages for invalid configurations

---

## 🔧 Implementation Details

### Backend Selection Flow

1. Scan the argument list, identifying backend selectors (`dsl=...`, `reasoning=...`) while collecting arguments that appear before any backend selector.
2. Parse backend-specific arguments
3. Parse global tool arguments
4. Instantiate backend objects
5. Return structured result with backends and parsed arguments

### Request/Response Flow

1. Create `LlmRequest` with prompts and parameters
2. Call `backend.complete(request)`
3. Backend routes to appropriate provider
4. Return model output as string

---

## ✅ License

MIT — see [LICENSE](../../LICENSE).
