# 🧠 fifo-dev-dsl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg) 
![Test Status](https://github.com/gh9869827/fifo-dev-dsl/actions/workflows/test.yml/badge.svg)

A suite of **domain-specific languages (DSLs)** for interpreting natural language and converting it into structured, executable logic within the `fifo-*` ecosystem.

At the center is **`dia`**, a DSL for interactive agents that translates user intent into tool calls, resolving missing information and runtime errors through user interaction or by querying contextual information available at runtime.

The repository also includes additional DSL modules under the `domain_specific` package for specialized conversions, such as date and recurrence interpretation.

---

## 🎯 Project Status & Audience

🚧 **Work in Progress** — This project is in **early development**. 🚧

This is a personal project developed and maintained by a solo developer.  
Contributions, ideas, and feedback are welcome, but development is driven by personal time and priorities.

`fifo-dev-dsl` is designed to support other `fifo-*` projects developed by the author.  
It is provided for **individual developers** interested in experimenting with DSL-driven natural language interpretation.

No official release or pre-release has been published yet. The code is provided for **preview and experimentation**.  
**Use at your own risk.**

---

## 📦 Install

Some functionality requires a working [airlock model environment](https://github.com/gh9869827/fifo-tool-airlock-model-env). See its [README](https://github.com/gh9869827/fifo-tool-airlock-model-env/blob/main/README.md) for setup instructions.

Install the DSL module in editable mode in a separate virtual environment:

```bash
# Create and activate a new virtual environment
python3 -m venv fifo_env_root
cd fifo_env_root
source bin/activate

# Clone the DSL module into the virtual environment directory
git clone https://github.com/gh9869827/fifo-dev-dsl.git
cd fifo-dev-dsl

# Run the setup script
./setup.sh
```

Requires Python 3.10 or later.

---

## 🧩 Modules

### `fifo_dev_dsl.dia`

A lightweight, composable DSL engine that turns natural language into **goal-driven intent execution**, with *interactive* **slot resolution** and **recoverable error handling** through direct user interaction.

**Features:**

- 🧠 **Intent invocation** using symbolic and composable function calls like `retrieve_screws(count=2, length=ASK(...))`
  - ⚙️ **Tool integration** via function-to-intent mapping that invokes Python functions defined by the user  
  - 📡 **Query sources** provide runtime context the model uses to answer questions or fill missing slots  
  - 🔒 Tool-calling with a **security focus**, using only explicitly registered functions with type-checked arguments, securely parsed and cast. There is no dynamic evaluation or arbitrary code execution (i.e., no use of Python's built-in `eval()` or `exec()`).
- 💬 **Dialog-based resolution** using `ASK(...)` and `QUERY_USER(...)` for interactive prompts. Errors are handled with an `IntentRuntimeErrorResolver` node, automatically injected when recoverable runtime errors are detected.
- 🔁 **Slot propagation and reuse** with `PROPAGATE_SLOT(...)` and `SAME_AS_PREVIOUS_INTENT()`  
- 🧪 **Deterministic evaluation** with traceable logic  

**Example – High-Level Usage**

The snippet below, from [`demo/robot_arm.py`](fifo_dev_dsl/dia/demo/robot_arm.py), demonstrates how to use the module end-to-end: from setting up the runtime context to executing a DSL-driven tool invocation.

```python
# User input
user_input = 'Give me two screws from the inventory'

# Set up the runtime context with available tools and query sources
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.dia.resolution.resolver import Resolver
from fifo_dev_dsl.dia.runtime.evaluator import Evaluator
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationStatus

# Assume we have a tool class like RobotArm providing methods and inventory
robot = RobotArm()

runtime_context = LLMRuntimeContext(
    container_name="phi",  # or any model label you use
    intent_sequencer_adapter="dia-intent-sequencer-robot-arm-adapter",
    tools=[
        robot.retrieve_screw,
        robot.initialize_components,
        robot.organize,
        robot.shutdown,
    ],
    query_sources=[
        robot.get_inventory
    ]
)

# Translate user input to parsed DSL:
# user_input: Give me two screws from the inventory
# user_input → LLM adapter → DSL string → parser → DSL object
# DSL object: retrieve_screw(count=2, length=ASK("What length do you need?"))
resolver = Resolver(runtime_context, prompt=user_input)

# Evaluate the DSL tree with interactive error recovery
while True:

    # Step 1: Resolve DSL (slot filling or follow-up questions).
    #         During this step, the user will be asked "What length do you need?"
    #         and will provide a value.
    resolver.fully_resolve_in_text_mode()
    dsl_elements = resolver.dsl_elements

    # Step 2: Evaluate the resolved DSL.
    #         This may succeed or fail, for example, if the screw inventory is insufficient.
    evaluator = Evaluator(runtime_context, dsl_elements)
    outcome = evaluator.evaluate()

    # Step 3: Recovery loop if needed
    if outcome.status != EvaluationStatus.ABORTED_RECOVERABLE:
        break  # Success or unrecoverable failure

    resolver = Resolver(runtime_context, dsl=dsl_elements)
```

This demonstrates how `dia` handles interactive slots (such as `ASK(...)`) and re-evaluates the DSL tree until resolution completes or fails with an unrecoverable error.

📺 Watch the demo video below:  
[![Demo Video](https://img.youtube.com/vi/wbdLcn9Wizc/hqdefault.jpg)](https://www.youtube.com/watch?v=wbdLcn9Wizc)

See [`dia/README.md`](fifo_dev_dsl/dia/README.md) for syntax, resolution phases, and evaluator logic.

---

### `fifo_dev_dsl.domain_specific` DSLs

- [`mini_date_converter_dsl`](fifo_dev_dsl/domain_specific/mini_date_converter_dsl/README.md): translates natural language date and time expressions into Python `datetime` objects.
- [`mini_recurrence_converter_dsl`](fifo_dev_dsl/domain_specific/mini_recurrence_converter_dsl/README.md): translates recurring schedule expressions into structured recurrence rules.

See [`domain_specific/README.md`](fifo_dev_dsl/domain_specific/README.md) for an overview of supported DSL subpackages.

---

## ✅ License

MIT — see [LICENSE](LICENSE).

---

## 📄 Disclaimer

This project is not affiliated with or endorsed by Hugging Face or the Python Software Foundation.  
It builds on their open-source technologies under their respective licenses.
