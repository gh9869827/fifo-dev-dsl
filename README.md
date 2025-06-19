# 🧠 fifo-dev-dsl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg) 
![Test Status](https://github.com/gh9869827/fifo-dev-dsl/actions/workflows/test.yml/badge.svg)

A suite of **domain-specific languages (DSLs)** for interpreting natural language and converting it into structured, executable logic across the `fifo-*` ecosystem.

At the center is **`dia`** — A DSL for interactive agents that translates user intent into tool calls and can resolve missing information or runtime errors through user interaction when needed. Additional modules support specialized parsing tasks, all fully tested under `tests/`.

---

## 🧩 Modules

### `fifo_dev_dsl.dia`

A lightweight, composable DSL engine that turns natural language into **goal-driven intent execution**, with *interactive* **slot resolution** and **recoverable error handling** through direct user interaction.

**Features:**

- 🧠 **Intent invocation** using symbolic and composable function calls like `retrieve_screws(count=2, length=ASK(...))`
  - ⚙️ **Tool integration** via function-to-intent mapping that executes real Python code  
  - 📡 **Query sources** provide runtime context the model uses to answer questions or fill missing slots  
- 💬 **Dialog-based resolution** with `ASK(...)`, `QUERY_USER(...)`, `QUERY_FILL(...)`, and `QUERY_GATHER(...)`  
- 🔁 **Slot propagation and reuse** with `PROPAGATE_SLOT(...)` and `SAME_AS_PREVIOUS_INTENT()`  
- 🧪 **Deterministic evaluation** with traceable logic  

**Example – High-Level Usage**

The snippet below, from [`demo/robot_arm.py`](fifo_dev_dsl/dia/demo/robot_arm.py), demonstrates how to use the module end-to-end: from setting up the runtime context to executing a DSL-driven tool invocation.

```python
# DSL input string (could also be resolved from natural language)
user_input = 'Give me two screws from the inventory'

# Setup the runtime context with available tools and query sources
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.dia.resolution.resolver import Resolver
from fifo_dev_dsl.dia.runtime.evaluator import Evaluator
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationStatus

# Assume we have a tool class like RobotArm providing methods and inventory
robot = RobotArm()

runtime_context = LLMRuntimeContext(
    container_name="phi",  # or any model label you use
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

# Translate the user input into DSL using LLM-based intent and slot resolution
# (e.g., retrieve_screw(count=2, length=ASK("what length do you need?")))
resolver = Resolver(runtime_context, prompt=user_input)

# Evaluate the DSL tree with interactive error recovery
while True:

    # Step 1: resolve DSL (slot filling or follow-up questions)
    resolver.fully_resolve_in_text_mode()
    dsl_elements = resolver.dsl_elements
    dsl_elements.pretty_print_dsl()

    # Step 2: evaluate resolved DSL (may succeed or fail)
    evaluator = Evaluator(runtime_context, dsl_elements)
    outcome = evaluator.evaluate()

    # Step 3: recovery loop if needed
    if outcome.status != EvaluationStatus.ABORTED_RECOVERABLE:
        break  # success or unrecoverable failure

    resolver = Resolver(runtime_context, dsl=dsl_elements)
```

This shows how `dia` handles interactive slots (such as `ASK(...)`) and re-evaluates the DSL tree until resolution completes or fails with an unrecoverable error.

See [`dia/README.md`](fifo_dev_dsl/dia/README.md) for syntax, resolution phases, and evaluator internals.

---

### `fifo_dev_dsl.domain_specific` DSLs

- [`mini_date_converter_dsl`](fifo_dev_dsl/domain_specific/mini_date_converter_dsl/README.md) – parses natural language date and time expressions into Python `datetime` objects.
- [`mini_recurrence_converter_dsl`](fifo_dev_dsl/domain_specific/mini_recurrence_converter_dsl/README.md) – translates recurring schedule expressions into structured recurrence rules.

See [`domain_specific/README.md`](fifo_dev_dsl/domain_specific/README.md) for an overview of supported DSL subpackages.

---

## ✅ License

MIT — see [LICENSE](LICENSE).

---

## 📄 Disclaimer

This project is not affiliated with or endorsed by Hugging Face or the Python Software Foundation.  
It builds on their open-source technologies under their respective licenses.
