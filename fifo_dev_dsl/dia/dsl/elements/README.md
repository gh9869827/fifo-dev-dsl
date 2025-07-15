# `fifo_dev_dsl.dia.dsl.elements`: DSL Elements

This module defines the core building blocks of the DSL used by the `dia` engine.  
Each class in this directory represents a specific type of node in the DSL tree.

These nodes are combined to form interpretable trees that describe how the agent should make decisions, ask questions, handle runtime errors, or perform actions.  
The `dia` runtime resolves and evaluates these trees step by step to drive goal-directed behavior.

---

## 🎯 Project Status & Audience

🚧 **Work in Progress** — Part of the **`fifo-dev-dsl`** project, currently in **early development**. 🚧

This is a personal project developed and maintained by a solo developer.  
Contributions, ideas, and feedback are welcome, but development is driven by personal time and priorities.

`fifo-dev-dsl` is designed to support other `fifo-*` projects developed by the author.  
It is provided for **individual developers** interested in experimenting with DSL-driven natural language interpretation.

No official release or pre-release has been published yet. The code is provided for **preview and experimentation**.  
**Use at your own risk.**

---

## Overview

The DSL supports:

- **Intent execution** with named parameters  
- **Slot resolution** via user interaction or interactive queries powered by LLM-based inference
- **Control flow** with support for aborts and redirection  
- **Value representation** for both concrete and fuzzy quantities  
- **Containers** for grouping values and propagating information across nodes

Each node is implemented as a Python class, with methods for rendering DSL syntax, resolving or evaluating behavior, and supporting visualization or inspection.

---

## Node Types

### Intent Nodes

- **`Intent`**  
  Represents a tool invocation with structured arguments. Arguments are defined as `Slot` instances.  
  Example: `retrieve_screw(count=2, length=12)`

- **`IntentEvaluatedSuccess`**  
  Wraps a successfully executed `Intent` along with its return value. Used to prevent re-execution during recovery or downstream evaluation.

- **`IntentRuntimeErrorResolver`**  
  Wraps a failed `Intent` along with its error message. Used to trigger user-guided recovery or replan logic.

### Value Nodes

- **`Value`**  
  A literal constant (string, number, etc.) that is already known and resolved. Always returns a fixed Python value.

- **`FuzzyValue`**  
  A fuzzy quantity descriptor (e.g., `"a few"`) that gets mapped to a number via internal normalization.

- **`SameAsPreviousIntent`**  
  Refers to the value of the same slot in the previously executed `Intent`.

- **`ReturnValue`**  
  Wraps another `Intent`, using its result as the value for a slot. Enables nested tool calls.

- **`ListValue`**  
  Represents a list of DSL values. Evaluates to a standard Python list.

### Interactive Resolution

- **`ASK`**  
  Prompts the user for a missing slot value. Remains unresolved until a response is provided.

- **`QueryFill`**  
  Uses LLM-based reasoning to infer a missing slot value from a structured query against runtime information. Enables context-aware inference.

- **`QueryGather`**  
  Resolves multiple interdependent slots in a single LLM-powered query over the runtime context. Used when values cannot be inferred independently, requiring atomic reasoning before re-evaluating the intent.

- **`QueryUser`**  
  Captures a question *from* the user (e.g., "How many screws are in inventory?"). Treated as a top-level information query, not a tool invocation. Carries over context from the previous exchange while building the query.

### Control Flow

- **`Abort`**  
  Immediately halts the current resolution path. Used to cancel or block further processing.

- **`AbortWithNewDsl`**  
  Halts the current path and replaces it with a new DSL subtree. Useful for graceful fallback, redirection, or replanning.

### Containers

- **`Slot`**  
  Maps a parameter name to a DSL value. Always wraps a single child node.

- **`PropagateSlots`**  
  Forwards slot values from a previous `Intent` when the user provides more information than expected.

- **`ListElement`**  
  A container for a heterogeneous list of DSL nodes. Used to group or sequence expressions and tool calls.

---

## Example: DSL Flow

The following DSL tree represents a tool call to `retrieve_screw`, where the `count` is inferred from a fuzzy descriptor and the `length` is resolved using an LLM query over runtime information.

```python
Intent(
    name="retrieve_screw",
    slots=[
        Slot("count", FuzzyValue("a few")),
        Slot("length", QueryFill("shortest available screw")),
    ]
)
```

---

## License

MIT — see [LICENSE](../../../../LICENSE).
