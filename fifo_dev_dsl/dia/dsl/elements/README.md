# DSL Elements

This module defines the building blocks of the DSL used in the `dia` engine. Each class in this directory represents a specific type of node in the DSL tree. Nodes are composed, resolved, and evaluated to drive the decision-making flow and execution logic of a goal-directed agent.

## Overview

The DSL supports:

- **Intent execution** with named parameters
- **Slot resolution**, including interactive queries and LLM inference
- **Control flow** such as aborts and redirection
- **Value representation**, including concrete and fuzzy quantities
- **Containers** for grouping and propagating values across nodes

Each node is a Python class with a method for rendering DSL syntax, hooks for evaluation or resolution, and support for visualization or inspection.

## Node Types

### Intent Nodes

- **`Intent`**  
  Represents a tool invocation with structured arguments. Arguments are defined as `Slot` instances. Example:  
  `retrieve_screw(count=2, length=12)`

- **`IntentEvaluatedSuccess`**  
  Marks that an `Intent` has been executed successfully. Stores the result to prevent re-execution in recovery scenarios.

- **`IntentRuntimeErrorResolver`**  
  Wraps a failing `Intent` with its error message. Used to initiate user-guided recovery or replan logic.

### Value Nodes

- **`Value`**  
  A literal constant (string, number, etc.) already known and resolved. Always returns a fixed Python value.

- **`FuzzyValue`**  
  A vague quantity descriptor (e.g., `"a few"`) that gets mapped to a number via internal normalization.

- **`SameAsPreviousIntent`**  
  Refers to the value of the same slot in the previously executed `Intent`.

- **`ReturnValue`**  
  Wraps another `Intent` whose result will be used as a slot argument. Enables nested calls.

- **`ListValue`**  
  Represents a list of DSL values. Evaluated into a standard Python list.

### Interactive Resolution

- **`ASK`**  
  Prompts the user for a missing slot value. Remains unresolved until answered.

- **`QueryFill`**  
  Uses LLM-based reasoning to infer a missing slot value from a structured query. Enables context-aware inference.

- **`QueryGather`**  
  Collects multiple interdependent values by issuing a single LLM-powered query. Typically used to prepare full intent generation.

- **`QueryUser`**  
  Captures a question *from* the user (e.g., "How many screws are in inventory?"). Treated as a top-level information query, not a tool invocation.

### Control Flow

- **`Abort`**  
  Immediately halts the current resolution path. Used to cancel or block further processing.

- **`AbortWithNewDsl`**  
  Stops the current path and replaces it with a new DSL subtree. Useful for graceful fallback or rerouting.

### Containers

- **`Slot`**  
  Maps a parameter name to a DSL value. Always contains a single child node.

- **`PropagateSlots`**  
  Carries slot values forward from a previous `Intent` when the user provides more information than was requested.

- **`ListElement`**  
  A container for a heterogeneous list of DSL nodes. Used to group and sequence expressions or tool calls.

---

## Example: DSL Flow

```python
Intent(
    name="retrieve_screw",
    slots=[
        Slot("count", FuzzyValue("a few")),
        Slot("length", QueryFill("shortest available screw")),
    ]
)
```