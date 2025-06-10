from __future__ import annotations
from abc import ABC

from fifo_dev_dsl.dia.dsl.elements.base import DslBase

class DSLValueBase(DslBase, ABC):
    """
    Abstract base class for all DSL nodes that evaluate to a runtime value.

    Any node that can produce a concrete value during execution—such as a constant,
    list, fuzzy descriptor, or computed result—should inherit from this class.
    Subclasses must implement :py:meth:`DslBase.eval`, which returns a Python
    value suitable for use as a tool's slot value or in composed expressions.

    Example use cases include:
        - Literal values (e.g., `Value("12mm")`)
        - Nested expressions (e.g., `ReturnValue(...)`)
        - Collections (e.g., `ListValue([...])`)
    """
