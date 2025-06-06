from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from fifo_dev_dsl.dia.dsl.elements.base import DslBase

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

class DSLValueBase(DslBase, ABC):
    """
    Abstract base class for all DSL nodes that evaluate to a runtime value.

    Any node that can produce a concrete value during execution—such as a constant,
    list, fuzzy descriptor, or computed result—should inherit from this class.
    Subclasses must implement the `eval()` method, which returns a Python value
    suitable for use as a tool's slot value or in composed expressions.

    Example use cases include:
        - Literal values (e.g., `Value("12mm")`)
        - Nested expressions (e.g., `ReturnValue(...)`)
        - Collections (e.g., `ListValue([...])`)
    """

    @abstractmethod
    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: MiniDocStringType | None = None) -> Any:
        """
        Compute and return the runtime value for this DSL element.

        Args:
            runtime_context (LLMRuntimeContext):
                The runtime environment available during evaluation.
                This includes:
                - Registered tools available for intent resolution
                - Domain-specific query sources (e.g., item inventory, memory store)
                - LLM prompt templates for all resolution and interaction phases

            value_type (MiniDocStringType | None):
                Optional expected type for the result, used to guide value interpretation.

        Returns:
            Any:
                The resolved Python value that this DSL node evaluates to.
                This value can be passed as a tool argument or used in other composed expressions.
        """
