from __future__ import annotations
from typing import Any, TYPE_CHECKING
from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase
from fifo_dev_dsl.dia.dsl.elements.base import DslBase

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
    from fifo_dev_dsl.dia.dsl.elements.intent import Intent
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class ReturnValue(DSLValueBase):
    """
    Use the result of another intent as an inline value.

    `ReturnValue` wraps an :class:`Intent`, allowing its result to be embedded
    as a value in another intent's slot. This supports nested execution where
    the output of one tool feeds directly into another.

    For example, retrieving the location of a box and then passing that location
    to a pickup intent can be expressed as:

        pickup(location=ReturnValue(get_box_location()))

    Attributes:
        intent (Intent):
            The intent whose evaluated result will be used as the slot value.
    """

    intent: Intent

    def eval(self,
             runtime_context: LLMRuntimeContext) -> Any:
        """
        Evaluate the wrapped intent and return its value.

        This node delegates to the embedded intent. If the nested intent
        contains unresolved placeholders, evaluation fails with a RuntimeError.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context providing tool access, query sources, and runtime helpers.


        Returns:
            Any:
                The value produced by the nested intent evaluation.

        Raises:
            RuntimeError: If the nested intent is not resolved.
        """

        return self.intent.eval(runtime_context)

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of a return value.

        This wraps an intent as an inline sub-expression, enabling nested calls like:
        `multiply(a=4, b=add(a=2, b=3))`.

        Returns:
            str:
                A string representation of the nested intent.
        """
        return self.intent.to_dsl_representation()

    def pretty_print_dsl(self, indent: int = 0) -> None:
        prefix = "  " * indent
        print(f"{prefix}{self.__class__.__name__}()")
        self.intent.pretty_print_dsl(indent + 1)

    def get_children(self) -> list[DslBase]:
        return [self.intent]
