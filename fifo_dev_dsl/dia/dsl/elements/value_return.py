from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import Any

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
             runtime_context: LLMRuntimeContext,
             value_type: MiniDocStringType | None = None) -> Any:
        """Evaluate the wrapped intent and return its value.

        If the ``ReturnValue`` node contains unresolved placeholders, evaluation
        fails with a :class:`RuntimeError`. Otherwise, it delegates to the
        embedded intent and casts the result to ``value_type``.

        Args:
            runtime_context: Execution context used to evaluate the inner
                intent.
            value_type: Expected type of the returned value.

        Returns:
            Any: The value produced by the nested intent evaluation.
        """

        if not self.is_resolved():
            raise RuntimeError(
                f"Unresolved DSL node: {self.__class__.__name__}"
            )

        if value_type is None:
            raise RuntimeError(
                "Missing expected type for evaluation of ReturnValue"
            )

        return self.intent.eval(runtime_context, value_type)

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
