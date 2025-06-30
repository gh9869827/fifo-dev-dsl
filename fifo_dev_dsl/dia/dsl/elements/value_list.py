from __future__ import annotations
from typing import TYPE_CHECKING, Any

from fifo_dev_dsl.dia.dsl.elements.base import make_dsl_container
from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

class ListValue(make_dsl_container(DSLValueBase), DSLValueBase):
    """
    A DSL node representing a list of values to be evaluated as a Python list.

    Each item in this container is a `DSLValueBase` element. During
    evaluation, all child nodes are resolved in order, and their results are
    collected into a standard Python list.

    This enables compound expressions such as passing multiple arguments,
    defining sets of options, or grouping multiple return values.

    Example:
        Used inside a slot:
            sum(v=ListValue([Value(1), Value(2), Value(3)])) → 6
    """

    def eval(self,
             runtime_context: LLMRuntimeContext) -> Any:
        """
        Evaluate each child value and return a list of results.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context providing tool access, query sources, and runtime helpers.


        Returns:
            list[Any]:
                The list of evaluated child values.

        Raises:
            RuntimeError: If any child is not resolved.
        """
        return [e.eval(runtime_context) for e in self.get_items()]

    async def eval_async(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Asynchronously evaluate each child value and return a list.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context providing tool access, query sources, and runtime helpers.

        Returns:
            list[Any]:
                The list of evaluated child values.

        Raises:
            RuntimeError: If any child is not resolved.
        """

        return [
            await e.eval_async(runtime_context)
            for e in self.get_items()
        ]

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of this list node.

        Each item's DSL representation is joined into a string like `[1, 2, "x"]`.

        Returns:
            str:
                The formatted DSL list representation.
        """
        return "[" + ", ".join(item.to_dsl_representation() for item in self.get_items()) + "]"
