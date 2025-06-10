from __future__ import annotations
from fifo_dev_dsl.dia.dsl.elements.base import DslBase, make_dsl_container


from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext


class ListElement(make_dsl_container(DslBase)):
    """
    A container node that holds a sequence of heterogeneous DSL elements.

    This class is used when multiple DSL nodes need to be grouped and treated
    as a single subtree. It is commonly used in constructs like `AbortWithNewDsl`
    to replace the current resolution path with an ordered set of new DSL nodes.

    Example:
        Used in an abort redirect:
            AbortWithNewDsl(
                ListElement([
                    Intent(name="action_1", slots=[]),
                    Intent(name="action_2", slots=[]),
                ])
            )
    """

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
        value_type: MiniDocStringType | None = None,
    ) -> Any:
        """
        Evaluate each child and return a list of their values.

        This node delegates evaluation to each of its children. If any child is
        unresolved, a RuntimeError will be raised during that child's evaluation.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context providing tool access, query sources, and runtime helpers.

            value_type (MiniDocStringType | None):
                Optional expected return type. When provided and represents
                `list[T]`, each child is evaluated using `T` as its expected type.

        Returns:
            list[Any]:
                The list of evaluated child values.

        Raises:
            RuntimeError: If any child is not resolved.
        """

        inner = value_type.is_list() if value_type is not None else None
        return [child.eval(runtime_context, inner) for child in self.get_items()]
