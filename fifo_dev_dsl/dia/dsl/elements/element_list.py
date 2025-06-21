from __future__ import annotations
from typing import TYPE_CHECKING, Any


from fifo_dev_dsl.dia.dsl.elements.base import DslBase, make_dsl_container

if TYPE_CHECKING:  # pragma: no cover
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
    ) -> Any:
        """
        Evaluate each child and return a list of their values.

        This node delegates evaluation to each of its children. If any child is
        unresolved, a RuntimeError will be raised during that child's evaluation.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context providing tool access, query sources, and runtime helpers.

        Returns:
            list[Any]:
                The list of evaluated child values.

        Raises:
            RuntimeError: If any child is not resolved.
        """

        return [child.eval(runtime_context) for child in self.get_items()]
