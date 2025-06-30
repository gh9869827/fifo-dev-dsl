from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.elements.element_list import ListElement

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext


@dataclass
class AbortWithNewDsl(DslBase):
    """
    Abort the current resolution path and replace it with a new DSL subtree.

    Like `Abort`, this node signals that the current resolution path is no longer valid.
    However, instead of terminating execution, it returns `ResolutionResult.NEW_DSL_NODES`
    with the provided `new_dsl`, allowing the resolver to continue with a new intent sequence.

    This is useful for graceful redirection—for example, when a requested item is unavailable
    and an alternative path should be proposed.

    Attributes:
        new_dsl (ListElement):
            DSL elements that will replace the aborted intent subtree.
    """

    new_dsl: ListElement

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the AbortWithNewDsl node.

        Returns:
            str:
                A string in DSL syntax replacing the current element with new ones,
                e.g., 'ABORT_WITH_NEW_INTENTS([foo(), bar(x=2)])'.
        """
        return f"ABORT_WITH_NEW_INTENTS({self.new_dsl.to_dsl_representation()})"

    def is_resolved(self) -> bool:
        """
        Always returns False, as this node represents an unresolved redirection.

        This element aborts the current resolution path and must be replaced
        by its `new_dsl` before evaluation can proceed. It serves as a signal
        that resolution is incomplete and should not remain in a fully resolved
        DSL tree.

        The Resolver in the resolution module guarantees that this node is removed
        before the final DSL tree is considered resolved.

        Returns:
            bool:
                False, indicating this node requires further transformation.
        """
        return False

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Raise a :class:`RuntimeError` for unresolved abort redirections.

        AbortWithNewDsl nodes should be replaced with `new_dsl` during
        resolution. If one remains during evaluation, a `RuntimeError` is
        raised to signal unresolved state.

        Raises:
            RuntimeError: Always raised with the message
                Unresolved DSL node: AbortWithNewDsl.
        """

        raise RuntimeError(f"Unresolved DSL node: {self.__class__.__name__}")

    async def eval_async(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Asynchronously raise a :class:`RuntimeError` for unresolved abort
        redirections.

        AbortWithNewDsl nodes should be replaced with ``new_dsl`` during
        resolution. If one remains during evaluation, a ``RuntimeError`` is
        raised to signal unresolved state.

        Raises:
            RuntimeError: Always raised with the message
                Unresolved DSL node: AbortWithNewDsl.
        """

        raise RuntimeError(f"Unresolved DSL node: {self.__class__.__name__}")
