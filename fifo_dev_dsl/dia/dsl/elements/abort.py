from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fifo_dev_dsl.dia.dsl.elements.base import DslBase

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext


@dataclass
class Abort(DslBase):
    """
    A control directive that halts further intent resolution and execution.

    When present in the DSL tree, the resolver returns
    `ResolutionResult.ABORT` and discards any subsequent nodes in the 
    execution branch, and moves to the execution of the next intent.
    This node is typically injected when the user cancels the conversation
    or explicitly requests to stop processing.
    """

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the Abort node.

        Returns:
            str:
                The fixed DSL syntax for aborting, always returns 'ABORT()'.
        """
        return "ABORT()"

    def is_resolved(self) -> bool:
        """
        Always returns False, as this node halts resolution and should not appear
        in a fully resolved DSL tree.

        This node triggers an immediate abort of the current intent resolution path.
        It is typically injected during user cancellation or early termination and
        signals that no further evaluation should occur.

        The Resolver in the resolution module guarantees that this node is removed
        before the final DSL tree is considered resolved.
        
        Returns:
            bool:
                False, indicating the node must be intercepted and removed before evaluation.
        """
        return False

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
        value_type: MiniDocStringType | None = None,
    ) -> Any:
        """Raise a :class:`RuntimeError` because Abort nodes are unresolved.

        Abort elements signal that an intent sequence should be terminated.
        They are removed by the resolver before a fully resolved DSL tree is
        produced. Encountering one during evaluation therefore indicates
        unresolved state.

        Raises:
            RuntimeError: Always raised with the message
                ``"Unresolved DSL node: Abort"``.
        """

        raise RuntimeError(f"Unresolved DSL node: {self.__class__.__name__}")
