from __future__ import annotations
from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.elements.element_list import ListElement


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
