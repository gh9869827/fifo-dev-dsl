from __future__ import annotations
from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.base import DslBase


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
