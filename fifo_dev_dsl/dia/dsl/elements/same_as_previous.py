from __future__ import annotations
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass
from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class SameAsPreviousIntent(DSLValueBase):
    """
    Reuse the value of the same slot from the immediately preceding intent.

    This node acts as a placeholder for a slot value that should be copied
    from the previous intent. During resolution, it is replaced with the
    value from the same-named slot in the most recent intent.

    This is useful when the user makes a follow-up request that implicitly refers
    to a previously specified value — such as saying "the same size" or "again".

    Example:
        Input:
            "Give me two screws of 12mm, and then four of the same size."
        Output:
            retrieve_screw(count=2, length=12),
            retrieve_screw(count=4, length=SAME_AS_PREVIOUS_INTENT())
    """

    def eval(self,
             runtime_context: LLMRuntimeContext) -> Any:
        """
        Retrieve the value of the same-named slot from the previous intent.

        During evaluation, this node will return the value of the corresponding
        slot from the most recently evaluated intent.

        This method is not yet implemented and currently raises a NotImplementedError.

        Raises:
            NotImplementedError: Always, until evaluation logic is implemented.
        """
        raise NotImplementedError()

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the SameAsPreviousIntent node.

        Returns:
            str:
                The fixed DSL syntax, always returns 'SAME_AS_PREVIOUS_INTENT()'.
        """
        return "SAME_AS_PREVIOUS_INTENT()"
