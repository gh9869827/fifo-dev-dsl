from __future__ import annotations
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass
from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:
    from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class SameAsPreviousIntent(DSLValueBase):

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: MiniDocStringType | None = None) -> Any:
        raise NotImplementedError()

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the SameAsPreviousIntent node.

        Returns:
            str:
                The fixed DSL syntax, always returns 'SAME_AS_PREVIOUS_INTENT()'.
        """
        return "SAME_AS_PREVIOUS_INTENT()"
