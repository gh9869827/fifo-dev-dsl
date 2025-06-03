from __future__ import annotations
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass

from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class Value(DSLValueBase):

    value: Any

    def represent_content_as_text(self) -> str | None:
        """
        Represent the content of this DSL node as a string.

        Returns:
            str | None:
                The value.
        """
        return self.value

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: MiniDocStringType | None = None) -> Any:

        if value_type is None:
            raise RuntimeError("Missing expected type for evaluation of Value")

        return value_type.cast(self.value)
