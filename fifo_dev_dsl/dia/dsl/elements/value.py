from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

from dataclasses import dataclass

from common.introspection.docstring import MiniDocStringType
from common.llm.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:
    from common.llm.dia.runtime.context import LLMRuntimeContext

@dataclass
class Value(DSLValueBase):
    value: Any

    def get_resolved_value_as_text(self) -> str:
        return self.value

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: Optional[MiniDocStringType] = None) -> Any:

        if value_type is None:
            raise RuntimeError("Missing expected type for evaluation of Value")

        return value_type.cast(self.value)
