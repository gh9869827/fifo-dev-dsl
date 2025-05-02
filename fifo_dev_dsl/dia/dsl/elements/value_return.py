from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from dataclasses import dataclass
from typing import Any

from common.introspection.docstring import MiniDocStringType
from common.llm.dia.dsl.elements.intent import Intent
from common.llm.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:
    from common.llm.dia.runtime.context import LLMRuntimeContext

@dataclass
class ReturnValue(DSLValueBase):

    intent: Intent

    def get_resolved_value_as_text(self) -> str:
        # return self.value
        assert False, "TODO"

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: Optional[MiniDocStringType] = None) -> Any:

        if value_type is None:
            raise RuntimeError("Missing expected type for evaluation of ReturnValue")

        return self.intent.eval(runtime_context, value_type)
