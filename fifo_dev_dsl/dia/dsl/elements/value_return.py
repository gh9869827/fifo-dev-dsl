from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import Any

from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
from fifo_dev_dsl.dia.dsl.elements.intent import Intent
from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class ReturnValue(DSLValueBase):

    intent: Intent

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: MiniDocStringType | None = None) -> Any:

        if value_type is None:
            raise RuntimeError("Missing expected type for evaluation of ReturnValue")

        return self.intent.eval(runtime_context, value_type)
