from __future__ import annotations
from typing import TYPE_CHECKING, Any


from common.introspection.docstring import MiniDocStringType
from common.llm.dia.dsl.elements.base import DslBase, DslContainerBase

if TYPE_CHECKING:
    from common.llm.dia.runtime.context import LLMRuntimeContext

class ListElement(DslContainerBase[DslBase]):

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: MiniDocStringType | None = None) -> Any:

        if value_type is not None:
            raise RuntimeError("value_type is ignored by ListElement")

        return [e.eval(runtime_context) for e in self._items]
