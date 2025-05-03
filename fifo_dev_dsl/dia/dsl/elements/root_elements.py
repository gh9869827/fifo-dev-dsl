from __future__ import annotations
from typing import TYPE_CHECKING, Any


from common.introspection.docstring import MiniDocStringType
from common.llm.dia.dsl.elements.base import DslBase, DslContainerBase

if TYPE_CHECKING:
    from common.llm.dia.runtime.context import LLMRuntimeContext

class RootElements(DslContainerBase[DslBase]):

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: MiniDocStringType | None = None) -> Any:

        if value_type is None:
            raise RuntimeError("Missing expected type for evaluation of ListValue")

        if (inner_type := value_type.is_list()) is not None:
            return [e.eval(runtime_context, inner_type) for e in self._items]

        raise ValueError(f"Invalid type for ListValue eval(): {value_type}")
