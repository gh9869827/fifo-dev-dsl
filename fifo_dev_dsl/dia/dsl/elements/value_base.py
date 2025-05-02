
from common.llm.dia.dsl.elements.base import DslBase


class DSLValueBase(DslBase):

    def get_resolved_value_as_text(self) -> str:
        raise NotImplementedError("get_resolved_value_as_text is not implemented")
