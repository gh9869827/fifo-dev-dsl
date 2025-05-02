
from dataclasses import dataclass
from common.llm.dia.dsl.elements.value_base import DSLValueBase


@dataclass
class FuzzyValue(DSLValueBase):

    value: str

    def get_resolved_value_as_text(self) -> str:
        return self.value
