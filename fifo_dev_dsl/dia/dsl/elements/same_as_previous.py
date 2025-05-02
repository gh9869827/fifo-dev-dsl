from dataclasses import dataclass
from common.llm.dia.dsl.elements.value_base import DSLValueBase


@dataclass
class SameAsPreviousIntent(DSLValueBase):
    pass

    # todo but pass context of previous intent
    # def get_resolved_value_as_text(self) -> str:
    #     return self.value
