
from dataclasses import dataclass
from common.llm.dia.dsl.elements.value_base import DSLValueBase


@dataclass
class FuzzyValue(DSLValueBase):

    value: str

    def represent_content_as_text(self) -> str | None:
        """
        Represent the content of this DSL node as a string.

        Returns:
            str | None:
                The fuzzy value.
        """
        return self.value

