from dataclasses import dataclass
from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase


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

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the fuzzy value.

        Formats the value using the DSL fuzzy marker syntax: F("...").
        Raises an error if the value contains double or single quotes,
        which are disallowed by the DSL specification.

        Returns:
            str:
                The fuzzy value wrapped in the DSL marker, e.g., `F("many")`.

        Raises:
            ValueError:
                If the value contains disallowed quote characters.
        """
        if '"' in self.value or "'" in self.value:
            raise ValueError("Fuzzy value contains quotes, which are not allowed.")
        return f'F("{self.value}")'
