from __future__ import annotations
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass

from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

_FUZZY_TO_NUMERIC: dict[str, int] = {
    "a couple": 2,
    "couple": 2,
    "a few": 3,
    "few": 3,
    "several": 5,
    "many": 8,
    "a dozen": 12,
    "dozen": 12,
    "dozens": 24,
}


@dataclass
class FuzzyValue(DSLValueBase):

    value: str

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

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
        value_type: MiniDocStringType | None = None
    ) -> Any:
        if value_type is None:
            raise RuntimeError("Missing expected type for evaluation of FuzzyValue")

        normalized = self.value.lower().strip()
        if normalized in _FUZZY_TO_NUMERIC:
            return value_type.cast(_FUZZY_TO_NUMERIC[normalized], allow_scalar_to_list=True)

        raise ValueError(f"Unrecognized fuzzy value: {self.value!r}")
