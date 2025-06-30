from __future__ import annotations
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:  # pragma: no cover
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
    """
    A fuzzy descriptor representing an approximate quantity in natural language.

    This node captures vague expressions like "a few" or "many" and maps them to
    concrete numeric values at evaluation time. It enables the DSL to gracefully
    interpret imprecise user input by referencing the internal `_FUZZY_TO_NUMERIC` table.

    Unlike `Value`, which stores exact constants, `FuzzyValue` supports common
    quantity idioms that users might naturally provide. This allows for a more
    conversational experience during resolution.

    Attributes:
        value (str):
            A fuzzy quantity phrase such as "a couple" or "several".

    Example:
        FuzzyValue("a few") → 3 (after evaluation)
    """

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
    ) -> Any:
        """
        Map the fuzzy value to a numeric value.

        During evaluation, the fuzzy value must be matched against known values.
        If no match is found, a `ValueError` is raised.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context (not used in this node).

        Returns:
            Any:
                Numeric representation corresponding to the fuzzy value.

        Raises:
            ValueError: If the fuzzy value is unknown.
        """
        normalized = self.value.lower().strip()
        if normalized in _FUZZY_TO_NUMERIC:
            return _FUZZY_TO_NUMERIC[normalized]

        raise ValueError(f"Unrecognized fuzzy value: {self.value!r}")

    async def eval_async(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Asynchronously map the fuzzy value to a numeric value.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context (not used in this node).

        Returns:
            Any:
                Numeric representation corresponding to the fuzzy value.

        Raises:
            ValueError: If the fuzzy value is unknown.
        """

        _ = runtime_context
        normalized = self.value.lower().strip()
        if normalized in _FUZZY_TO_NUMERIC:
            return _FUZZY_TO_NUMERIC[normalized]
        raise ValueError(f"Unrecognized fuzzy value: {self.value!r}")
