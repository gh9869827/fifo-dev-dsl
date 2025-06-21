from __future__ import annotations
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class Value(DSLValueBase):
    """
    A literal constant representing a fully known value in the DSL tree.

    `Value` nodes wrap concrete Python objects such as strings, numbers, or booleans.
    Unlike other DSL nodes that may require resolution, a `Value` is always considered
    resolved and can be evaluated directly.

    During evaluation, it is cast to the expected type by the execution layer.

    Attributes:
        value (Any):
            The underlying Python object to treat as a resolved constant.

    Examples:
        Value("12")       # A string constant
        Value(42)         # An integer constant
        Value(True)       # A boolean constant
    """

    value: Any

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of a single value.

        All string values are quoted. Non-string values (e.g., numbers)
        are emitted as-is.

        Returns:
            str:
                A clean DSL representation:
                - Strings appear quoted, e.g., `"hello"`
                - Numbers and other values appear unquoted, e.g., `42`
        """
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return f'{self.value}'

    def eval(self,
             runtime_context: LLMRuntimeContext) -> Any:
        """
        Return the stored constant value without casting.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context (not used in this node).


        Returns:
            Any:
                The stored value as-is.
        """
        return self.value
