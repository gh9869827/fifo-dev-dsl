from __future__ import annotations
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:
    from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class Value(DSLValueBase):

    value: Any

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of a single value.

        Attempts to coerce string values into numbers for unquoted output.
        Falls back to quoted string representation if coercion fails.
        This ensures DSL output like `42` or `"hello"` rather than always using repr.

        Returns:
            str:
                A clean, type-aware DSL representation:
                - numerics appear unquoted,
                - strings appear quoted.
        """
        if isinstance(self.value, str):
            try:
                # Try int first (more specific), then float
                int_val = int(self.value)
                return str(int_val)
            except ValueError:
                try:
                    float_val = float(self.value)
                    return str(float_val)
                except ValueError:
                    return f'"{self.value}"'
        return f'"{self.value}"'

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: MiniDocStringType | None = None) -> Any:

        if value_type is None:
            raise RuntimeError("Missing expected type for evaluation of Value")

        return value_type.cast(self.value, allow_scalar_to_list=True)
