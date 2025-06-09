from __future__ import annotations
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.value_base import DSLValueBase

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
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
        """Return the stored constant after casting to ``value_type``.

        A ``Value`` node is always resolved. If this expectation is violated for
        some reason, a :class:`RuntimeError` is raised. The result is cast to the
        provided ``value_type`` using ``MiniDocStringType.cast``.

        Args:
            runtime_context: Execution context (unused).
            value_type: Expected type used to cast ``self.value``.

        Returns:
            Any: ``self.value`` converted to the requested type.
        """

        if not self.is_resolved():
            raise RuntimeError(
                f"Unresolved DSL node: {self.__class__.__name__}"
            )

        if value_type is None:
            raise RuntimeError(
                "Missing expected type for evaluation of Value"
            )

        return value_type.cast(self.value, allow_scalar_to_list=True)
