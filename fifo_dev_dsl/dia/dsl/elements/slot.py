from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fifo_dev_dsl.dia.dsl.elements.base import DslBase, make_dsl_container

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.resolution.interaction import Interaction
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext


@dataclass
class Slot(make_dsl_container(DslBase)):
    """
    Associate a named parameter with a DSL value.

    Slots represent arguments passed to an `Intent`. Each slot pairs a name
    with a single DSL expression — either a literal (e.g., `Value("12")`)
    or a complex structure (e.g., `Ask`, `QueryFill`, or a nested `Intent`).

    Although each slot contains exactly one value, it inherits from the DSL container
    base to allow uniform traversal, transformation, and resolution hooks. This enables
    consistent handling of all DSL elements, including support for nesting and lifecycle
    hooks like `pre_resolution` and `post_resolution`.

    Attributes:
        name (str):
            The name of the argument to bind.

    Examples:
        Literal value:
            Slot("length", Value("12"))

        Nested invocation:
            Slot("target", ReturnValue(Intent(name="get_location", slots=[])))
    """

    name: str

    def __init__(self, name: str, value: DslBase):
        super().__init__([value])
        self.name = name

    def __eq__(self, other: Any) -> bool:
        return (
                isinstance(other, self.__class__)
            and self.name == other.name
            and super().__eq__(other)
        )

    @property
    def value(self) -> DslBase:
        """
        Get the value of the slot.

        Returns:
            DslBase:
                The single DSL node stored in this slot.
        """
        return self._items[0]

    @value.setter
    def value(self, new_value: DslBase) -> None:
        """
        Set a new value for this slot.

        Args:
            new_value (DslBase):
                The DSL node to assign to this slot.
        """
        self._items[0] = new_value

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the slot.

        Combines the slot name with the DSL representation of its value,
        formatted as `name=value`.

        Returns:
            str:
                The slot assignment in DSL form, e.g., `count=42`.
        """
        return f"{self.name}={self.value.to_dsl_representation()}"

    def pre_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       interaction: Interaction | None):
        super().pre_resolution(runtime_context, resolution_context, interaction)

        assert resolution_context.intent is not None

        resolution_context.slot = self
        resolution_context.other_slots = {}
        for slot in resolution_context.intent.get_items():
            if slot.name != resolution_context.slot.name:
                value_as_text = slot.value.to_dsl_representation()
                resolution_context.other_slots[slot.name] = value_as_text

    def post_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       interaction: Interaction | None):
        super().post_resolution(runtime_context, resolution_context, interaction)
        resolution_context.slot = None
        resolution_context.other_slots = None

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Evaluate the value stored in this slot and return the result of its evaluation.

        The slot directly delegates evaluation to its stored value. If the
        stored value is unresolved, it will raise a RuntimeError during its
        own evaluation.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context providing tool access, query sources, and runtime helpers.

        Returns:
            Any:
                The result from evaluating the stored value.

        Raises:
            RuntimeError: If the stored value is not resolved.
        """
        return self.value.eval(runtime_context)

    async def eval_async(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Asynchronously evaluate the value stored in this slot.

        The slot asynchronously delegates evaluation to its stored value. If the
        stored value is unresolved, it will raise a RuntimeError during its
        own evaluation.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context providing tool access, query sources, and runtime helpers.

        Returns:
            Any:
                The result from evaluating the stored value.

        Raises:
            RuntimeError: If the stored value is not resolved.
        """
        return await self.value.eval_async(runtime_context)
