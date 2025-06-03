from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fifo_dev_dsl.dia.dsl.elements.base import DslBase, make_dsl_container
from fifo_dev_dsl.dia.resolution.interaction import Interaction
from fifo_dev_dsl.dia.resolution.enums import AbortBehavior

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext


@dataclass
class Slot(make_dsl_container(DslBase)):

    name: str

    def __init__(self, name: str, value: DslBase):
        super().__init__([value])
        self.name = name

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
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None):
        super().pre_resolution(runtime_context, resolution_context, abort_behavior, interaction)

        assert resolution_context.intent is not None

        resolution_context.slot = self
        resolution_context.other_slots = {}
        for slot in resolution_context.intent.get_items():
            if slot.name != resolution_context.slot.name:
                value_as_text = slot.value.represent_content_as_text()
                if value_as_text is not None:
                    resolution_context.other_slots[slot.name] = value_as_text

    def post_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None):
        super().post_resolution(runtime_context, resolution_context, abort_behavior, interaction)
        resolution_context.slot = None
        resolution_context.other_slots = None
