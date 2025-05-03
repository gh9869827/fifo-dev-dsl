from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from common.llm.dia.dsl.elements.base import DslBase, DslContainerBase
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.enums import AbortBehavior

if TYPE_CHECKING:
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext

@dataclass
class Slot(DslContainerBase[DslBase]):

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

    def pre_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None):
        super().pre_resolution(runtime_context, resolution_context, abort_behavior, interaction)
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
