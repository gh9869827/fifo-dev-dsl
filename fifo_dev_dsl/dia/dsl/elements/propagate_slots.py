from __future__ import annotations
from typing import TYPE_CHECKING


from common.llm.dia.dsl.elements.base import DslBase, DslContainerBase
from common.llm.dia.dsl.elements.slot import Slot
from common.llm.dia.resolution.interaction import Interaction

if TYPE_CHECKING:
    from common.llm.dia.resolution.resolver import AbortBehavior
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext

class PropagateSlots(DslContainerBase[Slot]):

    def __init__(self, slots: list[Slot]):
        super().__init__(slots)

    def to_dict(self) -> dict[str, DslBase]:
        return {
            propagated_slot.name : propagated_slot.value for propagated_slot in self.get_items()
        }

    def pre_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None):
        super().pre_resolution(runtime_context, resolution_context, abort_behavior, interaction)
        resolution_context.slot = self
    
    def post_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None):
        super().post_resolution(runtime_context, resolution_context, abort_behavior, interaction)
        resolution_context.intent = None
