from __future__ import annotations
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass

from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
from fifo_dev_dsl.dia.dsl.elements.base import make_dsl_container
from fifo_dev_dsl.dia.dsl.elements.slot import Slot

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.resolution.interaction import Interaction
    from fifo_dev_dsl.dia.resolution.enums import AbortBehavior
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class Intent(make_dsl_container(Slot)):

    name: str

    def __init__(self, name: str, slots: list[Slot]):
        super().__init__(slots)
        self.name = name

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the intent.

        Formats the intent as a function-style call with named slot arguments,
        such as `set_task_due_date(task_uid=42, due_date="tomorrow at noon")`.

        Returns:
            str:
                A string representing the intent and its slots in DSL form.
        """
        parts = [slot.to_dsl_representation() for slot in self._items]
        return f"{self.name}({', '.join(parts)})"

    def _propagate_slots(self,
                         resolution_context: ResolutionContext):

        assert resolution_context.slot is None

        for propagated_slots in resolution_context.take_propagated_slots():
            pslots = propagated_slots.to_dict()
            updated: set[str] = set()

            for slot in self.get_items():
                pslot_value = pslots.get(slot.name)
                if pslot_value is not None:
                    print(f"--> propagating slots {slot.name}, "
                          f"{slot.value} replaced by {pslot_value} ")
                    slot.value = pslot_value
                    updated.add(slot.name)

            # process the unconsumed propagated slots
            for name, value in pslots.items():
                if name not in updated:
                    self._items.append(Slot(name, value))

    def pre_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None):
        super().pre_resolution(runtime_context, resolution_context, abort_behavior, interaction)
        resolution_context.intent = self

    def post_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None):
        super().post_resolution(runtime_context, resolution_context, abort_behavior, interaction)
        resolution_context.intent = None

    def on_reentry_resolution(self,
                              runtime_context: LLMRuntimeContext,
                              resolution_context: ResolutionContext,
                              abort_behavior: AbortBehavior,
                              interaction: Interaction | None):
        super().on_reentry_resolution(
            runtime_context, resolution_context, abort_behavior, interaction
        )
        self._propagate_slots(resolution_context)

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: MiniDocStringType | None = None) -> Any:

        tool = runtime_context.get_tool(self.name)

        args = {
            slot.name: slot.value.eval(
                runtime_context, tool.tool_docstring.get_arg_by_name(slot.name).pytype
            ) for slot in self._items
        }

        if tool.tool_docstring.return_type is None:
            tool(**args)

            return None

        ret = tool.tool_docstring.return_type.cast(tool(**args))

        return value_type.cast(ret) if value_type is not None else ret
