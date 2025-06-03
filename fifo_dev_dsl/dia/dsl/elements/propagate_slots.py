from __future__ import annotations

from fifo_dev_dsl.dia.dsl.elements.base import DslBase, make_dsl_container
from fifo_dev_dsl.dia.dsl.elements.slot import Slot

class PropagateSlots(make_dsl_container(Slot)):

    def __init__(self, slots: list[Slot]):
        super().__init__(slots)

    def to_dict(self) -> dict[str, DslBase]:
        return {
            propagated_slot.name : propagated_slot.value for propagated_slot in self.get_items()
        }
