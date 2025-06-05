from __future__ import annotations

from fifo_dev_dsl.dia.dsl.elements.base import DslBase, make_dsl_container
from fifo_dev_dsl.dia.dsl.elements.slot import Slot

class PropagateSlots(make_dsl_container(Slot)):

    def __init__(self, slots: list[Slot]):
        super().__init__(slots)

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the PropagateSlots node.

        Returns:
            str:
                A string in DSL syntax listing all propagated slots, e.g.,
                'PROPAGATE_SLOT(x=1, y=foo())'.
        """
        slots = ", ".join([i.to_dsl_representation() for i in self.get_items()])
        return f"PROPAGATE_SLOT({slots})"

    def to_dict(self) -> dict[str, DslBase]:
        return {
            propagated_slot.name : propagated_slot.value for propagated_slot in self.get_items()
        }
