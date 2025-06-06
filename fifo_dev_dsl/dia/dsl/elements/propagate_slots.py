from __future__ import annotations

from fifo_dev_dsl.dia.dsl.elements.base import DslBase, make_dsl_container
from fifo_dev_dsl.dia.dsl.elements.slot import Slot

class PropagateSlots(make_dsl_container(Slot)):
    """
    Forward extra slot values inferred from user input into the current resolution.

    This node captures values the user provides that were not explicitly requested
    but are still relevant. These values are propagated and merged into the
    current intent before final execution.

    It is commonly used when the DSL requests a specific slot value and the user
    responds with additional useful information — for example, specifying both
    a quantity and a length when only the quantity was asked.

    Example:
        Question: "How many screws do you need?"
        Answer: "Actually, give me 5 10mm screws."
        Result: value = 5, PropagateSlots([Slot("length", Value("10"))])
    """

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
