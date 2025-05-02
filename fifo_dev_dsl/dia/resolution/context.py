from __future__ import annotations
from ast import Tuple
from typing import TYPE_CHECKING, List, Optional, Dict, Union
import copy
from dataclasses import dataclass, field


if TYPE_CHECKING:
    from common.llm.dia.dsl.elements.query_user import QueryUser
    from common.llm.dia.dsl.elements.ask import Ask
    from common.llm.dia.dsl.elements.propagate_slot import PropagateSlot


@dataclass
class ResolutionContext:

    intent: Optional[str] = None
    slot: Optional[str] = None
    other_slots: Optional[Dict[str, str]] = None
    propagate: Optional[PropagateSlot] = None
    questions_being_clarified: List[Tuple[Union[Ask, QueryUser], str]] = field(default_factory=list)

    def deepcopy(self) -> ResolutionContext:
        return copy.deepcopy(self)

    def display_other_slots(self) -> str:
        if self.other_slots:
            return "".join([f"\n    - {key}: {value}" for key, value in self.other_slots.items()])
        return "none"

    def format_other_slots_yaml(self, padding: str="") -> str:
        if not self.other_slots:
            return f"{padding}other_slots: {{}}"

        lines = [f"{padding}other_slots:"]
        for key, value in self.other_slots.items():
            lines.append(f"{padding}  {key}: {value}")
        return "\n".join(lines)
