from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from common.llm.dia.dsl.elements.slot import Slot
    from common.llm.dia.dsl.elements.base import DslBase
    from common.llm.dia.dsl.elements.intent import Intent
    from common.llm.dia.dsl.elements.query_user import QueryUser
    from common.llm.dia.dsl.elements.ask import Ask
    from common.llm.dia.dsl.elements.propagate_slots import PropagateSlots


@dataclass
class ResolutionContextStackElement:
    obj: DslBase
    idx: int

@dataclass
class LLMCallLog:
    # each llm call is composed exactly of a system prompt a assistant prompt and a user(answer) prompt
    # the system prompt gives the instructions, the assistant prompt gives the resolution/runtime
    # specific context and the answer/user is the actual output of the llm.
    # a Call Log can be used to fine tune the model behavior by correcting the answer of a model
    # and retraining on the new expected output.  
    description: str
    system_prompt: str
    assistant: str
    answer: str

@dataclass
class ResolutionContext:

    intent: Intent | None = None
    slot: Slot | None = None
    other_slots: dict[str, str]  | None = None
    _propagate_slots: list[PropagateSlots] = field(default_factory=list, repr=False)
    questions_being_clarified: list[tuple[Ask | QueryUser, str]] = field(default_factory=list)
    call_stack: list[ResolutionContextStackElement] = field(default_factory=list)
    llm_call_logs: list[LLMCallLog]

    def format_call_log(self) -> str:
        if not self.llm_call_logs:
            return ""
        res = "---"
        for call_log in self.llm_call_logs:
            res = f"""{res}
$
{call_log.system_prompt}
>
{call_log.assistant}
<
{call_log.answer}
---
"""
        return res

    def format_other_slots_yaml(self, padding: str="") -> str:
        if not self.other_slots:
            return f"{padding}other_slots: {{}}"

        lines = [f"{padding}other_slots:"]
        for key, value in self.other_slots.items():
            lines.append(f"{padding}  {key}: {value}")
        return "\n".join(lines)

    def add_propagated_slot(self, slot: PropagateSlots) -> None:
        """
        Add a propagated slot set to the pending list.

        Args:
            slot (PropagateSlots):
                A slot propagation instruction to defer until reentry.
        """
        self._propagate_slots.append(slot)

    def take_propagated_slots(self) -> list[PropagateSlots]:
        """
        Consume and return all currently pending propagated slot sets.

        Returns:
            list[PropagateSlots]:
                All accumulated propagate instructions, and clears the queue.
        """
        slots = self._propagate_slots
        self._propagate_slots = []
        return slots
