from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

from fifo_dev_dsl.dia.dsl.elements.propagate_slots import PropagateSlots
from fifo_dev_dsl.dia.dsl.elements.intent_runtime_error_resolver import IntentRuntimeErrorResolver
from fifo_dev_dsl.dia.dsl.elements.query_user import QueryUser
from fifo_dev_dsl.dia.dsl.elements.ask import Ask
from fifo_dev_dsl.dia.dsl.elements.query_gather import QueryGather
from fifo_dev_dsl.dia.resolution.llm_call_log import LLMCallLog


if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.dsl.elements.slot import Slot
    from fifo_dev_dsl.dia.dsl.elements.base import DslBase
    from fifo_dev_dsl.dia.dsl.elements.intent import Intent


@dataclass
class ResolutionContextStackElement:
    obj: DslBase
    idx: int


@dataclass
class ResolutionContext:

    intent: Intent | None = None
    slot: Slot | None = None
    other_slots: dict[str, str]  | None = None
    _propagate_slots: list[PropagateSlots] = field(default_factory=list[PropagateSlots], repr=False)
    questions_being_clarified: list[tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str, str]] = field(default_factory=list[tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str, str]])
    call_stack: list[ResolutionContextStackElement] = field(default_factory=list[ResolutionContextStackElement])
    llm_call_logs: list[LLMCallLog] = field(default_factory=list[LLMCallLog])

    def format_previous_qna_block(self) -> str:
        if self.questions_being_clarified:
            previous_qna_yaml = "\n".join(
                f"    - question: {q}\n      answer: {a}" for _, q, a
                                                          in self.questions_being_clarified
            )
            return f"  previous_questions_and_answers:\n{previous_qna_yaml}"
        return "  previous_questions_and_answers: []"

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
---"""
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
