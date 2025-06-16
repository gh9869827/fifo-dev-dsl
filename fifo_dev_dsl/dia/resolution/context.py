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
    """
    Represents a single frame in the DSL resolution stack.

    Each element holds the current DSL object being processed, and an index
    indicating which child node is next in the traversal. This supports resumable,
    stack-based evaluation of nested DSL structures.

    Attributes:
        obj (DslBase):
            The DSL node being resolved at this stack frame.

        idx (int):
            The index of the next child to visit within `obj`, incremented as traversal progresses.
    """
    obj: DslBase
    idx: int

@dataclass
class ResolutionContext:
    """
    Carries the state of the current DSL resolution process.

    This context is passed to all DSL elements and evolves over time as the intent,
    slot, user answers, and LLM interactions change. It tracks the current intent and
    slot being resolved, any additional known slot values, clarification dialogs, and
    a full log of LLM calls for traceability.

    Attributes:
        intent (Intent | None):
            The current intent being resolved, or None if no intent is active.

        slot (Slot | None):
            The current slot being resolved, or None if no slot is active.

        other_slots (dict[str, str] | None):
            Known values for other slots associated with the current intent,
            excluding the one currently being resolved.

        _propagate_slots (list[PropagateSlots]):
            Internal list of slots to propagate. Not included in `repr()`.

        questions_being_clarified (list[tuple[IntentRuntimeErrorResolver | Ask | QueryUser 
                                              | QueryGather, str, str]]):
            Log of previous clarification interactions, each entry containing the clarifying
            element, the question asked, and the user's answer or gathered data.

        call_stack (list[ResolutionContextStackElement]):
            Tracks the active DSL elements being resolved, maintained by the Resolver
            during stack-based traversal.

            Used to support resumable evaluation and reentry after user interaction,
            enabling precise control over where resolution resumes.

        llm_call_logs (list[LLMCallLog]):
            Full record of LLM interactions during resolution, including system prompts,
            assistant outputs, and user responses. Useful for debugging, generating traces,
            or curating fine-tuning examples.
    """

    intent: Intent | None = None
    slot: Slot | None = None
    other_slots: dict[str, str]  | None = None
    _propagate_slots: list[PropagateSlots] = field(default_factory=list[PropagateSlots], repr=False)
    questions_being_clarified: list[tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str, str]] = field(default_factory=list[tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str, str]])
    call_stack: list[ResolutionContextStackElement] = field(default_factory=list[ResolutionContextStackElement])
    llm_call_logs: list[LLMCallLog] = field(default_factory=list[LLMCallLog])

    def format_previous_qna_block(self) -> str:
        """
        Format the list of previous clarification questions and answers into YAML.

        Returns:
            str:
                A YAML-formatted string representing `previous_questions_and_answers`, or
                an empty list if no clarifications were made.
        """
        if self.questions_being_clarified:
            previous_qna_yaml = "\n".join(
                f"    - question: {q}\n      answer: {a}" for _, q, a
                                                          in self.questions_being_clarified
            )
            return f"  previous_questions_and_answers:\n{previous_qna_yaml}"
        return "  previous_questions_and_answers: []"

    def format_call_log(self) -> str:
        """
        Format all LLM interaction logs into a DSL-like block format compatible
        with the Conversation adapter.

        Returns:
            str:
                A formatted string containing system prompts, assistant completions,
                and user answers for each LLM interaction. Returns an empty string
                if no call logs are present.
        """
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
        """
        Format the contents of `other_slots` as a YAML block with optional indentation.

        `other_slots` contains all known slot values for the current intent,
        excluding the slot currently being resolved.

        Args:
            padding (str):
                Optional string to prepend to each line for indentation.

        Returns:
            str:
                A YAML-formatted block of key-value pairs for all known non-current
                slots, or an empty object if none exist.
        """
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
