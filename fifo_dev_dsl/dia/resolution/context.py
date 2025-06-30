from __future__ import annotations
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.propagate_slots import PropagateSlots
from fifo_dev_dsl.dia.dsl.elements.intent_runtime_error_resolver import IntentRuntimeErrorResolver
from fifo_dev_dsl.dia.dsl.elements.query_user import QueryUser
from fifo_dev_dsl.dia.dsl.elements.ask import Ask
from fifo_dev_dsl.dia.dsl.elements.query_gather import QueryGather
from fifo_dev_dsl.dia.resolution.llm_call_log import LLMCallLog


@dataclass
class _ResolutionState:
    """
    Snapshot of the current intent, slot and known other slots.

    Attributes:
        intent (Intent | None):
            The current intent being resolved, or None if no intent is active.

        slot (Slot | None):
            The current slot being resolved, or None if no slot is active.

        other_slots (dict[str, str] | None):
            Known values for other slots associated with the current intent,
            excluding the one currently being resolved.
    """

    intent: Any
    slot: Any
    other_slots: dict[str, str] | None

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

        _state_stack (list[_ResolutionState]):
            Internal stack preserving `intent`, `slot` and `other_slots` when
            entering nested intents or slots. The public `intent`, `slot` and
            `other_slots` attributes provide access to the values stored at the
            top of this stack.
    """

    def __init__(
        self,
        intent: Intent | None = None,
        slot: Slot | None = None,
        other_slots: dict[str, str] | None = None,
        _propagate_slots: list[PropagateSlots] | None = None,
        questions_being_clarified: list[
            tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str, str]
        ] | None = None,
        call_stack: list[ResolutionContextStackElement] | None = None,
        llm_call_logs: list[LLMCallLog] | None = None,
    ) -> None:
        self._state_stack: list[_ResolutionState] = [
            _ResolutionState(intent, slot, dict(other_slots) if other_slots is not None else None)
        ]
        self._propagate_slots: list[PropagateSlots] = _propagate_slots or []
        self.questions_being_clarified = questions_being_clarified or []
        self.call_stack = call_stack or []
        self.llm_call_logs = llm_call_logs or []

    # ------------------------------------------------------------------
    # Properties exposing the current resolution state
    # ------------------------------------------------------------------
    @property
    def intent(self) -> Intent | None:
        """
        The current intent being resolved.

        Returns:
            Intent | None:
                The active intent from the top of the internal state stack.
        """
        return self._state_stack[-1].intent

    @intent.setter
    def intent(self, value: Intent | None) -> None:
        """
        Sets the current intent on the top of the internal state stack.

        Args:
            value (Intent | None):
                The intent to set as currently being resolved.
        """
        self._state_stack[-1].intent = value

    @property
    def slot(self) -> Slot | None:
        """
        The current slot being resolved.

        Returns:
            Slot | None:
                The active slot from the top of the internal state stack.
        """
        return self._state_stack[-1].slot

    @slot.setter
    def slot(self, value: Slot | None) -> None:
        """
        Sets the current slot on the top of the internal state stack.

        Args:
            value (Slot | None):
                The slot to set as currently being resolved.
        """
        self._state_stack[-1].slot = value

    @property
    def other_slots(self) -> dict[str, str] | None:
        """
        Known values for all other slots in the current intent.

        Returns:
            dict[str, str] | None:
                A mapping of other resolved slots (excluding the current one).
        """
        return self._state_stack[-1].other_slots

    @other_slots.setter
    def other_slots(self, value: dict[str, str] | None) -> None:
        """
        Sets the known values for other slots in the current intent.

        Args:
            value (dict[str, str] | None):
                A dictionary of resolved slots excluding the current one.
        """
        self._state_stack[-1].other_slots = value

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

    def entering_intent(self, intent: Intent) -> None:
        """
        Begin resolving a nested intent.

        Pushes the current resolution state onto the internal stack and 
        creates a new top-level state for the nested `intent`. The new 
        state starts with no active slot or other slot values.

        This must only be called when entering a nested intent, not the initial one,
        which is created automatically when the ResolutionContext is initialized.

        Args:
            intent (Intent):
                The nested intent that is now being resolved.
        """
        self._state_stack.append(_ResolutionState(intent, None, None))

    def exiting_intent(self) -> None:
        """
        Restore the previous resolution state when leaving an intent.
        """
        if self._state_stack:
            self._state_stack.pop()
        if not self._state_stack:
            self._state_stack.append(_ResolutionState(None, None, None))

    def reset_state(self) -> None:
        """
        Clear the entire state stack and reset intent and slot.
        """
        self._state_stack = [_ResolutionState(None, None, None)]

    def get_intent_name(self) -> str:
        """
        Returns the name of the current intent, or `"none"` if no intent is set.

        Returns:
            str:
                The intent name, or `"none"` if `self.intent` is `None`.
        """
        return self.intent.name if self.intent else "none"

    def get_slot_name(self) -> str:
        """
        Returns the name of the current slot, or `"none"` if no slot is set.

        Returns:
            str:
                The slot name, or `"none"` if `self.slot` is `None`.
        """
        return self.slot.name if self.slot else "none"
