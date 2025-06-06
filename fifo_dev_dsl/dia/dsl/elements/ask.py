from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

from fifo_dev_dsl.common.dsl_utils import quote_and_escape
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
import fifo_dev_dsl.dia.dsl.elements.helper as helper

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.resolution.enums import AbortBehavior
    from fifo_dev_dsl.dia.resolution.interaction import Interaction
    from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class Ask(DslBase):
    """
    Prompt the user to provide a missing slot value during resolution.

    ``ASK`` nodes act as placeholders for information that cannot be resolved
    automatically. They remain in the DSL tree until the user responds, at which
    point they are replaced with the returned value.

    Attributes:
        question (str):
            The question to present to the user.

    Example:
        create_task(
            task_description="...",
            task_list_short_name="...",
            due_date=ASK("What is the target date?")
        )
    """

    question: str

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the Ask node.

        Returns:
            str:
                The question in DSL syntax, with internal quotes escaped and the string properly
                quoted. For example: ASK("question").
        """
        return f'ASK({quote_and_escape(self.question)})'

    def do_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None) -> ResolutionOutcome:
        super().do_resolution(runtime_context, resolution_context, abort_behavior, interaction)

        return helper.ask_helper_slot_resolver(
            runtime_context=runtime_context,
            current=(self, self.question),
            resolution_context=resolution_context,
            interaction=interaction)

    def is_resolved(self) -> bool:
        """
        Indicate that ASK nodes are never resolved.

        ASK elements prompt the user for input and therefore remain
        placeholders in the DSL tree until replaced with the user's response.

        Returns:
            bool:
                Always ``False``.
        """
        return False
