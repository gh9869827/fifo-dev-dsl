from __future__ import annotations
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

from fifo_dev_common.introspection.mini_docstring import MiniDocStringType

from fifo_dev_dsl.common.dsl_utils import quote_and_escape
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.elements.helper import ask_helper_no_interaction
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.interaction import Interaction, InteractionRequest
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class Ask(DslBase):
    """
    Prompt the user to provide a missing slot value during resolution.

    `ASK` nodes act as placeholders for information that cannot be resolved
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

    def do_resolution(
        self,
        runtime_context: LLMRuntimeContext,
        resolution_context: ResolutionContext,
        interaction: Interaction | None,
    ) -> ResolutionOutcome:

        super().do_resolution(runtime_context, resolution_context, interaction)

        if (
            interaction is None
            or interaction.request.requester is not self
        ):
            return ResolutionOutcome(
                result=ResolutionResult.INTERACTION_REQUESTED,
                interaction=InteractionRequest(
                    message=self.question,
                    expected_type=MiniDocStringType(str),
                    slot=resolution_context.slot,
                    requester=self
                )
            )

        assert interaction.answer.consumed is False
        user_answer = interaction.answer.content
        interaction.answer.consumed = True

        resolution_text = f"""resolution_context:
  intent: {resolution_context.get_intent_name()}
  slot: {resolution_context.get_slot_name()}
{resolution_context.format_previous_qna_block()}
  current_question: {self.question}
  current_user_answer: {user_answer}"""

        return ask_helper_no_interaction(
            runtime_context,
            runtime_context.system_prompt_slot_resolver,
            (self, self.question),
            resolution_context,
            resolution_text,
            user_answer
        )

    def is_resolved(self) -> bool:
        """
        Indicate that ASK nodes are never resolved.

        ASK elements prompt the user for input and therefore remain
        placeholders in the DSL tree until replaced with the user's response.

        Returns:
            bool:
                Always `False`.
        """
        return False

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Raise a `RuntimeError` because Ask nodes are unresolved.

        ASK elements represent pending user input and remain unresolved until
        they are replaced with a concrete value during resolution. Attempting to
        evaluate such a node directly is invalid and results in an error.

        Raises:
            RuntimeError: Always raised with the message
                Unresolved DSL node: Ask
        """
        raise RuntimeError(f"Unresolved DSL node: {self.__class__.__name__}")

    async def eval_async(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Asynchronously raise a `RuntimeError` because Ask nodes are
        unresolved.

        ASK elements represent pending user input and remain unresolved until
        they are replaced with a concrete value during resolution. Attempting to
        evaluate such a node directly is invalid and results in an error.

        Raises:
            RuntimeError: Always raised with the message
                Unresolved DSL node: Ask
        """
        raise RuntimeError(f"Unresolved DSL node: {self.__class__.__name__}")
