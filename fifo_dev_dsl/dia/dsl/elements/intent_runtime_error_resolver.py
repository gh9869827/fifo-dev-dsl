from __future__ import annotations
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

from fifo_dev_common.introspection.mini_docstring import MiniDocStringType

from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.elements.helper import ask_helper_no_interaction
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.interaction import InteractionRequest
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.resolution.interaction import Interaction
    from fifo_dev_dsl.dia.dsl.elements.intent import Intent
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class IntentRuntimeErrorResolver(DslBase):
    """
    Placeholder for a failed intent awaiting user-guided recovery.

    This node is injected into the DSL tree when an :class:`Intent` raises a runtime error
    during evaluation. It captures the failing intent and the associated error message,
    pausing automatic resolution.

    During the next resolution pass, this node prompts the user for a corrective action —
    such as retrying, modifying parameters, or abandoning the operation — and allows
    the system to generate new DSL elements in response.

    This mechanism enables interactive recovery from unexpected execution failures
    while preserving context.

    Attributes:
        intent (Intent):
            The intent that encountered a runtime error.

        error_message (str):
            A human-readable description of the failure.
    """

    intent: Intent
    error_message: str

    def pretty_print_dsl(self, indent: int = 0) -> None:
        prefix = "  " * indent
        print(f"{prefix}{self.__class__.__name__}(error_message={repr(self.error_message)})")
        self.intent.pretty_print_dsl(indent + 1)

    def get_children(self) -> list[DslBase]:
        return [self.intent]

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
                    message=self.error_message,
                    expected_type=MiniDocStringType(str),
                    slot=resolution_context.slot,
                    requester=self
                )
            )

        assert interaction.answer.consumed is False
        user_answer = interaction.answer.content
        interaction.answer.consumed = True

        resolution_text = f"""resolution_context:
  intent: {self.intent.to_dsl_representation()}
{resolution_context.format_previous_qna_block()}
  error: {self.error_message}
  current_user_answer: {user_answer}"""

        return ask_helper_no_interaction(
            runtime_context,
            runtime_context.system_prompt_error_resolver,
            (self, self.error_message),
            resolution_context,
            resolution_text,
            user_answer
        )

    def is_resolved(self) -> bool:
        """
        Indicate that this node always stays unresolved.

        The resolver reflects a runtime error in the intent flow and cannot be
        reduced to a normal value.

        Returns:
            bool:
                Always ``False``.
        """
        return False

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Raise a RuntimeError because IntentRuntimeErrorResolver nodes are unresolved.

        These nodes indicate that an intent failed during evaluation and requires
        user intervention. They must be replaced during resolution before evaluation
        can proceed.

        Raises:
            RuntimeError: Always raised with the message
                Unresolved DSL node: IntentRuntimeErrorResolver
        """

        raise RuntimeError(
            f"Unresolved DSL node: {self.__class__.__name__}"
        )

    async def eval_async(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Asynchronously raise a RuntimeError because
        ``IntentRuntimeErrorResolver`` nodes are unresolved.

        These nodes indicate that an intent failed during evaluation and
        requires user intervention. They must be replaced during resolution
        before evaluation can proceed.

        Raises:
            RuntimeError: Always raised with the message
                Unresolved DSL node: IntentRuntimeErrorResolver
        """

        raise RuntimeError(
            f"Unresolved DSL node: {self.__class__.__name__}"
        )
