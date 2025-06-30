from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass

from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server

from fifo_dev_common.introspection.mini_docstring import MiniDocStringType

from fifo_dev_dsl.dia.resolution.llm_call_log import LLMCallLog
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.interaction import Interaction, InteractionRequest
from fifo_dev_dsl.dia.dsl.elements.helper import ask_helper_no_interaction
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.common.dsl_utils import quote_and_escape

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext


@dataclass
class QueryUser(DslBase):
    """
    Represent a user question that should be answered by the system.

    This node is used when the user's input is itself a question — not a request to
    invoke a tool, but a query seeking information. Unlike `Intent` nodes, which 
    trigger actions, `QueryUser` captures an information-seeking goal, such as 
    "How many screws are in the inventory?".

    During resolution, the system treats this as a direct user question and attempts
    to answer it immediately, rather than continuing with tool execution. Once the 
    answer is provided, the DSL requests a follow-up from the user — who may choose 
    to abort, refine their request, or proceed with a new intent now that they have 
    the information they needed.

    Attributes:
        query (str):
            The natural language question asked by the user.

    Example:
        Input:  "How many screws do we have in the inventory?"
        Output: QUERY_USER("How many screws do we have in the inventory?")
    """

    query: str

    def is_resolved(self) -> bool:
        """
        Report that the user's question has not yet been answered.

        `QUERY_USER` nodes capture a direct question from the user. The system
        must gather information and replace this placeholder with the answer
        before the intent can proceed.

        Returns:
            bool:
                Always `False`.
        """
        return False

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the QueryUser node.

        Returns:
            str:
                The query in DSL syntax, with internal quotes escaped and the value properly quoted.
                For example: QUERY_USER("query").
        """
        return f'QUERY_USER({quote_and_escape(self.query)})'

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Raise a RuntimeError because QueryUser nodes are unresolved.

        These nodes encapsulate a user question awaiting a response. They must
        be resolved to a concrete value during resolution before evaluation can
        proceed. Encountering one during evaluation indicates that resolution
        has not completed successfully.

        Raises:
            RuntimeError: Always raised with the message
                Unresolved DSL node: QueryUser
        """
        raise RuntimeError(f"Unresolved DSL node: {self.__class__.__name__}")

    async def eval_async(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Asynchronously raise a RuntimeError because `QueryUser` nodes are
        unresolved.

        These nodes encapsulate a user question awaiting a response. They must
        be resolved to a concrete value during resolution before evaluation can
        proceed. Encountering one during evaluation indicates that resolution
        has not completed successfully.

        Raises:
            RuntimeError: Always raised with the message
                Unresolved DSL node: QueryUser
        """

        raise RuntimeError(f"Unresolved DSL node: {self.__class__.__name__}")

    def do_resolution(
        self,
        runtime_context: LLMRuntimeContext,
        resolution_context: ResolutionContext,
        interaction: Interaction | None,
    ) -> ResolutionOutcome:

        super().do_resolution(runtime_context, resolution_context, interaction)

        if interaction is not None and interaction.request.requester is self:

            assert interaction.answer.consumed is False
            user_answer = interaction.answer.content
            interaction.answer.consumed = True

            resolution_text = "resolution_context:"

            if resolution_context.intent is not None and resolution_context.slot is not None:
                # The user question is asked within the context of a specific intent and slot.
                # This means the assistant is trying to resolve the value for that slot, so we use
                # the `system_prompt_slot_resolver` to guide the resolution process.
                #
                # If the user responds with a follow-up that redirects or overrides the intent,
                # the system may abort the current intent and create a new one.
                #
                # Example interaction:
                #   >  dia(1): What length do you need?
                #   > user(1): What length do you have in the inventory?
                #   >  dia(2): We have screws of lengths 8, 10, 11, 12, and 16 in the inventory.
                #   > user(2): OK, give me 8mm screws then.
                #
                # Here:
                # - `QueryUser.query` contains the original question: user(1)
                # - `answer = call_airlock_model_server(...)` returns dia's response: dia(2)
                # - `interaction.answer.content` contains the user follow-up: user(2)
                resolution_prompt = runtime_context.system_prompt_slot_resolver

                resolution_text = f"""{resolution_text}
  intent: {resolution_context.get_intent_name()}
  slot: {resolution_context.get_slot_name()}"""

            else:
                # The user question is not tied to any intent or slot resolution context.
                # We therefore call the intent sequencer directly to determine the user's intent.
                resolution_prompt = runtime_context.system_prompt_intent_sequencer

            resolution_text = f"""{resolution_text}
{resolution_context.format_previous_qna_block()}
  current_question: {interaction.request.message}
  current_user_answer: {user_answer}"""

            return ask_helper_no_interaction(
                runtime_context,
                resolution_prompt,
                (self, interaction.request.message),
                resolution_context,
                resolution_text,
                user_answer
            )

        prompt_user = runtime_context.get_user_prompt_dynamic_query(resolution_context, self.query)

        answer = call_airlock_model_server(
                    model=runtime_context.base_model,
                    messages=[
                        Message.system(runtime_context.system_prompt_query_user),
                        Message.user(prompt_user)
                    ],
                    parameters=GenerationParameters(
                        max_new_tokens=1024,
                        do_sample=False
                    ),
                    container_name=runtime_context.container_name,
                    host=runtime_context.host
                )

        resolution_context.llm_call_logs.append(
            LLMCallLog(
                description="QueryUser[do_resolution]",
                system_prompt=runtime_context.system_prompt_query_user,
                assistant=prompt_user,
                answer=answer
            )
        )

        match = re.search(
            r"reasoning:\s*(.*?)\nuser friendly answer:(.*)",
            answer,
            flags=re.DOTALL
        )

        if match:
            value = match[2].strip()
        else:
            value = "unknown"

        return ResolutionOutcome(
            result=ResolutionResult.INTERACTION_REQUESTED,
            interaction=InteractionRequest(
                message=value,
                expected_type=MiniDocStringType(str),
                slot=resolution_context.slot,
                requester=self
            )
        )
