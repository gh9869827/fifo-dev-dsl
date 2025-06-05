from __future__ import annotations
from typing import TYPE_CHECKING

from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message, Model, Role
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server

import fifo_dev_dsl.dia.dsl.parser.parser as parser
from fifo_dev_dsl.dia.resolution.llm_call_log import LLMCallLog
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.interaction import Interaction, InteractionRequest
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.dsl.elements.intent_runtime_error_resolver import IntentRuntimeErrorResolver
    from fifo_dev_dsl.dia.dsl.elements.intent import Intent
    from fifo_dev_dsl.dia.dsl.elements.query_gather import QueryGather
    from fifo_dev_dsl.dia.dsl.elements.query_user import QueryUser
    from fifo_dev_dsl.dia.dsl.elements.ask import Ask
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext


def ask_helper_slot_resolver(
        runtime_context: LLMRuntimeContext,
        current: tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str],
        resolution_context: ResolutionContext,
        interaction: Interaction | None = None
) -> ResolutionOutcome:

    current_object, current_question = current

    if (
           interaction is None
        or interaction.request.requester is not current_object
    ):
        return ResolutionOutcome(
            result=ResolutionResult.INTERACTION_REQUESTED,
            interaction=InteractionRequest(
                message=current_question,
                expected_type="str",
                slot=resolution_context.slot,
                requester=current_object
            )
        )

    assert interaction.answer.consumed is False
    user_answer = interaction.answer.content
    interaction.answer.consumed = True

    return ask_helper_no_interaction_slot_resolver(
        runtime_context, current, resolution_context, user_answer
    )

def ask_helper_error_resolver(
        runtime_context: LLMRuntimeContext,
        current: tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str],
        resolution_context: ResolutionContext,
        intent: Intent,
        interaction: Interaction | None = None) -> ResolutionOutcome:

    current_object, current_question = current

    if (
           interaction is None
        or interaction.request.requester is not current_object
    ):
        return ResolutionOutcome(
            result=ResolutionResult.INTERACTION_REQUESTED,
            interaction=InteractionRequest(
                message=current_question,
                expected_type="str",
                slot=resolution_context.slot,
                requester=current_object
            )
        )

    assert interaction.answer.consumed is False
    user_answer = interaction.answer.content
    interaction.answer.consumed = True

    return ask_helper_no_interaction_error_resolver(
        runtime_context, current, resolution_context, user_answer, intent
    )

def _ask_helper_no_interaction(
        system_prompt: str,
        current: tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str],
        resolution_context: ResolutionContext,
        resolution_text: str
) -> ResolutionOutcome:

    answer = call_airlock_model_server(
        model=Model.Phi4MiniInstruct,
        adapter="intent-sequencer",
        messages=[
            Message(role=Role.system, content=system_prompt),
            Message(role=Role.user, content=resolution_text)
        ],
        parameters=GenerationParameters(
            max_new_tokens=1024,
            do_sample=False
        ),
        container_name="dev-phi"
    )

    resolution_context.llm_call_logs.append(
        LLMCallLog(
            description=f"ask_helper[{current}]",
            system_prompt=system_prompt,
            assistant=resolution_text,
            answer=answer
        )
    )

    parsed_dsl = parser.parse_dsl(answer)

    return ResolutionOutcome(
        nodes=parsed_dsl.get_children(),
        result=ResolutionResult.NEW_DSL_NODES
    )

def ask_helper_no_interaction_slot_resolver(
        runtime_context: LLMRuntimeContext,
        current: tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str],
        resolution_context: ResolutionContext,
        user_answer: str
) -> ResolutionOutcome:

    current_object, current_question = current

    previous_qna_block = resolution_context.format_previous_qna_block()
    # intent and slot can be None if for example the user only ask a question without
    # mentioning any intent at all.
    intent_name = resolution_context.intent.name if resolution_context.intent else "none"
    slot_name = resolution_context.slot.name if resolution_context.slot else "none"

    resolution_text = f"""resolution_context:
  intent: {intent_name}
  slot: {slot_name}
{previous_qna_block}
  current_question: {current_question}
  current_user_answer: {user_answer}"""

    resolution_context.questions_being_clarified.append(
        (current_object, current_question, user_answer)
    )

    return _ask_helper_no_interaction(
        runtime_context.system_prompt_slot_resolver, current, resolution_context, resolution_text
    )

def ask_helper_no_interaction_error_resolver(
        runtime_context: LLMRuntimeContext,
        current: tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str],
        resolution_context: ResolutionContext,
        user_answer: str,
        intent: Intent
) -> ResolutionOutcome:

    current_object, error = current

    previous_qna_block = resolution_context.format_previous_qna_block()
    # intent and slot can be None if for example the user only ask a question without
    # mentioning any intent at all.

    resolution_text = f"""resolution_context:
  intent: {intent.to_dsl_representation()}
{previous_qna_block}
  error: {error}
  current_user_answer: {user_answer}"""

    resolution_context.questions_being_clarified.append(
        (current_object, error, user_answer)
    )

    return _ask_helper_no_interaction(
        runtime_context.system_prompt_error_resolver, current, resolution_context, resolution_text
    )

def ask_helper_no_interaction_intent_sequencer(
        runtime_context: LLMRuntimeContext,
        current: tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str],
        resolution_context: ResolutionContext,
        gathered_data: str
) -> ResolutionOutcome:

    current_object, current_question = current

    resolution_text = f"""{current_question}

Here is the data you should use to generate the intents:
{gathered_data}"""

    resolution_context.questions_being_clarified.append(
        (current_object, current_question, gathered_data)
    )

    return _ask_helper_no_interaction(
        runtime_context.system_prompt_intent_sequencer, current, resolution_context, resolution_text
    )
