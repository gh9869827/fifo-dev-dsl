from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Tuple

from common.llm.airlock_model_env.common.models import GenerationParameters, Message, Model, Role
from common.llm.airlock_model_env.sdk.client_sdk import call_airlock_model_server

import common.llm.dia.dsl.parser.parser as parser
from common.llm.dia.resolution.context import LLMCallLog
from common.llm.dia.resolution.enums import ResolutionResult
from common.llm.dia.resolution.interaction import Interaction, InteractionRequest
from common.llm.dia.resolution.outcome import ResolutionOutcome
from common.llm.dia.dsl.elements.query_user import QueryUser

if TYPE_CHECKING:
    from common.llm.dia.dsl.elements.ask import Ask
    from common.llm.dia.runtime.context import LLMRuntimeContext
    from common.llm.dia.resolution.context import ResolutionContext


def ask_helper(runtime_context: LLMRuntimeContext,
               current: Tuple[Union[Ask, QueryUser], str],
               resolution_context: ResolutionContext,
               interaction: Optional[Interaction] = None) -> ResolutionOutcome:

    current_object, current_question = current

    if (
           interaction is None
        or interaction is not None and interaction.request.requester is not current_object
    ):
        return ResolutionOutcome(
            result=ResolutionResult.INTERACTION_REQUESTED,
            interaction=InteractionRequest(
                message=current_question,
                expected_type="str",
                slot_name=resolution_context.slot,
                requester=current_object
            )
        )

    assert interaction.answer.consumed is False
    user_answer = interaction.answer.content
    interaction.answer.consumed = True

    if resolution_context.questions_being_clarified:
        previous_qna_yaml = "\n".join(
            f"    - question: {q}\n      answer: {a}" for _, q, a in resolution_context.questions_being_clarified
        )
        previous_qna_block = f"  previous_questions_and_answers:\n{previous_qna_yaml}"
    else:
        previous_qna_block = "  previous_questions_and_answers: []"

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

    answer = call_airlock_model_server(
        model=Model.Phi4MiniInstruct,
        adapter="intent-sequencer",
        messages=[
            Message(role=Role.system, content=runtime_context.system_prompt_slot_resolver),
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
            system_prompt=runtime_context.system_prompt_slot_resolver,
            assistant=resolution_text,
            answer=answer
        )
    )

    resolution_context.questions_being_clarified.append(
        (current_object, current_question, user_answer)
    )

    parsed_dsl = parser.parse_dsl(answer)

    return ResolutionOutcome(
        node=parsed_dsl.get_children(),
        result=ResolutionResult.NEW_DSL_NODES
    )
