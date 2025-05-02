from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Tuple

from common.llm.airlock_model_env.common.models import GenerationParameters, Message, Model, Role
from common.llm.airlock_model_env.sdk.client_sdk import call_airlock_model_server

import common.llm.dia.dsl.parser.parser as parser
from common.llm.dia.resolution.enums import ResolutionKind, ResolutionResult
from common.llm.dia.resolution.interaction import Interaction, InteractionRequest
from common.llm.dia.resolution.outcome import ResolutionOutcome
from common.llm.dia.dsl.elements.query_user import QueryUser
from common.llm.dia.resolution.resolver import AbortBehavior, resolve

if TYPE_CHECKING:
    from common.llm.dia.dsl.elements.ask import Ask
    from common.llm.dia.runtime.context import LLMRuntimeContext
    from common.llm.dia.resolution.context import ResolutionContext


def ask_helper(runtime_context: LLMRuntimeContext,
               current: Tuple[Union[Ask, QueryUser], str],
               kind: set[ResolutionKind],
               resolution_context: ResolutionContext,
               interaction: Optional[Interaction] = None) -> ResolutionOutcome:

    current_object, current_question = current

    if len(resolution_context.questions_being_clarified) > 0:
        _source_object, source_question, _ = resolution_context.questions_being_clarified[0]
    else:
        _source_object, source_question = current

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
            ),
            resolved=current_object,
            propagate_slots=[]
        )

    assert interaction.answer.consumed is False
    user_answer = interaction.answer.content
    interaction.answer.consumed = True

#   previous_questions_and_answers:
#     - question: ...
#       answer: ...
#     - question: ...
#       answer: ...

    if resolution_context.questions_being_clarified:
        previous_qna_yaml = "\n".join(
            f"    - question: {q}\n      answer: {a}" for _, q, a in resolution_context.questions_being_clarified
        )
        previous_qna_block = f"  previous_questions_and_answers:\n{previous_qna_yaml}"
    else:
        previous_qna_block = "  previous_questions_and_answers: []"

    resolution_text = f"""resolution_context:
  intent: {resolution_context.intent}
  slot: {resolution_context.slot}
{previous_qna_block}
  current_question: {source_question}
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

    print("ask_helper", current)
    print("---")
    print("$")
    print(runtime_context.system_prompt_slot_resolver)
    print(">")
    print(resolution_text)
    print("<")
    print(answer)
    print("---")

    resolution_context = resolution_context.deepcopy()

    resolution_context.questions_being_clarified.append((current_object, current_question, user_answer))

    parsed_dsl = parser.parse_dsl(answer)

    print("parsed dsl", parsed_dsl)

    resolved = resolve(
                dsl_elements=parsed_dsl,
                runtime_context=runtime_context,
                resolution_context=resolution_context,
                resolution_kind=kind,
                abort_behavior=AbortBehavior.ABORT,
                interaction=interaction
    )

    print("resolved dsl", resolved)

    assert(len(resolved.resolved) == 1)
    resolved.resolved = resolved.resolved[0]

    return resolved

    # parsed_dsl_values: list[DslBase] = []
    # parsed_dsl_propagations: list[DslBase] = []
    # parsed_dsl_other: list[DslBase] = []
    # for dsl_element in parsed_dsl:
    #     if isinstance(dsl_element, DSLValueBase):
    #         parsed_dsl_values.append(dsl_element)
    #     elif isinstance(dsl_element, PropagateSlot):
    #         parsed_dsl_propagations.append(dsl_element)
    #     else:
    #         parsed_dsl_other.append(dsl_element)

    # assert len(parsed_dsl_values) <= 1  # to be handled later

    # if len(parsed_dsl_values) > 0:
    #     outcome = parsed_dsl[0].resolve(runtime_context, kind, context, interaction)
    # else:
    #     outcome = ResolutionOutcome(
    #         result=ResolutionResult.NOT_APPLICABLE,
    #         resolved=None,
    #         propagate_slots=[]
    #     )

    # parsed_dsl_other_outcome: list[ResolutionOutcome] = []
    # for dsl_element_other in parsed_dsl_other:
    #     if isinstance(dsl_element_other, (Abort, AbortWithNewIntent)):
    #         return dsl_element_other.resolve(runtime_context, kind, context, interaction)

    #     parsed_dsl_other_outcome.append(dsl_element_other.resolve(runtime_context, kind, context, interaction))

    # # at this stage... parsed_dsl_other must be 0 or 1 element
    # assert len(parsed_dsl_other_outcome) <= 1
    # if len(parsed_dsl_other_outcome) == 1:
    #     if isinstance(parsed_dsl_other_outcome[0].resolved, (QueryUser, QueryFill)):
    #         # if we have QueryUser, QueryFill, then value must be 0
    #         assert len(parsed_dsl_values) == 0
    #         outcome = ResolutionOutcome(
    #             result=ResolutionResult.APPLICABLE_SUCCESS,
    #             resolved=parsed_dsl_other_outcome[0].resolved,
    #             propagate_slots=[]
    #         )
    #     else:
    #         raise RuntimeError(f"unexpected DSL element '{type(parsed_dsl_other_outcome[0].resolved)}' in Ask")

    # outcome_rnp_combined = ResolutionOutcomeResultAndPropagationSlot(result=outcome.result, propagate_slots=[])

    # for dsl_element_propagation in parsed_dsl_propagations:
    #     tmp = dsl_element_propagation.resolve(runtime_context, kind, context, interaction)
    #     outcome_rnp_combined.propagate_slots.append(tmp.resolved) # here we move the resolved dsl_element_propagation into the propagated slots
    #     outcome_rnp_combined.result = outcome_rnp_combined.result.combine(tmp.result)

    # return ResolutionOutcome(
    #     result=outcome_rnp_combined.result,
    #     resolved=outcome.resolved,
    #     propagate_slots=outcome_rnp_combined.propagate_slots
    # )
