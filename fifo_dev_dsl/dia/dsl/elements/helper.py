from __future__ import annotations
from typing import TYPE_CHECKING

from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server

from fifo_dev_dsl.dia.dsl.parser import parser
from fifo_dev_dsl.dia.resolution.llm_call_log import LLMCallLog
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.dsl.elements.intent_runtime_error_resolver import IntentRuntimeErrorResolver
    from fifo_dev_dsl.dia.dsl.elements.query_gather import QueryGather
    from fifo_dev_dsl.dia.dsl.elements.query_user import QueryUser
    from fifo_dev_dsl.dia.dsl.elements.ask import Ask
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext


def ask_helper_no_interaction(
        runtime_context: LLMRuntimeContext,
        system_prompt: str,
        current: tuple[IntentRuntimeErrorResolver | Ask | QueryUser | QueryGather, str],
        resolution_context: ResolutionContext,
        resolution_text: str,
        gatherered_data_or_user_answer: str
) -> ResolutionOutcome:
    """Resolve follow-up questions without further interaction."""

    resolution_context.questions_being_clarified.append(
        (*current, gatherered_data_or_user_answer)
    )

    answer = call_airlock_model_server(
        model=runtime_context.base_model,
        adapter=runtime_context.intent_sequencer_adapter,
        messages=[
            Message.system(system_prompt),
            Message.user(resolution_text)
        ],
        parameters=GenerationParameters(
            max_new_tokens=1024,
            do_sample=False
        ),
        container_name=runtime_context.container_name
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
