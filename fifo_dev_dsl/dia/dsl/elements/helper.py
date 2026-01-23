from __future__ import annotations
from typing import TYPE_CHECKING

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

    answer = runtime_context.call_llm_dsl(
        system_prompt=system_prompt,
        user_prompt=resolution_text
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
