from __future__ import annotations
from typing import TYPE_CHECKING
import re
from dataclasses import dataclass

from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message, Model, Role
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
import fifo_dev_dsl.dia.dsl.elements.helper as helper
from fifo_dev_dsl.dia.resolution.context import LLMCallLog
from fifo_dev_dsl.dia.resolution.enums import AbortBehavior
from fifo_dev_dsl.dia.resolution.interaction import Interaction
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext


@dataclass
class QueryGather(DslBase):

    original_intent: str
    query: str

    def is_resolved(self) -> bool:
        return False

    def do_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None) -> ResolutionOutcome:
        super().do_resolution(runtime_context, resolution_context, abort_behavior, interaction)

        prompt_user = runtime_context.get_user_prompt_dynamic_query(resolution_context, self.query)

        answer = call_airlock_model_server(
                    model=Model.Phi4MiniInstruct,
                    messages=[
                        Message(
                            role=Role.system,
                            content=runtime_context.system_prompt_query_gather
                        ),
                        Message(
                            role=Role.user,
                            content=prompt_user
                        )
                    ],
                    parameters=GenerationParameters(
                        max_new_tokens=1024,
                        do_sample=False
                    ),
                    container_name="dev-phi"
                )

        resolution_context.llm_call_logs.append(
            LLMCallLog(
                description="QueryGather[do_resolution]",
                system_prompt=runtime_context.system_prompt_query_gather,
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

        return helper.ask_helper_no_interaction_intent_sequencer(
            runtime_context, (self, self.original_intent), resolution_context, value
        )
