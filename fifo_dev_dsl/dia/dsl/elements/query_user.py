from __future__ import annotations
import re
from typing import TYPE_CHECKING

from dataclasses import dataclass

from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message, Model, Role
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server
import fifo_dev_dsl.dia.dsl.elements.helper as helper
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.resolution.context import LLMCallLog
from fifo_dev_dsl.dia.resolution.enums import AbortBehavior
from fifo_dev_dsl.dia.resolution.interaction import Interaction
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext


@dataclass
class QueryUser(DslBase):

    query: str

    def is_resolved(self) -> bool:
        return False

    def do_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None) -> ResolutionOutcome:
        super().do_resolution(runtime_context, resolution_context, abort_behavior, interaction)

        if interaction is not None and interaction.request.requester is self:
            return helper.ask_helper_slot_resolver(
                runtime_context=runtime_context,
                current=(self, interaction.request.message),
                resolution_context=resolution_context,
                interaction=interaction
            )

        prompt_user = runtime_context.get_user_prompt_dynamic_query(resolution_context, self.query)

        answer = call_airlock_model_server(
                    model=Model.Phi4MiniInstruct,
                    messages=[
                        Message(
                            role=Role.system,
                            content=runtime_context.system_prompt_query_user
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
                description="QueryUser[do_resolution]",
                system_prompt=runtime_context.system_prompt_query_user,
                assistant=prompt_user,
                answer=answer
            )
        )

        for line in answer.splitlines():
            if line.startswith("user friendly answer: "):
                value = line[len("user friendly answer: "):].strip()
                # handle multiple values and enforce reasoning
                break

        return helper.ask_helper_slot_resolver(
            runtime_context=runtime_context,
            current=(self, value),
            resolution_context=resolution_context,
            interaction=interaction
        )
