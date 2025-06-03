from __future__ import annotations
import re
from typing import TYPE_CHECKING

from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message, Model, Role
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server
from fifo_dev_dsl.dia.resolution.context import LLMCallLog
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.resolution.enums import AbortBehavior, ResolutionResult
from fifo_dev_dsl.dia.resolution.interaction import Interaction
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome
from fifo_dev_dsl.dia.dsl.elements.value import Value

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class QueryFill(DslBase):

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
                            content=runtime_context.system_prompt_query_fill
                        ),
                        Message(
                            role=Role.user,
                            content=prompt_user
                        ),
                    ],
                    parameters=GenerationParameters(
                        max_new_tokens=1024,
                        do_sample=False
                    ),
                    container_name="dev-phi"
                )

        resolution_context.llm_call_logs.append(
            LLMCallLog(
                description="QueryFill[do_resolution]",
                system_prompt=runtime_context.system_prompt_query_fill,
                assistant=prompt_user,
                answer=answer
            )
        )

        match = re.search(
            r"reasoning:\s*(.*?)\nvalue:\s*(.*?)\nabort:\s*(.*)",
            answer,
            flags=re.DOTALL
        )

        if match:
            # If 'abort:' is non-empty, raise; otherwise use extracted value
            if match[3].strip():
                raise RuntimeError("QueryFill failed: abort message was returned")

            value = match[2].strip()
        else:
            value = "unknown"

        return ResolutionOutcome(
            result=ResolutionResult.NEW_DSL_NODES,
            node=[Value(value)]
        )
