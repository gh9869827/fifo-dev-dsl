from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from dataclasses import dataclass

from common.llm.airlock_model_env.common.models import GenerationParameters, Message, Model, Role
from common.llm.airlock_model_env.sdk.client_sdk import call_airlock_model_server
from common.llm.dia.dsl.elements.base import DslBase
from common.llm.dia.resolution.enums import ResolutionKind, ResolutionResult
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.outcome import ResolutionOutcome
from common.llm.dia.dsl.elements.value import Value

if TYPE_CHECKING:
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext

@dataclass
class QueryFill(DslBase):
    query: str

    def is_resolved(self) -> bool:
        return False

    def resolve(self,
                runtime_context: LLMRuntimeContext,
                kind: set[ResolutionKind],
                context: ResolutionContext,
                interaction: Optional[Interaction] = None) -> ResolutionOutcome:

        if ResolutionKind.QUERY_FILL not in kind:
            return super().resolve(runtime_context, kind, context, interaction)

        prompt_user = runtime_context.get_user_prompt_dynamic_query(context, self.query)

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

        print("---")
        print("$")
        print(runtime_context.system_prompt_query_user)
        print(">")
        print(prompt_user)
        print("<")
        print(answer)
        print("---")

        for line in answer.splitlines():
            if line.startswith("value: "):
                value = line[len("value: "):].strip()
                # handle multiple values and enforce reasoning
                break

        outcome = Value(value).resolve(runtime_context, kind, context, interaction)

        return ResolutionOutcome(
            result=ResolutionResult.APPLICABLE_SUCCESS.combine(outcome.result),
            resolved=outcome.resolved,
            propagate_slots=outcome.propagate_slots
        )
