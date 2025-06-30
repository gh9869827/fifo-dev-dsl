from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass

from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server

from fifo_dev_dsl.dia.resolution.llm_call_log import LLMCallLog
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome
from fifo_dev_dsl.dia.dsl.elements.value import Value
from fifo_dev_dsl.common.dsl_utils import quote_and_escape

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.resolution.interaction import Interaction
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class QueryFill(DslBase):
    """
    Compute or infer a missing value using LLM-based reasoning.

    This node is used when a required value cannot be determined from local context alone.
    During resolution, it triggers a call to a large language model, passing a structured
    prompt derived from the current intent, prior state, and relevant runtime information.

    Unlike `ASK`, which delegates to the user, `QueryFill` invokes autonomous reasoning—
    for example, querying an LLM with access to inventory data, system state, or sensor outputs.

    Once resolved, this node is replaced with a concrete value such as `Value("12")`.

    Attributes:
        query (str):
            A short description of the desired value, e.g., "longest screw available".

    Example:
        retrieve_screw(count=6, length=QUERY_FILL("longest length you have"))
    """

    query: str

    def is_resolved(self) -> bool:
        """
        Indicate that this node has not yet been resolved.

        `QueryFill` nodes require external inference to produce a value.
        Until they are replaced with a concrete result (e.g., a `Value`),
        they are considered unresolved.

        Returns:
            bool:
                Always `False`.
        """
        return False

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the QueryFill node.

        Returns:
            str:
                The query in DSL syntax, with internal quotes escaped and the value properly quoted.
                For example: QUERY_FILL("query").
        """
        return f'QUERY_FILL({quote_and_escape(self.query)})'

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Raise a RuntimeError because QueryFill nodes are unresolved.

        These nodes must be replaced by a concrete Value during resolution.
        Attempting to evaluate them directly indicates that resolution has
        not completed successfully.

        Raises:
            RuntimeError: Always raised with the message
                Unresolved DSL node: QueryFill
        """
        raise RuntimeError(f"Unresolved DSL node: {self.__class__.__name__}")

    async def eval_async(
        self,
        runtime_context: LLMRuntimeContext,
    ) -> Any:
        """
        Asynchronously raise a RuntimeError because `QueryFill` nodes are
        unresolved.

        These nodes must be replaced by a concrete `Value` during resolution.
        Attempting to evaluate them directly indicates that resolution has not
        completed successfully.

        Raises:
            RuntimeError: Always raised with the message
                Unresolved DSL node: QueryFill
        """

        raise RuntimeError(f"Unresolved DSL node: {self.__class__.__name__}")

    def do_resolution(
        self,
        runtime_context: LLMRuntimeContext,
        resolution_context: ResolutionContext,
        interaction: Interaction | None,
    ) -> ResolutionOutcome:
        super().do_resolution(runtime_context, resolution_context, interaction)

        prompt_user = runtime_context.get_user_prompt_dynamic_query(resolution_context, self.query)

        answer = call_airlock_model_server(
                    model=runtime_context.base_model,
                    messages=[
                        Message.system(runtime_context.system_prompt_query_fill),
                        Message.user(prompt_user),
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
            nodes=[Value(value)]
        )
