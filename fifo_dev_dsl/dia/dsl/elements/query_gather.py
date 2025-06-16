from __future__ import annotations
from typing import TYPE_CHECKING, Any
import re
from dataclasses import dataclass

from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server

from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.elements.helper import ask_helper_no_interaction
from fifo_dev_dsl.dia.resolution.llm_call_log import LLMCallLog
from fifo_dev_dsl.common.dsl_utils import quote_and_escape

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.resolution.interaction import Interaction
    from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
    from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext


@dataclass
class QueryGather(DslBase):
    """
    Collect contextual information to begin generating a complete intent.

    This node is used when the system lacks sufficient information to construct
    a well-formed intent from the user's request. Unlike `QueryFill`, which resolves
    one missing slot at a time, `QueryGather` performs a broader, one-shot inference
    to extract multiple interdependent values simultaneously.

    It sends the `query` to the runtime reasoning engine (e.g., an LLM) based on the
    original user request. Once the information is retrieved, the DSL can initiate
    intent generation accordingly.

    This is especially useful when slot values are tightly coupled — for example,
    when both `length` and `count` must be derived together to avoid inconsistent logic.

    Attributes:
        original_intent (str):
            The user's natural language request.

        query (str):
            A clarification or refinement question used to gather context from the runtime engine.

    Example:
        Input:  "Give me all the shortest screws in the inventory"
        Output: QUERY_GATHER(
                    "Give me all the shortest screws in the inventory",
                    "Shortest screw length and count in the inventory"
                )
    """

    original_intent: str
    query: str

    def is_resolved(self) -> bool:
        """
        Indicate that this placeholder has not yet been expanded.

        ``QUERY_GATHER`` nodes collect additional information used to refine the
        user's intent. They stay unresolved until their query is executed and
        the resulting data is inserted.

        Returns:
            bool:
                Always ``False``.
        """
        return False

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the QueryGather node.

        Returns:
            str:
                The query in DSL syntax, with internal quotes escaped and each argument
                properly quoted. For example: QUERY_GATHER("intent", "query").
        """
        i = quote_and_escape(self.original_intent)
        q = quote_and_escape(self.query)
        return f'QUERY_GATHER({i}, {q})'

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
        value_type: MiniDocStringType | None = None,
    ) -> Any:
        """
        Raise a RuntimeError because QueryGather nodes are unresolved.

        These placeholders must be replaced with fully specified intents during
        resolution. Encountering one during evaluation indicates that resolution
        has not completed successfully.

        Raises:
            RuntimeError: Always raised with the message
                Unresolved DSL node: QueryGather
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
                        Message.system(runtime_context.system_prompt_query_gather),
                        Message.user(prompt_user)
                    ],
                    parameters=GenerationParameters(
                        max_new_tokens=1024,
                        do_sample=False
                    ),
                    container_name=runtime_context.container_name
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
            gathered_data = match[2].strip()
        else:
            gathered_data = "unknown"

        resolution_text = f"""{self.original_intent}

Here is the data you should use to generate the intents:
{gathered_data}"""

        return ask_helper_no_interaction(
            runtime_context,
            runtime_context.system_prompt_intent_sequencer,
            (self, self.original_intent),
            resolution_context,
            resolution_text,
            gathered_data
        )
