from __future__ import annotations
from typing import TYPE_CHECKING
import textwrap

from fifo_dev_common.containers.read_only.read_only_list import ReadOnlyList
from fifo_dev_common.introspection.tool_decorator import ToolHandler, ToolQuerySource

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext

class LLMRuntimeContext:
    """
    Runtime context passed to all DSL elements during resolution and evaluation.

    It contains:
      - A list of available tools (`ToolHandler`) that can be invoked via DSL intents
      - A list of query sources used to respond to QUERY_FILL, QUERY_USER and QUERY_GATHER phases
      - Precompiled LLM prompt templates for specific phases:
          - intent sequencing
          - slot resolution
          - query fill (autofill from context)
          - query user (responding to user questions)
          - query gather (automatically gathering details to properly deduce intents)

    This context acts as the main registry for tools, enabling type-safe evaluation
    of resolved intents using their documented parameter and return types. It is also
    responsible for generating structured tool documentation that gets embedded into
    LLM prompts. This documentation helps the model understand which tools exist,
    what parameters they accept, and how to respond accurately when sequencing or
    resolving intents.
    """

    _tools: ReadOnlyList[ToolHandler]
    _tool_name_to_tool: dict[str, ToolHandler]
    _query_sources: ReadOnlyList[ToolQuerySource]

    _prompt_query_fill: str
    _prompt_query_user: str
    _prompt_query_gather: str
    _prompt_intent_sequencer: str
    _prompt_slot_resolver: str
    _prompt_error_resolver: str

    def __init__(self, tools: list[ToolHandler], query_sources: list[ToolQuerySource]):
        """
        Initialize the runtime context with tools and query sources.

        Args:
            tools (list[ToolHandler]):
                List of tools available for use in function execution.

            query_sources (list[ToolQuerySource]):
                Sources that can be queried at runtime to answer user or slot-filling questions.
        """
        self._tools = ReadOnlyList(tools)
        self._query_sources = ReadOnlyList(query_sources)

        yaml_tools = "\n".join(tool.to_schema_yaml() for tool in self._tools)
        yaml_sources = "\n".join(source.get_description() for source in self._query_sources)

        yaml_info = yaml_tools, yaml_sources
        self._prompt_query_fill = self._precompile_prompt_query_fill(*yaml_info)
        self._prompt_query_user = self._precompile_prompt_query_user(*yaml_info)
        self._prompt_query_gather = self._precompile_prompt_query_gather(*yaml_info)
        self._prompt_intent_sequencer = self._precompile_prompt_intent_sequencer(*yaml_info)
        self._prompt_slot_resolver = self._precompile_prompt_slot_resolver(*yaml_info)
        self._prompt_error_resolver = self._precompile_prompt_error_resolver(*yaml_info)

        self._tool_name_to_tool = { t.tool_name: t for t in self._tools }

    def get_tool(self, name: str) -> ToolHandler:
        """
        Retrieve a tool by name.

        Args:
            name (str):
                The tool name.

        Returns:
            ToolHandler:
                The tool instance.
        """
        return self._tool_name_to_tool[name]

    @property
    def system_prompt_query_fill(self) -> str:
        """
        System prompt used for QUERY_FILL resolution.

        Returns:
            str:
                The precompiled system prompt to query runtime sources to fill missing information.
        """
        return self._prompt_query_fill

    def get_user_prompt_dynamic_query(self, resolution_context: ResolutionContext, question: str) -> str:
        """
        Dynamically create the user prompt used for QUERY_FILL, QUERY_USER and QUERY_GATHER
        resolution.

        Args:
            context (ResolutionContext):
                The current resolution context containing the intent, slot, and other contextual
                information.

            question (str):
                The user-facing clarification question to inject in the generated prompt.

        Returns:
            str:
                The dynamically created user prompt to query runtime sources to fill missing
                information.
        """
        dynamic_runtime_info = "\n".join(
            source.get_description() for source in self._query_sources
        )

        # intent and slot can be None if for example the user only ask a question without
        # mentioning any intent at all.
        intent_name = resolution_context.intent.name if resolution_context.intent else "none"
        slot_name = resolution_context.slot.name if resolution_context.slot else "none"

        return f"""query context:
  intent: {intent_name}
  slot: {slot_name}
{resolution_context.format_other_slots_yaml('  ')}
  question: {question}
  runtime_information:
{textwrap.indent(dynamic_runtime_info, prefix='    ')}
"""

    @property
    def system_prompt_query_user(self) -> str:
        """
        System prompt used for QUERY_USER resolution.

        Returns:
            str:
                The precompiled system prompt to respond to user-initiated exploratory questions.
        """
        return self._prompt_query_user

    @property
    def system_prompt_query_gather(self) -> str:
        """
        System prompt used for QUERY_GATHER resolution.

        Returns:
            str:
                The precompiled system prompt to respond to gather queries.
        """
        return self._prompt_query_gather

    @property
    def system_prompt_intent_sequencer(self) -> str:
        """
        System prompt used to sequence atomic intents from user input.

        Returns:
            str:
                The precompiled system prompt used to guide intent extraction.
        """
        return self._prompt_intent_sequencer

    @property
    def system_prompt_slot_resolver(self) -> str:
        """
        System prompt used to resolve individual slots in the current intent.

        Returns:
            str:
                The precompiled system prompt used during slot-level resolution.
        """
        return self._prompt_slot_resolver

    @property
    def system_prompt_error_resolver(self) -> str:
        """
        System prompt used to resolve error in the current intent.

        Returns:
            str:
                The precompiled system prompt used during error resolution.
        """
        return self._prompt_error_resolver

    # formatting prompts
    # pylint: disable=line-too-long

    def _get_sources(self, yaml_sources: str) -> str:
        if yaml_sources != "":
            return f"""You have access to the following sources that can be queried to fill in missing information using QUERY_FILL:\n{yaml_sources}"""

        return "QUERY_FILL cannot be used as no information can be retrieved at runtime."

    def _precompile_prompt_query_fill(self, yaml_tools_definition: str, _yaml_sources: str):
        return f"""You are a precise agent that answers user questions according to the scope defined by the intents below:

{yaml_tools_definition}

answer on three lines as follows:
reasoning: your reasoning to answer the question. Clearly investigate each item that is provided in the 'query context' section with a special attention to the 'runtime_information' section. Pay special attention to the type you return. If the user asks for a single value, and multiple ones can be return, only return one.
value: the value of the requested slot. Only include the value, no explanation. When return a list use [...].
abort: if the answer to the question cannot be deduced, include the error message here"""

    def _precompile_prompt_query_user(self, yaml_tools_definition: str, _yaml_sources: str):
        return f"""You are a precise agent that answers user questions according to the scope defined by the intents below:

{yaml_tools_definition}

answer on two lines as follows:
reasoning: your reasoning to answer the question. Clearly investigate each item that is provided in the 'query context' section with a special attention to the 'runtime_information' section. Pay special attention to the type you return. If the user asks for a single value, and multiple ones can be return, only return one.
user friendly answer: the value of the requested slot. Include the value, and just enough explanation like if you are takling to a colleague asking a question and who is in a hurry. if the answer to the question cannot be deduced, include the error message here"""

    def _precompile_prompt_query_gather(self, yaml_tools_definition: str, _yaml_sources: str):
        return f"""You are a precise agent that answers questions according to the scope defined by the intents below:

{yaml_tools_definition}

answer on two lines as follows:
reasoning: your reasoning to answer the question. Clearly investigate each item that is provided in the 'query context' section with a special attention to the 'runtime_information' section. Pay special attention to the type you return. If the user asks for a single value, and multiple ones can be return, only return one.
user friendly answer: the detailled answer to the question. if the answer to the question cannot be deduced, include the error message here"""

    def _precompile_prompt_intent_sequencer(self, yaml_tools_definition: str, yaml_sources: str):
        return f"""You are a precise intent sequencer. You parse the user's prompt and split it into atomic intents that match one of the defined intents below:

{yaml_tools_definition}

{self._get_sources(yaml_sources)}"""

    def _precompile_prompt_slot_resolver(self, yaml_tools_definition: str, yaml_sources: str):
        return f"""You are a precise slot resolver. You resolve one slot at a time based on the current resolution context, but the user may change or override the task. Here are the available intents:

{yaml_tools_definition}

If the user's answer does not directly resolve to a value, return a QUERY_FILL(...), QUERY_USER(...) or a follow-up ASK(...).

{self._get_sources(yaml_sources)}"""

    def _precompile_prompt_error_resolver(self, yaml_tools_definition: str, yaml_sources: str):
        return f"""You are a precise error resolver. You resolve one error at a time based on the current resolution context, but the user may change or override the task. Here are the available intents:

{yaml_tools_definition}

If the user's answer does not directly resolve to a value, return a QUERY_FILL(...), QUERY_USER(...) or a follow-up ASK(...).

{self._get_sources(yaml_sources)}"""
