from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

from dataclasses import dataclass

from common.introspection.docstring import MiniDocStringType
from common.llm.dia.dsl.elements.base import DslBase
from common.llm.dia.dsl.elements.value_base import DSLValueBase
from common.llm.dia.resolution.enums import ResolutionKind, ResolutionResult
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext

@dataclass
class Intent(DslBase):

    name: str
    params: dict[str, DslBase]

    def resolve(self,
                runtime_context: LLMRuntimeContext,
                kind: set[ResolutionKind],
                context: ResolutionContext,
                interaction: Optional[Interaction] = None) -> ResolutionOutcome:

        new_params: dict[str, DslBase] = {}
        resolutions_result = ResolutionResult.NOT_APPLICABLE

        skip = False
        for key, val in self.params.items():
            if skip:
                new_params[key] = val
                continue

            ctx = context.deepcopy()
            ctx.intent = self.name
            ctx.slot = key
            ctx.other_slots = {}

            # todo we need to combine self.params and new_params with priority on new_params and if not found go back to self.params
            for other_key, other_val in self.params.items():
                if (
                    not other_key is key
                    and other_val.is_resolved()
                    and isinstance(other_val, DSLValueBase)
                ):
                    ctx.other_slots[other_key] = other_val.get_resolved_value_as_text()

            outcome = val.resolve(runtime_context, kind, ctx, interaction)

            print("intent resolution outcome", outcome)

            resolutions_result = resolutions_result.combine(outcome.result)

            if outcome.result is ResolutionResult.ABORT:
                return outcome

            if outcome.result is ResolutionResult.INTERACTION_REQUESTED:
                skip = True

            if outcome.resolved is None:
                raise RuntimeError(f"Intent '{self.name}': resolved value for key '{key}' is None")

            new_params[key] = outcome.resolved

            # todo fix we need to take new_params and not self.params
            for p in outcome.propagate_slots:
                for pk, pv in p.slots.items():
                    self.params[pk] = pv

        return ResolutionOutcome(
            result=resolutions_result,
            resolved=Intent(self.name, new_params),
            propagate_slots=[],
            interaction=None if skip is False else outcome.interaction
        )

    def is_resolved(self) -> bool:
        return all(val.is_resolved() for val in self.params.values())

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: Optional[MiniDocStringType] = None) -> Any:

        tool = runtime_context.get_tool(self.name)

        args = {
            param_name: param_value.eval(
                runtime_context, tool.tool_docstring.get_arg_by_name(param_name).pytype
            ) for param_name, param_value in self.params.items()
        }

        ret = tool.tool_docstring.return_type.cast(tool(**args))

        return value_type.cast(ret) if value_type is not None else ret
