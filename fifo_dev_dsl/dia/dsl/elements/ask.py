from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

from common.llm.dia.dsl.elements.base import DslBase
import common.llm.dia.dsl.elements.helper as helper
from common.llm.dia.resolution.enums import AbortBehavior
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext

@dataclass
class Ask(DslBase):

    question: str

    def do_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None) -> ResolutionOutcome:
        super().do_resolution(runtime_context, resolution_context, abort_behavior, interaction)

        return helper.ask_helper_slot_resolver(
            runtime_context=runtime_context,
            current=(self, self.question),
            resolution_context=resolution_context,
            interaction=interaction)

    def is_resolved(self) -> bool:
        return False
