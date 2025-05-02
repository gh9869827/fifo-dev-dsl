from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

from common.llm.dia.dsl.elements.base import DslBase
import common.llm.dia.dsl.elements.helper as helper
from common.llm.dia.resolution.enums import ResolutionKind
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext

@dataclass
class Ask(DslBase):

    question: str

    def resolve(self,
                runtime_context: LLMRuntimeContext,
                kind: set[ResolutionKind],
                context: ResolutionContext,
                interaction: Optional[Interaction] = None) -> ResolutionOutcome:

        if ResolutionKind.ASK not in kind:
            return super().resolve(runtime_context, kind, context, interaction)

        return helper.ask_helper(
            runtime_context=runtime_context,
            current=(self, self.question),
            kind=kind,
            resolution_context=context,
            interaction=interaction)

    def is_resolved(self) -> bool:
        return False
