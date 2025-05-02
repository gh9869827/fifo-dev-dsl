from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

from common.llm.dia.dsl.elements.base import DslBase
from common.llm.dia.resolution.enums import ResolutionKind, ResolutionResult
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext

@dataclass
class Abort(DslBase):
    """
    A control directive that halts intent resolution and prevents execution.

    When encountered, this node returns a ResolutionResult.ABORT. It signals that
    the current user path is no longer valid, and no further resolution or execution
    should take place.

    This is typically used when the user decides to cancel or redirect the intent.
    """

    def resolve(self,
                runtime_context: LLMRuntimeContext,
                kind: set[ResolutionKind],
                context: ResolutionContext,
                interaction: Optional[Interaction] = None) -> ResolutionOutcome:

        return ResolutionOutcome(
            result=ResolutionResult.ABORT,
            resolved=None,
            propagate_slots=[]
        )
