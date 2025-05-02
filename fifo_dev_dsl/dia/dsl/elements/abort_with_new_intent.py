from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

from common.llm.dia.dsl.elements.base import DslBase
from common.llm.dia.dsl.elements.intent import Intent
from common.llm.dia.resolution.enums import ResolutionKind, ResolutionResult
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext

@dataclass
class AbortWithNewIntent(DslBase):
    """
    Aborts the current resolution path and replaces it with a new intent.

    Like `Abort`, this node returns ResolutionResult.ABORT, signaling that the
    current intent is no longer valid. However, the `resolved` field contains
    a new `Intent` object to be used in its place.

    This is useful for graceful redirection, for example, when an item is
    unavailable and a fallback is suggested.

    Params:
        intent (Intent):
            New intent to install as a replacement for the aborted one.
    """

    intent: Intent

    def resolve(self,
                runtime_context: LLMRuntimeContext,
                kind: set[ResolutionKind],
                context: ResolutionContext,
                interaction: Optional[Interaction] = None) -> ResolutionOutcome:

        return ResolutionOutcome(
            result=ResolutionResult.ABORT,
            resolved=self.intent,
            propagate_slots=[]
        )
