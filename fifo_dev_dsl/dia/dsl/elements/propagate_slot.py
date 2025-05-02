from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from dataclasses import dataclass
from typing import Dict

from common.llm.dia.dsl.elements.base import DslBase
from common.llm.dia.resolution.enums import ResolutionKind, ResolutionResult
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.outcome import ResolutionOutcome, ResolutionOutcomeResultAndPropagationSlot

if TYPE_CHECKING:
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext

@dataclass
class PropagateSlot(DslBase):

    slots: Dict[str, DslBase]

    def resolve(self,
                runtime_context: LLMRuntimeContext,
                kind: set[ResolutionKind],
                context: ResolutionContext,
                interaction: Optional[Interaction] = None) -> ResolutionOutcome:

        new_slots: Dict[str, DslBase] = {}

        outcome_rnp = ResolutionOutcomeResultAndPropagationSlot()

        skip = False
        for key, val in self.slots.items():
            if skip:
                new_slots[key] = val
                continue

            ctx = context.deepcopy()
            ctx.slot = key

            outcome = val.resolve(runtime_context, kind, ctx, interaction)

            outcome_rnp += outcome

            if outcome.result is ResolutionResult.ABORT:
                return outcome

            if outcome.result is ResolutionResult.INTERACTION_REQUESTED:
                skip = True

            if outcome.resolved is None:
                raise RuntimeError(f"Propagated solt: resolved value for key '{key}' is None")

            new_slots[key] = outcome.resolved

        return ResolutionOutcome(
            result=outcome_rnp.result,
            resolved=PropagateSlot(new_slots),
            propagate_slots=outcome.propagate_slots
        )

    def is_resolved(self) -> bool:
        return all(val.is_resolved() for val in self.slots.values())
