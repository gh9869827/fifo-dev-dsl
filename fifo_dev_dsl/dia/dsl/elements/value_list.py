from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional

from dataclasses import dataclass

from common.introspection.docstring import MiniDocStringType
from common.llm.dia.dsl.elements.value_base import DSLValueBase
from common.llm.dia.resolution.enums import ResolutionKind, ResolutionResult
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.outcome import (
    ResolutionOutcome,
    ResolutionOutcomeResultAndPropagationSlot
)
from common.typeutils.cast import strict_cast

if TYPE_CHECKING:
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext

@dataclass
class ListValue(DSLValueBase):

    values: List[DSLValueBase]

    def resolve(self,
                runtime_context: LLMRuntimeContext,
                kind: set[ResolutionKind],
                context: ResolutionContext,
                interaction: Optional[Interaction] = None) -> ResolutionOutcome:

        new_items: List[DSLValueBase] = []
        outcome_rnp = ResolutionOutcomeResultAndPropagationSlot()

        skip = False
        for val in self.values:
            if skip:
                new_items.append(val)
                continue

            outcome = val.resolve(runtime_context, kind, context.deepcopy(), interaction)

            outcome_rnp += outcome

            if outcome.result is ResolutionResult.ABORT:
                return outcome

            if outcome.result is ResolutionResult.INTERACTION_REQUESTED:
                skip = True

            new_items.append(strict_cast(DSLValueBase, outcome.resolved))

        return ResolutionOutcome(
            result=outcome_rnp.result,
            resolved=ListValue(new_items),
            propagate_slots=outcome_rnp.propagate_slots
            # todo where is interraction here ?
        )

    def is_resolved(self) -> bool:
        return all(val.is_resolved() for val in self.values)

    def get_resolved_value_as_text(self) -> str:
        lst = ",".join([v.get_resolved_value_as_text() for v in self.values])
        return f"[{lst}]"

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: Optional[MiniDocStringType] = None) -> Any:

        if value_type is None:
            raise RuntimeError("Missing expected type for evaluation of ListValue")

        if (inner_type := value_type.is_list()) is not None:
            return [e.eval(runtime_context, inner_type) for e in self.values]

        raise ValueError(f"Invalid type for ListValue eval(): {value_type}")
