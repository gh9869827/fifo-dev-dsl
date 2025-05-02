
from enum import Enum
from common.llm.dia.dsl.elements.base import DslBase
from common.llm.dia.dsl.elements.propagate_slot import PropagateSlot
from common.llm.dia.dsl.elements.value_base import DSLValueBase
from common.llm.dia.resolution.context import ResolutionContext
from common.llm.dia.resolution.enums import ResolutionKind, ResolutionResult
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.outcome import ResolutionOutcome, ResolutionOutcomeResultAndPropagationSlot
from common.llm.dia.runtime.context import LLMRuntimeContext
from common.typeutils.cast import strict_cast

class AbortBehavior(Enum):
    ABORT = 1
    SKIP = 1


def resolve4(abort_behavior: AbortBehavior, runtime_context: LLMRuntimeContext, dsl_elements: list[DslBase], kind: set[ResolutionKind], interaction: Interaction | None) -> ResolutionOutcome:
    while True:

        resolutions_result = ResolutionResult.NOT_APPLICABLE
        new_dsl_elements = []
        skip = False
        outcome = None

        for dsl_element in dsl_elements:
            if skip:
                new_dsl_elements.append(dsl_element)
                continue

            outcome = dsl_element.resolve(runtime_context, kind, ResolutionContext(), interaction)

            if outcome.result is ResolutionResult.ABORT:
                # we drop the current intent as it was aborted
                if abort_behavior is AbortBehavior.SKIP:
                    continue
                else:
                    return outcome

            resolutions_result = resolutions_result.combine(outcome.result)

            if outcome.result is ResolutionResult.INTERACTION_REQUESTED:
                skip = True
            
            if outcome.resolved is not None:
                new_dsl_elements.append(outcome.resolved)

        dsl_elements = new_dsl_elements

        if skip and outcome is not None:
            return outcome.interaction, dsl_elements
        
        if resolutions_result is ResolutionResult.NOT_APPLICABLE:
            return None, dsl_elements

def resolve(dsl_elements: list[DslBase],
            runtime_context: LLMRuntimeContext,
            resolution_context: ResolutionContext,
            resolution_kind: set[ResolutionKind],
            abort_behavior: AbortBehavior,
            interaction: Interaction | None) -> ResolutionOutcome:

    new_items: list[DSLValueBase] = []
    outcome_rnp = ResolutionOutcomeResultAndPropagationSlot()

    interaction_requested = False
    propagated_slots = []

    for val in dsl_elements:

        if interaction_requested:
            new_items.append(val)
            continue

        outcome = val.resolve(runtime_context, resolution_kind, resolution_context.deepcopy(), interaction)

        outcome_rnp += outcome

        if outcome.result is ResolutionResult.ABORT:
            if abort_behavior is AbortBehavior.ABORT:
                return outcome
            # else
            continue

        if outcome.result is ResolutionResult.NOT_APPLICABLE and isinstance(outcome.resolved, PropagateSlot):
            propagated_slots.append(outcome.resolved)
            continue

        if outcome.result is ResolutionResult.INTERACTION_REQUESTED:
            interaction_requested = True

        new_items.append(strict_cast(DslBase, outcome.resolved))

    if interaction_requested:
        return ResolutionOutcome(
            result=outcome_rnp.result,
            resolved=new_items,
            propagate_slots=outcome_rnp.propagate_slots + propagated_slots,
            interaction=outcome.interaction
        )
    elif outcome_rnp.result is ResolutionResult.NOT_APPLICABLE:
        assert len(outcome_rnp.propagate_slots) == 0
        return ResolutionOutcome(
            result=outcome_rnp.result,
            resolved=new_items,
            propagate_slots=outcome_rnp.propagate_slots + propagated_slots
        )
    else:
        tmp = resolve(new_items, runtime_context, resolution_context, resolution_kind, abort_behavior, interaction)
        tmp.propagate_slots += propagated_slots
        return tmp
