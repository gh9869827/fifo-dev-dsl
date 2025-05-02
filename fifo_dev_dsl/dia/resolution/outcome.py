from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from dataclasses import dataclass, field

from common.llm.dia.resolution.enums import ResolutionResult
from common.llm.dia.resolution.interaction import InteractionRequest

if TYPE_CHECKING:
    from common.llm.dia.dsl.elements.base import DslBase
    from common.llm.dia.dsl.elements.propagate_slot import PropagateSlot


@dataclass
class ResolutionOutcomeResultAndPropagationSlot:
    """
    Captures the result of resolving a DSL element, without the resolved value itself.

    This includes:
      - The resolution result (success, not applicable, or abort)
      - Any slot values that should be propagated to the current intent

    Slot propagation is used when the user provides multiple values at once or gives
    a value that wasn't specifically requested. These values need to be injected
    back into the parent intent being constructed.

    Attributes:
        result (ResolutionResult):
            The result of the resolution step.

        propagate_slots (List[PropagateSlot]):
            Slot values to apply to the intent being resolved.

    Methods:
        __iadd__(other):
            Combine this outcome with another one by:
              - Merging the resolution result using combine()
              - Appending all propagated slots
    """

    result: ResolutionResult = ResolutionResult.NOT_APPLICABLE
    propagate_slots: List[PropagateSlot] = field(default_factory=list)

    def __iadd__(self,
                 other: ResolutionOutcomeResultAndPropagationSlot
                 ) -> ResolutionOutcomeResultAndPropagationSlot:
        """
        Merge another outcome into this one.

        Combines the resolution result using the combine() method,
        and appends all propagated slots from the other outcome.

        Args:
            other (ResolutionOutcomeResultAndPropagationSlot):
                The outcome to merge into this one.

        Returns:
            ResolutionOutcomeResultAndPropagationSlot:
                The updated outcome after merging.
        """
        # Combine resolution result using the Enum's combine method
        self.result = self.result.combine(other.result)

        # Extend the propagate slots
        self.propagate_slots.extend(other.propagate_slots)

        return self

@dataclass
class ResolutionOutcome(ResolutionOutcomeResultAndPropagationSlot):
    """
    Full result of resolving a DSL element, including the resolved version of the element.

    This extends ResolutionOutcomeResultAndPropagationSlot by adding the resolved element
    itself. If resolution was successful, this holds the updated version of the DSL element.
    If the result is ABORT, the resolved value may be None or replaced (in the case of
    AbortWithNewIntent).

    Attributes:
        resolved (DslBase | list[DslBase] | None):
            The DSL element(s) after resolution. May be None for ABORT or unchanged otherwise.

    Constructor Args:
        base (Optional[ResolutionOutcomeResultAndPropagationSlot]):
            Base outcome to inherit result and propagation from.

        result (Optional[ResolutionResult]):
            Resolution result if not using a base.

        propagate_slots (Optional[List[PropagateSlot]]):
            Slot values to propagate if not using a base.

        resolved (Optional[DslBase]):
            Final resolved DSL element to carry forward.
    """
    resolved: DslBase | list[DslBase] | None = None

    interaction: InteractionRequest | None = None # todo doc... interaction is mandatory when ResolutionResult is INTERACTION_REQUESTED

    def __init__(self,
                 base: Optional[ResolutionOutcomeResultAndPropagationSlot] = None,
                 result: Optional[ResolutionResult] = None,
                 propagate_slots: Optional[list[PropagateSlot]] = None,
                 resolved: Optional[DslBase] = None,
                 interaction: Optional[InteractionRequest] = None):
        """
        Initialize a ResolutionOutcome.

        You can either pass a `base` object to copy from, or provide both `result` and
        `propagate_slots` explicitly.

        Args:
            base (Optional[ResolutionOutcomeResultAndPropagationSlot]):
                If provided, the result and propagate_slots will be copied from this object.

            result (Optional[ResolutionResult]):
                The result of the resolution. Required if base is not provided.

            propagate_slots (Optional[list[PropagateSlot]]):
                Slots to propagate. Required if base is not provided.

            resolved (Optional[DslBase]):
                The resolved DSL element. May be None (for example in the case of ABORT).

            todo: doc
        """
        if base is not None:
            result = base.result
            propagate_slots = list(base.propagate_slots)
        elif result is None or propagate_slots is None:
            if interaction is None:
                raise ValueError("Must provide either `base` or both `result` and `propagate_slots` or `interaction`") # todo be sure the implementation matches the logic

        super().__init__(result=result, propagate_slots=propagate_slots)
        self.resolved = resolved
        self.interaction = interaction
