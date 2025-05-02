from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from dataclasses import dataclass, field

from common.llm.dia.resolution.enums import ResolutionResult

if TYPE_CHECKING:
    from common.llm.dia.dsl.elements.base import DslBase
    from common.llm.dia.dsl.elements.propagate_slot import PropagateSlot

@dataclass
class InteractionRequest:
    """
    Represents a prompt for user interaction during resolution.
    """
    message: str
    expected_type: str  # e.g. 'str', 'int', 'choice'
    requester: DslBase
    slot_name: Optional[str] = None  # Which slot this is clarifying

@dataclass
class InteractionAnswer:
    content: str
    consumed: bool = False


@dataclass
class Interaction:
    request: InteractionRequest
    answer: InteractionAnswer