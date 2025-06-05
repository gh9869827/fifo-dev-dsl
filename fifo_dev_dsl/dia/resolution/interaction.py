from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.dsl.elements.slot import Slot
    from fifo_dev_dsl.dia.dsl.elements.base import DslBase

@dataclass
class InteractionRequest:
    """
    Represents a prompt for user interaction during resolution.
    """
    message: str
    expected_type: str  # e.g. 'str', 'int', 'choice'
    requester: DslBase
    slot: Slot | None = None  # Which slot this is clarifying


@dataclass
class InteractionAnswer:
    content: str
    consumed: bool = False


@dataclass
class Interaction:
    request: InteractionRequest
    answer: InteractionAnswer
