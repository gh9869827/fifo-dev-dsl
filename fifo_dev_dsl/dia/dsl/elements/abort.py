from __future__ import annotations
from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.base import DslBase


@dataclass
class Abort(DslBase):
    """
    A control directive that halts intent resolution and prevents execution.

    When encountered, this node returns a ResolutionResult.ABORT. It signals that
    the current user path is no longer valid, and no further resolution or execution
    should take place.

    This is typically used when the user decides to cancel or redirect the intent.
    """
