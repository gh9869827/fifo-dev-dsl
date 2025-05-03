from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

from common.llm.dia.dsl.elements.base import DslBase

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
