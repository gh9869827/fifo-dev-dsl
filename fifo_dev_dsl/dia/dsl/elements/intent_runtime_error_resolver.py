from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.base import DslBase
import fifo_dev_dsl.dia.dsl.elements.helper as helper

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.resolution.enums import AbortBehavior
    from fifo_dev_dsl.dia.resolution.interaction import Interaction
    from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome
    from fifo_dev_dsl.dia.dsl.elements.intent import Intent
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class IntentRuntimeErrorResolver(DslBase):
    """
    Placeholder for a failed intent awaiting user-guided recovery.

    This node is injected into the DSL tree when an :class:`Intent` raises a runtime error
    during evaluation. It captures the failing intent and the associated error message,
    pausing automatic resolution.

    During the next resolution pass, this node prompts the user for a corrective action —
    such as retrying, modifying parameters, or abandoning the operation — and allows
    the system to generate new DSL elements in response.

    This mechanism enables interactive recovery from unexpected execution failures
    while preserving context.

    Attributes:
        intent (Intent):
            The intent that encountered a runtime error.

        error_message (str):
            A human-readable description of the failure.
    """

    intent: Intent
    error_message: str

    def pretty_print_dsl(self, indent: int = 0) -> None:
        prefix = "  " * indent
        print(f"{prefix}{self.__class__.__name__}(error_message={repr(self.error_message)})")
        self.intent.pretty_print_dsl(indent + 1)

    def get_children(self) -> list[DslBase]:
        return [self.intent]

    def do_resolution(self,
                       runtime_context: LLMRuntimeContext,
                       resolution_context: ResolutionContext,
                       abort_behavior: AbortBehavior,
                       interaction: Interaction | None) -> ResolutionOutcome:
        super().do_resolution(runtime_context, resolution_context, abort_behavior, interaction)

        return helper.ask_helper_error_resolver(
            runtime_context=runtime_context,
            current=(self, self.error_message),
            resolution_context=resolution_context,
            intent=self.intent,
            interaction=interaction)

    def is_resolved(self) -> bool:
        """
        Indicate that this node always stays unresolved.

        The resolver reflects a runtime error in the intent flow and cannot be
        reduced to a normal value.

        Returns:
            bool:
                Always ``False``.
        """
        return False
