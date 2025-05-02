from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

from common.introspection.docstring import MiniDocStringType
from common.llm.dia.resolution.enums import ResolutionKind, ResolutionResult
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext

class DslBase:
    """
    Base class for all DSL elements.

    Each DSL element represents a node in the intent structure and can optionally
    participate in resolution during a given resolution wave (e.g., user interaction,
    runtime queries), or in evaluation (producing a final value).

    Subclasses override `resolve()` to participate in specific resolution phases,
    and `eval()` if they produce a concrete value during execution.
    """

    def resolve(self,
                runtime_context: LLMRuntimeContext,
                kind: set[ResolutionKind],
                context: ResolutionContext,
                interaction: Optional[Interaction] = None) -> ResolutionOutcome:
        """
        Attempt to resolve this element for the given resolution wave/kind.

        This method is called during resolution waves (e.g., ASK, QUERY_FILL) to give
        the element a chance to participate. Subclasses override this method to provide
        custom resolution behavior.
        By default, the base class does not respond to any resolution phase
        and simply returns itself as-is with a NOT_APPLICABLE result.

        Args:
            runtime_context (LLMRuntimeContext):
                Provides access to tools, query sources, and precompiled LLM prompts used
                to guide model responses and fill in missing information.

            kind (Set[ResolutionKind]):
                The type of resolution pass currently being executed (ASK, QUERY_FILL, etc).

            context (ResolutionContext):
                Tracks the current resolution state, including the intent being resolved,
                the slot in question, and any values that need to be propagated.

            interaction: todo if not None we resume evaluation where we left off with the new answer provided by the user

        Returns:
            ResolutionOutcome:
                Indicates whether the element was modified, aborted, or unchanged.
        """
        _ = runtime_context, kind, context, interaction

        return ResolutionOutcome(
            result=ResolutionResult.NOT_APPLICABLE,
            resolved=self,
            propagate_slots=[]
        )

    def is_resolved(self) -> bool:
        """
        Indicates whether this DSL element is fully resolved.

        A resolved element has no remaining ASK, QUERY_FILL, or QUERY_USER nodes
        and is ready for evaluation or execution. Subclasses override this when
        their resolution state depends on internal fields.

        Returns:
            bool:
                True if the element is resolved, False otherwise.
        """
        return True

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: Optional[MiniDocStringType] = None) -> Any:
        """
        Evaluate the DSL element and return its final value.

        Only fully resolved elements should be evaluated. Subclasses override this
        to return a concrete value (e.g., a string, number, or list). If called on
        an unresolved element or one that does not support evaluation, a RuntimeError
        is raised.

        Args:
            runtime_context (LLMRuntimeContext):
                Runtime context providing access to tools, sources, and LLM prompts.

            value_type (Optional[MiniDocStringType]):
                Optional hint about the expected type of the result.

        Returns:
            Any:
                The concrete value resulting from evaluation.

        Raises:
            RuntimeError:
                If the element cannot be evaluated.
        """
        raise RuntimeError("Expression cannot be evaluated, be sure it is resolvable/resolved.")
