from __future__ import annotations
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass
from fifo_dev_dsl.dia.dsl.elements.base import DslBase

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
    from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationOutcome
    from fifo_dev_dsl.dia.dsl.elements.intent import Intent


@dataclass
class IntentEvaluatedSuccess(DslBase):
    """
    Marks that an intent has successfully completed evaluation.

    This node wraps a previously executed intent and stores its evaluation outcome,
    including return values and status. It prevents redundant re-execution by
    preserving results from earlier stages in the DSL evaluation process.

    This mechanism is essential for non-idempotent operations or side-effecting
    actions — for example, if a robot has already moved or an item was deleted,
    re-evaluating that intent could cause incorrect behavior. By wrapping the intent
    in a success marker, we ensure correct recovery and safe replay during error
    handling or full-tree re-evaluation.

    Attributes:
        intent (Intent):
            The intent that completed without errors.

        evaluation_outcome (EvaluationOutcome):
            The result of evaluating the intent, including its return value and status.
    """

    intent: Intent
    evaluation_outcome: EvaluationOutcome

    def pretty_print_dsl(self, indent: int = 0) -> None:
        prefix = "  " * indent
        print(f"{prefix}{self.__class__.__name__}(status={self.evaluation_outcome.status.name})")
        self.intent.pretty_print_dsl(indent + 1)

    def get_children(self) -> list[DslBase]:
        return [self.intent]

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
        value_type: MiniDocStringType | None = None,
    ) -> Any:
        """Return the stored evaluation value.

        This node records the result of a previous evaluation pass. Evaluation
        simply returns the preserved value without re-invoking the wrapped
        intent. If an unresolved placeholder somehow remains, a
        :class:`RuntimeError` is raised.

        Args:
            runtime_context: Execution context (unused).
            value_type: Optional type hint used to cast the stored result.

        Returns:
            Any: The previously computed value from ``evaluation_outcome``.
        """

        if not self.is_resolved():
            raise RuntimeError(
                f"Unresolved DSL node: {self.__class__.__name__}"
            )

        result = self.evaluation_outcome.value
        return (
            value_type.cast(result) if value_type is not None else result
        )
