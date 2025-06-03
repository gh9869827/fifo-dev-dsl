from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
from fifo_dev_dsl.dia.dsl.elements.element_list import ListElement
from fifo_dev_dsl.dia.dsl.elements.intent_evaluated_success import IntentEvaluatedSuccess
from fifo_dev_dsl.dia.resolution.context import ResolutionContextStackElement
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.elements.intent import Intent
from fifo_dev_dsl.dia.dsl.elements.intent_runtime_error_resolver import IntentRuntimeErrorResolver
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationOutcome, EvaluationStatus
from fifo_dev_dsl.dia.runtime.exceptions import ApiErrorAbortAndResolve


class Evaluator:

    def __init__(self, runtime_context: LLMRuntimeContext, root: DslBase):
        self._runtime_context = runtime_context
        self._call_stack: list[ResolutionContextStackElement] = [
            ResolutionContextStackElement(root, 0)
        ]

    def evaluate(self, expected_type: MiniDocStringType | None = None) -> EvaluationOutcome:
        while self._call_stack:
            current = self._call_stack[-1]

            if isinstance(current.obj, Intent):
                try:
                    value = current.obj.eval(self._runtime_context, expected_type)
                    wrapped = IntentEvaluatedSuccess(
                        intent=current.obj,
                        evaluation_outcome=EvaluationOutcome(value)
                    )
                    if len(self._call_stack) >= 2:
                        parent = self._call_stack[-2]
                        parent.obj.update_child(parent.idx, wrapped)
                    self._call_stack.pop()
                    if self._call_stack:
                        self._call_stack[-1].idx += 1
                except Exception as ex:
                    outcome = self._handle_eval_error(current.obj, ex)
                    if outcome:
                        if len(self._call_stack) >= 2:
                            parent = self._call_stack[-2]
                            parent.obj.update_child(parent.idx, outcome)
                        return EvaluationOutcome(
                            value=None,
                            status=EvaluationStatus.ABORTED_RECOVERABLE,
                            error=ex
                        )
                    else:
                        return EvaluationOutcome(
                            value=None,
                            status=EvaluationStatus.ABORTED_UNRECOVERABLE,
                            error=ex
                        )

            elif isinstance(current.obj, ListElement):
                children = current.obj.get_children()

                if current.idx >= len(children):
                    self._call_stack.pop()
                    if self._call_stack:
                        self._call_stack[-1].idx += 1
                    else:
                        return EvaluationOutcome(value=None, status=EvaluationStatus.SUCCESS)
                else:
                    child = children[current.idx]
                    self._call_stack.append(ResolutionContextStackElement(child, 0))

            elif isinstance(current.obj, IntentEvaluatedSuccess):
                # Already evaluated, just skip
                self._call_stack.pop()
                if self._call_stack:
                    self._call_stack[-1].idx += 1

            else:
                return EvaluationOutcome(
                    status=EvaluationStatus.ABORTED_UNRECOVERABLE, 
                    error=TypeError(f"Unexpected node type in eval: {type(current.obj).__name__}")
                )

        return EvaluationOutcome(
            status=EvaluationStatus.ABORTED_UNRECOVERABLE,
            error=RuntimeError("Evaluation terminated unexpectedly.")
        )

    def _handle_eval_error(self, node: DslBase, error: Exception) -> DslBase | None:
        print(f"[eval error] {node}: {error}")

        if isinstance(error, ApiErrorAbortAndResolve) and isinstance(node, Intent):
            return IntentRuntimeErrorResolver(intent=node, error_message=str(error))

        return None
