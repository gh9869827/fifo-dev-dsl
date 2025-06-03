from dataclasses import dataclass
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.elements.intent import Intent
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationOutcome


@dataclass
class IntentEvaluatedSuccess(DslBase):

    intent: Intent
    evaluation_outcome: EvaluationOutcome

    def pretty_print_dsl(self, indent: int = 0) -> None:
        prefix = "  " * indent
        print(f"{prefix}{self.__class__.__name__}(status={self.evaluation_outcome.status.name})")
        self.intent.pretty_print_dsl(indent + 1)

    def get_children(self) -> list[DslBase]:
        return [self.intent]
