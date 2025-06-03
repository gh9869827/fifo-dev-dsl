from dataclasses import dataclass
from enum import Enum
from typing import Any


class EvaluationStatus(Enum):
    SUCCESS = "success"
    ABORTED_RECOVERABLE = "aborted_recoverable"
    ABORTED_UNRECOVERABLE = "aborted_unrecoverable"


@dataclass
class EvaluationOutcome:
    value: Any = None
    status: EvaluationStatus = EvaluationStatus.SUCCESS
    error: Exception | None = None
