from dataclasses import dataclass
from enum import Enum
from typing import Any


class EvaluationStatus(Enum):
    """
    Possible results of evaluating a DSL node tree.

    Attributes:
        SUCCESS:
            Evaluation finished without errors.

        ABORTED_RECOVERABLE:
            Evaluation was aborted due to a recoverable issue, such as missing resources
            or input that cannot be fulfilled exactly as requested. For example, if the user
            asks to retrieve 4 screws of 12 mm but only 2 are available at runtime, evaluation
            is aborted and this status is returned. The user can then rephrase or adjust
            the request based on the feedback.

        ABORTED_UNRECOVERABLE:
            Evaluation was aborted due to a fatal or internal error that cannot be
            resolved
    """

    SUCCESS = "success"
    ABORTED_RECOVERABLE = "aborted_recoverable"
    ABORTED_UNRECOVERABLE = "aborted_unrecoverable"


@dataclass
class EvaluationOutcome:
    """
    Contains the result of an evaluation pass over a DSL node tree.

    Attributes:
        status (EvaluationStatus):
            The overall outcome of the evaluation.

        value (Any):
            The result value if evaluation succeeded. May be None if evaluation
            was aborted or if the intent does not produce a return value.

        error (Exception | None):
            The exception that caused the evaluation to abort, if any.
    """

    value: Any = None
    status: EvaluationStatus = EvaluationStatus.SUCCESS
    error: Exception | None = None
