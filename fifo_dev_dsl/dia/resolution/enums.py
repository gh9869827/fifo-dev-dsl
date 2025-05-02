from __future__ import annotations
from enum import Enum, IntEnum


class ResolutionKind(Enum):
    """
    Defines the different resolution strategies applied to DSL elements.

    These kinds represent distinct phases in the intent grounding process,
    applied in sequence to resolve or respond to incomplete or exploratory input.

    Values:
        ASK:
            Prompt the user for a missing slot value using ASK(...).
            Example: ASK("What quantity do you need?")
        
        QUERY_FILL:
            Attempt to automatically fill missing values from the runtime context.
            Example: QUERY_FILL("longest screw available")
        
        QUERY_USER:
            Represents a user-initiated query. This may occur:
              - During slot resolution (e.g., replying with a question instead of a value)
              - As a top-level input (e.g., "Do you have any metal screws?")
            
            The agent responds factually and may resume resolution or prompt
            the user for follow-up intent afterward.

    These resolution phases are typically applied in waves, allowing agents to
    progressively clarify, complete, or respond to user input with context-awareness.
    """
    ASK = "ask"
    QUERY_FILL = "query_fill"
    QUERY_USER = "query_user"

class ResolutionResult(IntEnum):
    """
    Represents the outcome of attempting to resolve a DSL element during a resolution phase.

    Values (precedence order):
        ABORT = 3
            Critical failure. Halts the entire resolution process.

        INTERACTION_REQUESTED = 2
            User interaction is needed before continuing.

        APPLICABLE_SUCCESS = 1
            The resolution applied successfully and modified the element.

        NOT_APPLICABLE = 0
            This kind of resolution doesn't apply to the current element.

    Methods:
        combine(other: ResolutionResult) -> ResolutionResult:
            Combine two resolution results by keeping the one with the highest precedence.
    """

    ABORT = 3
    INTERACTION_REQUESTED = 2
    APPLICABLE_SUCCESS = 1
    NOT_APPLICABLE = 0

    def combine(self, other: ResolutionResult) -> ResolutionResult:
        """
        Combine this result with another, returning the one with the highest precedence.

        Args:
            other (ResolutionResult):
                Another resolution result.

        Returns:
            ResolutionResult:
                The more severe or applicable of the two.
        """
        return max(self, other)
