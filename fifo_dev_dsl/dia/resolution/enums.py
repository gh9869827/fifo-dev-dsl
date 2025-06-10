from __future__ import annotations
from enum import IntEnum

class ResolutionResult(IntEnum):
    """
    Represents the outcome of attempting to resolve a DSL element during a resolution phase.

    Values (in order of precedence, highest first):
        ABORT = 4
            Critical failure or intent-level cancellation. Halts the current intent's resolution,
            optionally replacing it with a new one.

        INTERACTION_REQUESTED = 3
            The system requires user input or clarification before proceeding.

        NEW_DSL_NODES = 2
            One or more new DSL nodes were produced and should be integrated into the DSL tree.
            This typically includes cases where resolution expands a placeholder into multiple
            concrete nodes (e.g., a value and associated propagation metadata).

        CHANGED = 1
            The current node was updated in-place (e.g., internal state change).
            No structural changes to the tree were made.

        UNCHANGED = 0
            No changes occurred. Traversal continues normally.

    Methods:
        combine(other: ResolutionResult) -> ResolutionResult:
            Combine two resolution results by selecting the one with higher precedence.
    """

    ABORT = 4
    INTERACTION_REQUESTED = 3
    NEW_DSL_NODES = 2
    CHANGED = 1
    UNCHANGED = 0
