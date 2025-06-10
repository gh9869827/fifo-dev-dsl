from __future__ import annotations
from enum import IntEnum

class ResolutionResult(IntEnum):
    """
    Represents the result of resolving a DSL element during a resolution pass.

    Resolution outcomes are prioritized in the following order (highest first):

        ABORT = 3
            Critical failure or intent-level override. Halts resolution of the current
            intent, optionally replacing it with a new one.

        INTERACTION_REQUESTED = 2
            Evaluation requires user input or clarification. Resolution is paused
            until interaction is handled externally.

        NEW_DSL_NODES = 1
            Resolution expanded a placeholder into one or more new DSL nodes. These
            should replace the current node in the DSL tree before traversal continues.

        UNCHANGED = 0
            No changes were made during resolution. Traversal proceeds to the next node.
    """

    ABORT = 3
    INTERACTION_REQUESTED = 2
    NEW_DSL_NODES = 1
    UNCHANGED = 0
