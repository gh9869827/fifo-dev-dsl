from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.interaction import InteractionRequest

if TYPE_CHECKING:
    from fifo_dev_dsl.dia.dsl.elements.base import DslBase

@dataclass
class ResolutionOutcome:
    """
    Captures the result of resolving a DSL element during tree traversal.

    This object summarizes what happened during a resolution step and informs the resolution
    loop what to do next: continue traversal, prompt for user input, integrate newly created nodes
    into the DSL tree, or abort an intent.

    Attributes:
        result (ResolutionResult):
            The outcome of the resolution:
              - UNCHANGED: No change was made to the current node.
              - CHANGED: The node was updated internally (in-place); no structural change required.
              - NEW_DSL_NODES: One or more nodes should be integrated into the DSL tree.
              - INTERACTION_REQUESTED: User input is required to proceed.
              - ABORT: The intent should be aborted or replaced entirely.

        node (DslBase | None):
            A single replacement node, if applicable.

        nodes (list[DslBase] | None):
            A list of replacement nodes, if multiple are produced.

        interaction (InteractionRequest | None):
            A request for user input.
            Must be provided only if `result` is INTERACTION_REQUESTED.
    """

    result: ResolutionResult
    node: DslBase | None = None
    nodes: list[DslBase] | None = None
    interaction: InteractionRequest | None = None

    def __init__(
        self,
        result: ResolutionResult = ResolutionResult.UNCHANGED,
        node: DslBase | None = None,
        nodes: list[DslBase] | None = None,
        interaction: InteractionRequest | None = None
    ):
        self.result = result
        self.node = node
        self.nodes = nodes
        self.interaction = interaction

        # Validation logic
        if result == ResolutionResult.INTERACTION_REQUESTED:
            if interaction is None:
                raise ValueError("Missing interaction object for INTERACTION_REQUESTED")
            if node is not None or nodes is not None:
                raise ValueError("node and nodes must be None for INTERACTION_REQUESTED")

        elif result in (ResolutionResult.UNCHANGED, ResolutionResult.CHANGED):
            if node is not None or nodes is not None:
                raise ValueError(f"node and nodes must not be set when result is {result.name}")
            if interaction is not None:
                raise ValueError("Interaction is only allowed for INTERACTION_REQUESTED")

        elif result == ResolutionResult.NEW_DSL_NODES:
            if not nodes:
                raise ValueError("NEW_DSL_NODES requires a non-empty list of replacement nodes")
            if node is not None:
                raise ValueError("node must be None when using nodes")

        elif result == ResolutionResult.ABORT:
            if nodes is not None and not nodes:
                raise ValueError("ABORT with empty list is invalid; use None or a non-empty list")
            if node is not None and nodes is not None:
                raise ValueError("Cannot provide both node and nodes for ABORT")

        else:
            raise ValueError(f"Unexpected ResolutionResult: {result}")
