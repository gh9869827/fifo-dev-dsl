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
    into the dsl tree, or abort an intent.

    Attributes:
        result (ResolutionResult):
            The outcome of the resolution:
              - UNCHANGED: No change was made to the current node.
              - CHANGED: The node was updated internally (in-place); no structural change required.
              - NEW_DSL_NODES: One or more nodes should be integrated into the DSL tree.
              - INTERACTION_REQUESTED: User input is required to proceed.
              - ABORT: The intent should be aborted or replaced entirely.

        node (DslBase | list[DslBase] | None):
            The replacement node(s), if applicable.
            - Required when `result` is ABORT with replacement or NEW_DSL_NODES.
            - Should be None when `result` is UNCHANGED, CHANGED, or INTERACTION_REQUESTED.
            - May be a list of nodes when multiple replacements are produced.

        interaction (InteractionRequest | None):
            A request for user input.
            Must be provided only if `result` is INTERACTION_REQUESTED.

    Example Usage:
        - Node was updated internally:
            ResolutionOutcome(result=CHANGED)

        - Expand placeholder into new DSL nodes:
            ResolutionOutcome(result=NEW_DSL_NODES, node=[new_node1, new_node2])

        - Prompt the user for clarification:
            ResolutionOutcome(result=INTERACTION_REQUESTED, interaction=query)

        - Abort and replace with a new intent:
            ResolutionOutcome(result=ABORT, node=new_intent)

        - Abort without replacement:
            ResolutionOutcome(result=ABORT)
    """

    result: ResolutionResult = ResolutionResult.UNCHANGED
    node: DslBase | list[DslBase] | None = None
    interaction: InteractionRequest | None = None

    def __init__(self,
                 result: ResolutionResult = ResolutionResult.UNCHANGED,
                 node: DslBase | list[DslBase] | None = None,
                 interaction: InteractionRequest | None = None):
        """
        Initialize a ResolutionOutcome.

        Args:
            result (ResolutionResult):
                The outcome of the resolution.

            node (DslBase | list[DslBase] | None):
                Replacement node(s), only valid when result is NEW_DSL_NODES or ABORT with
                replacement.

            interaction (InteractionRequest | None):
                Interaction object for user input.
                Must be provided only when result is INTERACTION_REQUESTED.
        """
        self.result = result
        self.node = node
        self.interaction = interaction

        if result is ResolutionResult.INTERACTION_REQUESTED:
            if interaction is None:
                raise ValueError("Missing interaction object for INTERACTION_REQUESTED")
            if node is not None:
                raise ValueError("Node must be None for INTERACTION_REQUESTED")

        if result in (ResolutionResult.CHANGED, ResolutionResult.UNCHANGED):
            if node is not None:
                raise ValueError(f"Node must not be set when result is {result.name}")

        if result is not ResolutionResult.INTERACTION_REQUESTED and interaction is not None:
            raise ValueError("Interaction is only allowed for INTERACTION_REQUESTED")

        if result is ResolutionResult.NEW_DSL_NODES and node is None:
            raise ValueError("NEW_DSL_NODES requires at least one replacement node")
