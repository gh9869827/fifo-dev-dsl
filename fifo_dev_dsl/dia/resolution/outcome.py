from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

from fifo_dev_dsl.dia.resolution.enums import ResolutionResult

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.resolution.interaction import InteractionRequest
    from fifo_dev_dsl.dia.dsl.elements.base import DslBase

@dataclass
class ResolutionOutcome:
    """
    Captures the outcome of resolving a DSL element during tree traversal.

    This object summarizes the result of a resolution step and guides the next action:
    continue traversal, prompt for user input, insert new nodes into the DSL tree, or
    abort and replace an intent.

    Attributes:
        result (ResolutionResult):
            The outcome of resolving the current DSL node:
              - UNCHANGED:
                  No change was made. Traversal proceeds normally.
              - NEW_DSL_NODES:
                  One or more new DSL nodes should replace the current node.
              - INTERACTION_REQUESTED:
                  Evaluation paused pending user input or clarification.
              - ABORT:
                  Resolution should terminate or the current intent should be replaced.

        node (DslBase | None):
            A single replacement node, used if only one new node is produced.

        nodes (list[DslBase] | None):
            A list of replacement nodes, used when resolution yields multiple elements.

        interaction (InteractionRequest | None):
            A request for user input.
            Required only when `result` is INTERACTION_REQUESTED.
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
        if result is ResolutionResult.INTERACTION_REQUESTED:
            if interaction is None:
                raise ValueError("Missing interaction object for INTERACTION_REQUESTED")
            if node is not None or nodes is not None:
                raise ValueError("node and nodes must be None for INTERACTION_REQUESTED")

        elif result is ResolutionResult.UNCHANGED:
            if node is not None or nodes is not None:
                raise ValueError(f"node and nodes must not be set when result is {result.name}")
            if interaction is not None:
                raise ValueError("Interaction is only allowed for INTERACTION_REQUESTED")

        elif result is ResolutionResult.NEW_DSL_NODES:
            if not nodes:
                raise ValueError("NEW_DSL_NODES requires a non-empty list of replacement nodes")
            if node is not None:
                raise ValueError("node must be None when using nodes")

        elif result is ResolutionResult.ABORT:
            if nodes is not None and not nodes:
                raise ValueError("ABORT with empty list is invalid; use None or a non-empty list")
            if node is not None and nodes is not None:
                raise ValueError("Cannot provide both node and nodes for ABORT")

        else:
            raise ValueError(f"Unexpected ResolutionResult: {result}")
