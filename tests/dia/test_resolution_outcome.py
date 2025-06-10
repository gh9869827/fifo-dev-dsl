import pytest
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.interaction import InteractionRequest
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome
from fifo_dev_dsl.dia.dsl.elements.intent import Intent
from fifo_dev_dsl.dia.dsl.elements.slot import Slot

# Dummy DSL node for testing
DUMMY_NODE = Intent("foo", [])

def make_dummy_interaction(message: str="Choose:",
                           expected_type: str="choice",
                           requester: DslBase=DUMMY_NODE,
                           slot: Slot | None=None):
    return InteractionRequest(
        message=message,
        expected_type=expected_type,
        requester=requester,
        slot=slot
    )

def test_valid_interaction_requested():
    interaction = make_dummy_interaction(message="What's your name?", expected_type="str")
    outcome = ResolutionOutcome(
        result=ResolutionResult.INTERACTION_REQUESTED,
        interaction=interaction
    )
    assert outcome.result is ResolutionResult.INTERACTION_REQUESTED
    assert outcome.interaction is not None
    assert outcome.interaction.message == "What's your name?"
    assert outcome.interaction.expected_type == "str"
    assert outcome.interaction.requester is DUMMY_NODE
    assert outcome.interaction.slot is None

def test_invalid_interaction_requested_missing_interaction():
    with pytest.raises(ValueError, match="Missing interaction object"):
        ResolutionOutcome(result=ResolutionResult.INTERACTION_REQUESTED)

def test_invalid_interaction_requested_with_node():
    with pytest.raises(ValueError, match="node and nodes must be None"):
        ResolutionOutcome(
            result=ResolutionResult.INTERACTION_REQUESTED,
            interaction=make_dummy_interaction(),
            node=DUMMY_NODE
        )

def test_invalid_interaction_requested_with_nodes():
    with pytest.raises(ValueError, match="node and nodes must be None"):
        ResolutionOutcome(
            result=ResolutionResult.INTERACTION_REQUESTED,
            interaction=make_dummy_interaction(),
            nodes=[DUMMY_NODE]
        )

def test_valid_unchanged():
    outcome = ResolutionOutcome(result=ResolutionResult.UNCHANGED)
    assert outcome.result is ResolutionResult.UNCHANGED

def test_invalid_unchanged_with_node():
    with pytest.raises(ValueError, match="must not be set when result is UNCHANGED"):
        ResolutionOutcome(result=ResolutionResult.UNCHANGED, node=DUMMY_NODE)

def test_invalid_unchanged_with_nodes():
    with pytest.raises(ValueError, match="must not be set when result is UNCHANGED"):
        ResolutionOutcome(result=ResolutionResult.UNCHANGED, nodes=[DUMMY_NODE])

def test_invalid_unchanged_with_interaction():
    with pytest.raises(ValueError, match="Interaction is only allowed"):
        ResolutionOutcome(result=ResolutionResult.UNCHANGED, interaction=make_dummy_interaction())

def test_valid_new_dsl_nodes():
    outcome = ResolutionOutcome(result=ResolutionResult.NEW_DSL_NODES, nodes=[DUMMY_NODE])
    assert outcome.nodes == [DUMMY_NODE]

def test_invalid_new_dsl_nodes_empty():
    with pytest.raises(ValueError, match="requires a non-empty list"):
        ResolutionOutcome(result=ResolutionResult.NEW_DSL_NODES, nodes=[])

def test_invalid_new_dsl_nodes_with_node():
    with pytest.raises(ValueError, match="node must be None"):
        ResolutionOutcome(result=ResolutionResult.NEW_DSL_NODES, node=DUMMY_NODE, nodes=[DUMMY_NODE])

def test_valid_abort_with_node():
    outcome = ResolutionOutcome(result=ResolutionResult.ABORT, node=DUMMY_NODE)
    assert outcome.node == DUMMY_NODE

def test_valid_abort_with_nodes():
    outcome = ResolutionOutcome(result=ResolutionResult.ABORT, nodes=[DUMMY_NODE])
    assert outcome.nodes == [DUMMY_NODE]

def test_invalid_abort_with_empty_nodes():
    with pytest.raises(ValueError, match="ABORT with empty list is invalid"):
        ResolutionOutcome(result=ResolutionResult.ABORT, nodes=[])

def test_invalid_abort_with_both_node_and_nodes():
    with pytest.raises(ValueError, match="Cannot provide both node and nodes"):
        ResolutionOutcome(result=ResolutionResult.ABORT, node=DUMMY_NODE, nodes=[DUMMY_NODE])

def test_invalid_unexpected_result():
    class FakeResult: pass
    with pytest.raises(ValueError, match="Unexpected ResolutionResult"):
        ResolutionOutcome(result=FakeResult())  # type: ignore
