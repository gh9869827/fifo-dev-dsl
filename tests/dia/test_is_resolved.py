import pytest

from fifo_dev_dsl.dia.dsl.elements.abort import Abort
from fifo_dev_dsl.dia.dsl.elements.abort_with_new_dsl import AbortWithNewDsl
from fifo_dev_dsl.dia.dsl.elements.ask import Ask
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.elements.element_list import ListElement
from fifo_dev_dsl.dia.dsl.elements.intent import Intent
from fifo_dev_dsl.dia.dsl.elements.intent_evaluated_success import (
    IntentEvaluatedSuccess,
)
from fifo_dev_dsl.dia.dsl.elements.intent_runtime_error_resolver import (
    IntentRuntimeErrorResolver,
)
from fifo_dev_dsl.dia.dsl.elements.propagate_slots import PropagateSlots
from fifo_dev_dsl.dia.dsl.elements.query_fill import QueryFill
from fifo_dev_dsl.dia.dsl.elements.query_gather import QueryGather
from fifo_dev_dsl.dia.dsl.elements.query_user import QueryUser
from fifo_dev_dsl.dia.dsl.elements.same_as_previous import SameAsPreviousIntent
from fifo_dev_dsl.dia.dsl.elements.slot import Slot
from fifo_dev_dsl.dia.dsl.elements.value import Value
from fifo_dev_dsl.dia.dsl.elements.value_fuzzy import FuzzyValue
from fifo_dev_dsl.dia.dsl.elements.value_list import ListValue
from fifo_dev_dsl.dia.dsl.elements.value_return import ReturnValue
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationOutcome

# Elements that explicitly report unresolved state
@pytest.mark.parametrize(
    "element",
    [
        Ask("question"),
        QueryFill("value"),
        QueryGather("orig", "info"),
        QueryUser("query"),
        IntentRuntimeErrorResolver(Intent("foo", []), "err"),
    ],
)
def test_is_resolved_unresolved_elements(element: DslBase):
    assert element.is_resolved() is False


# Elements that are always considered resolved
@pytest.mark.parametrize(
    "element",
    [
        Abort(),
        AbortWithNewDsl(ListElement([])),
        Value("1"),
        FuzzyValue("a few"),
        SameAsPreviousIntent(),
        ReturnValue(Intent("bar", [])),
        IntentEvaluatedSuccess(Intent("baz", []), EvaluationOutcome()),
    ],
)
def test_is_resolved_resolved_elements(element: DslBase):
    assert element.is_resolved() is True


# Container nodes propagate resolution state from their children
@pytest.mark.parametrize(
    "element,expected",
    [
        (Slot("x", Value("1")), True),
        (Slot("x", Ask("?")), False),
        (Intent("foo", [Slot("x", Value("1"))]), True),
        (Intent("foo", [Slot("x", Ask("?"))]), False),
        (PropagateSlots([Slot("x", Value("1"))]), True),
        (PropagateSlots([Slot("x", Ask("?"))]), False),
        (ListElement([Value("1"), Abort()]), True),
        (ListElement([Value("1"), Ask("?")]), False),
        (ListValue([Value("1"), FuzzyValue("few")]), True),
    ],
)
def test_is_resolved_container_elements(element: DslBase, expected: bool):
    assert element.is_resolved() is expected
