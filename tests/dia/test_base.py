import pytest
from fifo_dev_dsl.dia.dsl.parser.parser import parse_dsl_element


@pytest.mark.parametrize("dsl_str", [
    "\"hello\"",
    "123",
    "[1, 2]",
    "foo(x=1)",
    'F("some")',
    'ASK("what?")',
    'QUERY_FILL("value")',
    'QUERY_USER("u?")',
    'QUERY_GATHER("orig", "info")',
    "SAME_AS_PREVIOUS_INTENT()",
    "PROPAGATE_SLOT(x=1, y=foo())",
    "ABORT_WITH_NEW_INTENTS([foo(), bar(x=2)])",
    "ABORT()",
    'foo(count=ASK("c?"), length=QUERY_FILL("between x and y"))',
    "foo(count=4, length=SAME_AS_PREVIOUS_INTENT())",
    'foo(count=F("some"), length=ASK("which length?"))',
])
def test_to_dsl_representation_runs(dsl_str: str):
    obj = parse_dsl_element(dsl_str, wrap_intent_as_value=False)
    result = obj.to_dsl_representation()

    assert dsl_str == result

from fifo_dev_dsl.dia.dsl.elements.base import DslBase, DslContainerBase
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext


def test_dsl_base_leaf_mutations() -> None:
    base = DslBase()
    with pytest.raises(RuntimeError, match="DslBase is a leaf node"):
        base.update_child(0, DslBase())
    with pytest.raises(RuntimeError, match="DslBase is a leaf node"):
        base.insert_child(0, DslBase())
    with pytest.raises(RuntimeError, match="DslBase is a leaf node"):
        base.remove_child(0)


def test_dsl_base_eval_not_implemented() -> None:
    ctx = LLMRuntimeContext([], [])
    with pytest.raises(NotImplementedError):
        DslBase().eval(ctx)


def test_container_expected_type_not_implemented() -> None:
    container = DslContainerBase([])
    with pytest.raises(NotImplementedError):
        container._expected_type()


def test_container_eq_notimplemented() -> None:
    container = DslContainerBase([])
    result = container.__eq__(object())
    assert result is NotImplemented
