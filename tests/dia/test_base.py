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
