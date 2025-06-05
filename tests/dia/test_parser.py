import pytest
from fifo_dev_dsl.dia.dsl.elements.intent import Intent
from fifo_dev_dsl.dia.dsl.elements.slot import Slot
from fifo_dev_dsl.dia.dsl.elements.value import Value
from fifo_dev_dsl.dia.dsl.elements.value_list import ListValue
from fifo_dev_dsl.dia.dsl.elements.value_return import ReturnValue
from fifo_dev_dsl.dia.dsl.parser.parser import parse_intent, parse_dsl_element
from fifo_dev_dsl.dia.dsl.elements.ask import Ask
from fifo_dev_dsl.dia.dsl.elements.abort import Abort
from fifo_dev_dsl.dia.dsl.elements.abort_with_new_dsl import AbortWithNewDsl
from fifo_dev_dsl.dia.dsl.elements.query_fill import QueryFill
from fifo_dev_dsl.dia.dsl.elements.query_gather import QueryGather
from fifo_dev_dsl.dia.dsl.elements.query_user import QueryUser
from fifo_dev_dsl.dia.dsl.elements.value_fuzzy import FuzzyValue
from fifo_dev_dsl.dia.dsl.elements.same_as_previous import SameAsPreviousIntent
from fifo_dev_dsl.dia.dsl.elements.element_list import ListElement
from fifo_dev_dsl.dia.dsl.elements.propagate_slots import PropagateSlots

@pytest.mark.parametrize("name,args,expected", [
    (
        "move",
        "x=1, y=2",
        Intent(
            name="move",
            slots=[
                Slot("x", Value("1")),
                Slot("y", Value("2")),
            ]
        )
    ),
    (
        "schedule",
        'day="Monday", time="10am"',
        Intent(
            name="schedule",
            slots=[
                Slot("day", Value("Monday")),
                Slot("time", Value("10am")),
            ]
        )
    ),
    (
        "compute",
        "result=add(a=1, b=2)",
        Intent(
            name="compute",
            slots=[
                Slot("result", ReturnValue(
                    Intent(
                        name="add",
                        slots=[
                            Slot("a", Value("1")),
                            Slot("b", Value("2")),
                        ]
                    )
                ))
            ]
        )
    ),
    (
        "math",
        "v=[1, 2, 3]",
        Intent(
            name="math",
            slots=[
                Slot("v", ListValue([
                    Value("1"),
                    Value("2"),
                    Value("3"),
                ]))
            ]
        )
    ),
    (
        "fallback",
        "x='hello, world', y=\"ok\"",
        Intent(
            name="fallback",
            slots=[
                Slot("x", Value("hello, world")),
                Slot("y", Value("ok")),
            ]
        )
    ),
])
def test_parse_intent_valid(name: str, args: str, expected: Intent):
    result = parse_intent(name, args)
    assert result.name == expected.name
    assert len(result.slots) == len(expected.slots)
    for r_slot, e_slot in zip(result.slots, expected.slots):
        assert r_slot.name == e_slot.name
        assert type(r_slot.value) == type(e_slot.value)
        assert r_slot.value == e_slot.value


@pytest.mark.parametrize("name,args",  [
    ("missing_equals", "x 1"),              # No '=' present
    ("partial_assignment", "a=1, b"),       # One correct, one incomplete
    ("empty_value", "x=, y=2"),             # Value is empty
    ("missing_key", "=1"),                  # Key missing
    ("missing_value", "x="),                # Value missing
    ("double_comma", "x=1,,"),              # Redundant comma
])
def test_parse_intent_invalid(name: str, args: str):
    with pytest.raises(ValueError):
        parse_intent(name, args)
# Additional tests for parse_dsl_element

@pytest.mark.parametrize(
    "text,wrap,expected",
    [
        ("'hello'", False, Value("hello")),
        ("123", False, Value("123")),
        ("[1, 2]", False, ListValue([Value("1"), Value("2")])),
        ("foo(x=1)", False, Intent(name="foo", slots=[Slot("x", Value("1"))])),
        ("foo(x=1)", True, ReturnValue(Intent(name="foo", slots=[Slot("x", Value("1"))]))),
        ("F(\"some\")", False, FuzzyValue("some")),
        ("ASK(\"what?\")", False, Ask("what?")),
        ("QUERY_FILL(\"value\")", False, QueryFill("value")),
        ("QUERY_USER(\"u?\")", False, QueryUser("u?")),
        ("QUERY_GATHER(\"orig\", \"info\")", False, QueryGather("orig", "info")),
        ("SAME_AS_PREVIOUS_INTENT()", False, SameAsPreviousIntent()),
        (
            "PROPAGATE_SLOT(x=1, y=foo())",
            False,
            PropagateSlots([
                Slot("x", Value("1")),
                Slot("y", ReturnValue(Intent(name="foo", slots=[]))),
            ]),
        ),
        (
            "ABORT_WITH_NEW_INTENTS([foo(), bar(x=2)])",
            False,
            AbortWithNewDsl(
                ListElement([
                    Intent(name="foo", slots=[]),
                    Intent(name="bar", slots=[Slot("x", Value("2"))]),
                ])
            ),
        ),
        ("ABORT()", False, Abort()),
        (
            "foo(count=ASK(\"c?\"), length=QUERY_FILL(\"between x and y\"))",
            False,
            Intent(
                name="foo",
                slots=[
                    Slot("count", Ask("c?")),
                    Slot("length", QueryFill("between x and y")),
                ],
            ),
        ),
        (
            "foo(count=4, length=SAME_AS_PREVIOUS_INTENT())",
            False,
            Intent(
                name="foo",
                slots=[
                    Slot("count", Value("4")),
                    Slot("length", SameAsPreviousIntent()),
                ],
            ),
        ),
        (
            "foo(count=F(\"some\"), length=ASK(\"which length?\"))",
            False,
            Intent(
                name="foo",
                slots=[
                    Slot("count", FuzzyValue("some")),
                    Slot("length", Ask("which length?")),
                ],
            ),
        ),
    ],
)
def test_parse_dsl_element_valid(text: str, wrap: bool, expected):
    result = parse_dsl_element(text, wrap)
    assert type(result) == type(expected)
    assert result == expected


@pytest.mark.parametrize(
    "text,wrap,list_type,list_content_type,exc",
    [
        ("", False, ListValue, Value, ValueError),
        ("PROPAGATE_SLOT(x)", False, ListValue, Value, ValueError),
        ("[foo()]", False, ListValue, Value, TypeError),
    ],
)
def test_parse_dsl_element_invalid(text: str, wrap: bool, list_type, list_content_type, exc):
    with pytest.raises(exc):
        parse_dsl_element(text, wrap, list_type, list_content_type)
