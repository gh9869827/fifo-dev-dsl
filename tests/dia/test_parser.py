import pytest
from fifo_dev_dsl.dia.dsl.elements.intent import Intent
from fifo_dev_dsl.dia.dsl.elements.slot import Slot
from fifo_dev_dsl.dia.dsl.elements.value import Value
from fifo_dev_dsl.dia.dsl.elements.value_list import ListValue
from fifo_dev_dsl.dia.dsl.elements.value_return import ReturnValue
from fifo_dev_dsl.dia.dsl.parser.parser import parse_intent

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
