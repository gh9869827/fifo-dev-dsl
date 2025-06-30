"""Unit tests for `DslBase.pretty_print_dsl`."""

import io
from contextlib import redirect_stdout
from typing import Callable
import pytest

from fifo_dev_dsl.dia.dsl.parser.parser import parse_dsl_element
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.elements.element_list import ListElement
from fifo_dev_dsl.dia.dsl.elements.slot import Slot
from fifo_dev_dsl.dia.dsl.elements.intent import Intent
from fifo_dev_dsl.dia.dsl.elements.intent_evaluated_success import IntentEvaluatedSuccess
from fifo_dev_dsl.dia.dsl.elements.intent_runtime_error_resolver import IntentRuntimeErrorResolver
from fifo_dev_dsl.dia.dsl.elements.value_return import ReturnValue
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationOutcome


def expected_pretty(obj: DslBase, indent: int = 0) -> str:
    """Recursively build the expected pretty print output for `obj`."""
    prefix = "  " * indent
    if isinstance(obj, IntentEvaluatedSuccess):
        line = f"{prefix}{obj.__class__.__name__}(status={obj.evaluation_outcome.status.name})"
    elif isinstance(obj, IntentRuntimeErrorResolver):
        line = f"{prefix}{obj.__class__.__name__}(error_message={repr(obj.error_message)})"
    elif isinstance(obj, ReturnValue):
        line = f"{prefix}{obj.__class__.__name__}()"
    else:
        line = f"{prefix}{repr(obj)}"
    lines = [line]
    for child in obj.get_children():
        lines.append(expected_pretty(child, indent + 1))
    return "\n".join(lines)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: parse_dsl_element("'hello'", False),
        lambda: parse_dsl_element('F("few")', False),
        lambda: parse_dsl_element("SAME_AS_PREVIOUS_INTENT()", False),
        lambda: parse_dsl_element("foo(x=1)", True),
        lambda: parse_dsl_element("[1, 2]", False),
        lambda: parse_dsl_element('ASK("what?")', False),
        lambda: parse_dsl_element('QUERY_FILL("value")', False),
        lambda: parse_dsl_element('QUERY_GATHER("orig", "info")', False),
        lambda: parse_dsl_element('QUERY_USER("u")', False),
        lambda: parse_dsl_element('PROPAGATE_SLOT(x=1)', False),
        lambda: parse_dsl_element('ABORT()', False),
        lambda: parse_dsl_element('ABORT_WITH_NEW_INTENTS([foo()])', False),
        lambda: parse_dsl_element('foo(x=1)', False),
        lambda: Slot('x', parse_dsl_element('1', False)),
        lambda: parse_dsl_element('[foo()]', False, ListElement, DslBase),
        lambda: IntentEvaluatedSuccess(Intent('foo', []), EvaluationOutcome()),
        lambda: IntentRuntimeErrorResolver(Intent('foo', []), 'err'),
    ],
)
def test_pretty_print_dsl(builder: Callable[[], DslBase]):
    obj = builder()
    f = io.StringIO()
    with redirect_stdout(f):
        obj.pretty_print_dsl()
    out = f.getvalue().strip()

    assert out == expected_pretty(obj)


def test_return_value_pretty_print():
    """Verify `pretty_print_dsl` on ReturnValue shows the nested intent."""
    obj = parse_dsl_element("foo()", True)
    f = io.StringIO()
    with redirect_stdout(f):
        obj.pretty_print_dsl()
    out = f.getvalue().strip().splitlines()

    assert out == [
        "ReturnValue()",
        "  Intent(name='foo')",
    ]


def test_intent_nested_pretty_print():
    """Check indentation for nested DSL elements."""
    obj = parse_dsl_element("foo(x=bar(y=1))", False)
    f = io.StringIO()
    with redirect_stdout(f):
        obj.pretty_print_dsl()
    out = f.getvalue().strip().splitlines()

    assert out == [
        "Intent(name='foo')",
        "  Slot(name='x')",
        "    ReturnValue()",
        "      Intent(name='bar')",
        "        Slot(name='y')",
        "          Value(value=1)",
    ]

def test_intent_nested_pretty_print_string():
    """Check indentation for nested DSL elements."""
    obj = parse_dsl_element("foo(x=bar(str='string'))", False)
    f = io.StringIO()
    with redirect_stdout(f):
        obj.pretty_print_dsl()
    out = f.getvalue().strip().splitlines()

    assert out == [
        "Intent(name='foo')",
        "  Slot(name='x')",
        "    ReturnValue()",
        "      Intent(name='bar')",
        "        Slot(name='str')",
        "          Value(value='string')",
    ]
