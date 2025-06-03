from typing import Tuple, cast
import pytest
from fifo_dev_dsl.domain_specific.common.dsl_utils import parse_dsl_expression, split_dsl_args

@pytest.mark.parametrize(
    "args_str,expected",
    [
        ("a, b, c", ["a", "b", "c"]),
        ("a(b, c), d", ["a(b, c)", "d"]),
        ("a, b(c, d(e, f)), g", ["a", "b(c, d(e, f))", "g"]),
        ("", []),
        ("single", ["single"]),
        ("func1(), func2(x)", ["func1()", "func2(x)"]),
        ("outer(inner1(a, b), inner2(c)), d", ["outer(inner1(a, b), inner2(c))", "d"]),
        ("a(b(c(d, e)), f), g", ["a(b(c(d, e)), f)", "g"]),
        ("a, b(c, d), e(f, g(h, i)), j", ["a", "b(c, d)", "e(f, g(h, i))", "j"]),
        ("  a , b  ,c ", ["a", "b", "c"]),
        ("a(b, c(d, e(f))), g", ["a(b, c(d, e(f)))", "g"]),
        ("a(b, c), d(e, f(g, h)), i", ["a(b, c)", "d(e, f(g, h))", "i"]),
        ("a(b, c(d, e), f), g", ["a(b, c(d, e), f)", "g"]),
        ("OFFSET(TODAY, 2, DAY)", ["OFFSET(TODAY, 2, DAY)"]),
        ("SET_TIME(TODAY, 9, 0), OFFSET(TODAY, 1, DAY)", ["SET_TIME(TODAY, 9, 0)", "OFFSET(TODAY, 1, DAY)"]),
        # Mixed bracket and paren
        ("WEEKLY(1, [MO, TU])", ["WEEKLY(1, [MO, TU])"]),
        ("WRAP(SET_TIME(TODAY, 8, 0), [MO, TU])", ["WRAP(SET_TIME(TODAY, 8, 0), [MO, TU])"]),
    ]
)
def test_split_dsl_args(args_str: str, expected: list[str]):
    assert split_dsl_args(args_str) == expected

@pytest.mark.parametrize(
    "args_str,expected",
    [
        # Unbalanced parentheses: extra opening
        ("a(b, c", ValueError),

        # Unbalanced parentheses: extra closing
        ("a, b)", ValueError),

        # Misplaced commas (comma at start)
        (",a(b, c)", ValueError),

        # Incomplete function call
        ("a(", ValueError),

        # Only closing parenthesis
        (")", ValueError),

        # Garbage nesting (should raise or be handled)
        ("a(b(c, d), e))", ValueError),

        # Multiple unmatched nesting
        ("a((b, c), d", ValueError),

        # Misplaced commas (comma at end)
        ("a(b, c),", ValueError),

        # Mismatched group symbols
        ("a(b, [c, d))", ValueError),
        ("a[b, c)", ValueError),
        ("a(b, c]d)", ValueError),
        ("func((x, y], z)", ValueError),

        # Incomplete bracket group
        ("WEEKLY(1, [MO, TU", ValueError),
        ("WEEKLY(1, MO, TU])", ValueError),

        # Trailing comma with no value
        ("x, y, ", ValueError),

        # Double comma
        ("x,,y", ValueError),
    ]
)
def test_split_dsl_args_invalid(args_str: str, expected: list[str]):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            split_dsl_args(args_str)
    else:
        assert split_dsl_args(args_str) == expected

def dummy_evaluator(func_name: str, args: list[str]) -> Tuple[str, list[str]]:
    return (func_name, args)

@pytest.mark.parametrize(
    "expr,allow_bare,result",
    cast(Tuple[str, bool, list[str]], [
        ("FUNC(a, b, c)", False, ("FUNC", ["a", "b", "c"])),
        ("FUNC()", False, ("FUNC", [])),
        ("FUNC(  a  ,b )", False, ("FUNC", ["a", "b"])),
        ("OUTER(INNER(x, y), z)", False, ("OUTER", ["INNER(x, y)", "z"])),
        ("SINGLE", True, ("SINGLE", [])),
        ("  FUNC2(1, 2)  ", False, ("FUNC2", ["1", "2"])),
        ("NESTED(a(b, c), d)", False, ("NESTED", ["a(b, c)", "d"])),
        ("IDENT", True, ("IDENT", [])),
    ])
)
def test_parse_dsl_expression_valid(expr: str, allow_bare: bool, result: list[str]):
    assert parse_dsl_expression(expr, dummy_evaluator, allow_bare) == result

@pytest.mark.parametrize(
    "expr,allow_bare,exc_type",
    [
        ("FUNC(a, b", False, ValueError),  # Unbalanced
        ("FUNC(a, b))", False, ValueError),  # Unbalanced
        ("FUNC(", False, ValueError),  # Incomplete
        ("FUNC)", False, ValueError),  # Invalid
        ("FUNC(a,,b)", False, ValueError),  # Empty arg
        ("FUNC(a,)", False, ValueError),  # Trailing comma
        ("FUNC", False, ValueError),  # Bare identifier not allowed
        ("", False, ValueError),  # Empty string
        ("FUNC a, b", False, ValueError),  # Missing parentheses
    ]
)
def test_parse_dsl_expression_invalid(expr: str, allow_bare: bool, exc_type: type[ValueError]):
    with pytest.raises(exc_type):
        parse_dsl_expression(expr, dummy_evaluator, allow_bare)
