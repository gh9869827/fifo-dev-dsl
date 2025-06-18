from typing import Tuple, cast
import pytest
from fifo_dev_dsl.domain_specific.common.dsl_utils import parse_dsl_expression

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
        ("FUNC(a) extra", False, ValueError),  # Trailing garbage
    ]
)
def test_parse_dsl_expression_invalid(expr: str, allow_bare: bool, exc_type: type[ValueError]):
    with pytest.raises(exc_type):
        parse_dsl_expression(expr, dummy_evaluator, allow_bare)
