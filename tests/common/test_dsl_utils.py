import pytest
from fifo_dev_dsl.domain_specific.common.dsl_utils import split_dsl_args

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
