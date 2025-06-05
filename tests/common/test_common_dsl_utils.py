from typing import Type
import pytest
from fifo_dev_dsl.domain_specific.common.dsl_utils import split_top_level_commas

@pytest.mark.parametrize(
    "args_str,expected",
    [
        # 🧪 Basic cases
        ("a, b, c", ["a", "b", "c"]),
        ("a=1, b=2", ["a=1", "b=2"]),
        ("", []),
        ("single", ["single"]),
        ("  a , b  ,c ", ["a", "b", "c"]),
        ("  a=1 , b = 2  ", ["a=1", "b = 2"]),

        # 🧮 Nested parentheses/brackets/braces
        ("a(b, c), d", ["a(b, c)", "d"]),
        ("a(b(c(d, e)), f), g", ["a(b(c(d, e)), f)", "g"]),
        ("a(b, c(d, e(f))), g", ["a(b, c(d, e(f)))", "g"]),
        ("a(b, c), d(e, f(g, h)), i", ["a(b, c)", "d(e, f(g, h))", "i"]),
        ("a(b, c(d, e), f), g", ["a(b, c(d, e), f)", "g"]),
        ("a, b(c, d(e, f)), g", ["a", "b(c, d(e, f))", "g"]),
        ("a, b(c, d), e(f, g(h, i)), j", ["a", "b(c, d)", "e(f, g(h, i))", "j"]),
        ("func1(), func2(x)", ["func1()", "func2(x)"]),
        ("outer(inner1(a, b), inner2(c)), d", ["outer(inner1(a, b), inner2(c))", "d"]),
        ("OFFSET(TODAY, 2, DAY)", ["OFFSET(TODAY, 2, DAY)"]),
        ("SET_TIME(TODAY, 9, 0), OFFSET(TODAY, 1, DAY)", ["SET_TIME(TODAY, 9, 0)", "OFFSET(TODAY, 1, DAY)"]),
        ("WEEKLY(1, [MO, TU])", ["WEEKLY(1, [MO, TU])"]),
        ("WRAP(SET_TIME(TODAY, 8, 0), [MO, TU])", ["WRAP(SET_TIME(TODAY, 8, 0), [MO, TU])"]),
        ("v=[1,2,3], w=4", ["v=[1,2,3]", "w=4"]),
        ("list=[invert(x=1), 2], text=\"a, b\"", ['list=[invert(x=1), 2]', 'text="a, b"']),
        ("x=[(1,2), {3,4}], y='nested, \"quoted\"'", ["x=[(1,2), {3,4}]", 'y=\'nested, "quoted"\'']),
        ("a(b, ), c", ["a(b, )", "c"]),  # trailing comma inside nested context is allowed

        # 🧵 Strings with commas
        ('x="a, b", y="c, d"', ['x="a, b"', 'y="c, d"']),
        ("x='a, b', y='c, d'", ["x='a, b'", "y='c, d'"]),
        ("a='x, y', b=(1, 2)", ["a='x, y'", "b=(1, 2)"]),
        ("x='a, \"b, c\"', y=2", ['x=\'a, "b, c"\'', 'y=2']),

        # 🧊 Escaped quotes and quote-in-quote
        ("a=\"nested 'quote' inside\", b=2", ["a=\"nested 'quote' inside\"", "b=2"]),
        ("a='nested \"quote\" inside', b=3", ["a='nested \"quote\" inside'", "b=3"]),
        ("escaped=\"a, \\\"b, c\\\"\", d=1", ["escaped=\"a, \\\"b, c\\\"\"", "d=1"]),
        ("x='', y=\"\"", ["x=''", 'y=""']),
        ("empty=''", ["empty=''"]),
    ]
)
def test_split_dsl_args(args_str: str, expected: list[str]):
    assert split_top_level_commas(args_str) == expected

@pytest.mark.parametrize(
    "args_str,expected",
    [
        # Unbalanced parentheses/brackets
        ("a(b, c", ValueError),         # unbalanced (
        ("a[b, c", ValueError),         # unbalanced [
        ("a, b)", ValueError),          # extra closing )
        ("a(b, [c, d}", ValueError),    # mismatched group

        # Incomplete constructs
        ("a(", ValueError),             # incomplete call
        ("a='unterminated string", ValueError),  # unclosed quote
        ("x='unterminated", ValueError),
        ("x=\"unterminated", ValueError),

        # Comma issues
        (",a(b, c)", ValueError),       # comma at start
        ("a=b,,c", ValueError),         # double comma
        ("x,,y", ValueError),
        ("a=b,", ValueError),           # trailing comma
        ("x, y, ", ValueError),         # trailing comma
        (" , , ", ValueError),          # only commas
        ("a(b, c), ,", ValueError),     # garbage split
        ("a(b, c),", ValueError),       # trailing comma at top level

        # Garbage nesting
        ("a(b(c, d), e))", ValueError),
        ("a((b, c), d", ValueError),

        # Mismatched group symbols
        ("a(b, [c, d))", ValueError),
        ("a[b, c)", ValueError),
        ("a(b, c]d)", ValueError),
        ("func((x, y], z)", ValueError),
        ("x=[1, 2), y=3", ValueError),  # mismatched close

        # Quotes mismatched or escaped incorrectly
        ("x='a, b\", y=2", ValueError),     # mismatched quotes
        ("x=\"a, 'b, y=2", ValueError),     # mismatched quotes
        ("x='a, b\\", ValueError),          # escape at end
        ("x='a\\", ValueError),
        ("x='\\'", ValueError),             # escape inside
        ("x='a, b', y='", ValueError),      # second quote missing

        # Only quote
        ("'", ValueError),
        ("\"", ValueError),
    ]
)
def test_split_dsl_args_invalid(args_str: str, expected: Type[Exception]):
    with pytest.raises(expected):
        split_top_level_commas(args_str)
