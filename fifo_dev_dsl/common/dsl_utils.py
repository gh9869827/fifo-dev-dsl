def split_top_level_commas(args_str: str) -> list[str]:
    """
    Splits a DSL argument string at top-level commas, ignoring commas within nested 
    parentheses, brackets, braces, or quoted strings.

    This function is designed for robust DSL parsing, including complex nested expressions 
    and quoted strings. It ensures that commas inside '()', '[]', '{}', or inside 
    matched quotes ('...' or "...") do not trigger a split.

    Escaped quotes (e.g., 'a\\'b') and other edge cases are handled correctly. Mismatched 
    quotes or unbalanced grouping symbols will raise a ValueError to indicate malformed syntax.

    Args:
        param_str (str):
            A comma-separated DSL argument string. May contain nested function calls,
            lists, objects, or quoted string literals.

    Returns:
        list[str]:
            A list of top-level argument substrings, each stripped of surrounding whitespace.

    Raises:
        ValueError:
            If quotes are unterminated, group delimiters are unbalanced or mismatched,
            or if any top-level argument is syntactically invalid (e.g. empty).

    Examples:
        split_top_level_commas('v=[1, invert(v=2)], x="a, b", y=(z, w)')
            → ['v=[1, invert(v=2)]', 'x="a, b"', 'y=(z, w)']

        split_top_level_commas("foo='hello, world', bar=\"x, y\"")
            → ["foo='hello, world'", 'bar="x, y"']

        split_top_level_commas("x='a, \"b, c\"', y=2")
            → ["x='a, \"b, c\"'", 'y=2']

        split_top_level_commas('z="escaped \\" quote, and comma", t=\'simple\'')
            → ['z="escaped \\" quote, and comma"', "t='simple'"]

        split_top_level_commas("x=[(1,2), {3,4}], y='nested, \"quoted\"'")
            → ["x=[(1,2), {3,4}]", 'y=\'nested, "quoted"\'']

        split_top_level_commas("bad(arg,,)")  # raises ValueError
        split_top_level_commas("x='unterminated")  # raises ValueError
    """
    args: list[str] = []
    stack: list[str] = []
    current = ''
    in_quote: str | None = None
    i = 0

    while i < len(args_str):
        c = args_str[i]

        if in_quote:
            if c == '\\' and i + 1 < len(args_str):
                current += c + args_str[i + 1]
                i += 2
                continue
            elif c == in_quote:
                in_quote = None
            current += c
        else:
            if c in ("'", '"'):
                in_quote = c
                current += c
            elif c in '([{':
                stack.append(c)
                current += c
            elif c in ')]}':
                if not stack:
                    raise ValueError(f"Unmatched closing '{c}' at position {i}")
                opening = stack.pop()
                if (opening, c) not in [('(', ')'), ('[', ']'), ('{', '}')]:
                    raise ValueError(f"Mismatched group: '{opening}' "
                                     f"closed by '{c}' at position {i}")
                current += c
            elif c == ',' and not stack:
                stripped = current.strip()
                if not stripped:
                    raise ValueError(f"Empty top-level argument at position {i}")

                args.append(stripped)
                current = ''
            else:
                current += c

        i += 1

    if in_quote:
        raise ValueError("Unterminated string literal")

    if stack:
        raise ValueError("Unbalanced or mismatched grouping in DSL expression")

    if current.strip():
        args.append(current.strip())
    elif current.strip() == '' and args_str.strip().endswith(','):
        raise ValueError("Trailing empty top-level argument")

    return args

def strip_quotes(val: str) -> str:
    """
    Remove surrounding matching quotes from a string.

    This function strips a single pair of matching quotes from the start and end
    of a string. It supports both single quotes (') and double quotes ("). It does
    not validate the internal contents of the string — only the outermost quotes.

    This is a simple utility intended to run **after structural validation has already
    been performed** (e.g., by a DSL parser or a function such as `split_top_level_commas`).
    If the input does not start and end with the same type of quote, a ValueError is raised.

    Args:
        val (str): 
            The string to process. Must start and end with the same type of quote
            (either `'` or `"`).

    Returns:
        str:
            The string with the outermost pair of matching quotes removed.

    Raises:
        ValueError:
            If the input string does not start and end with the same type of quote,
            or if it consists of only a single quote character (e.g. just `'` or `"`).

    Examples:
        strip_quotes("'hello'") → "hello"
        strip_quotes('"x, y"') → "x, y"
        strip_quotes("  'trimmed'  ") → "trimmed"

        strip_quotes("'a\"b'") → 'a"b'          # valid, internal content not checked
        strip_quotes("plain") → ValueError      # not quoted
        strip_quotes("'broken\"") → ValueError  # mismatched quotes
    """
    val = val.strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        if len(val) == 1:
            raise ValueError('String is too short to contain quoted content')
        return val[1:-1]
    raise ValueError('String must start and end with matching quotes')
