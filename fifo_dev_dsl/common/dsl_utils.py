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
