def split_dsl_args(args_str: str) -> list[str]:
    """
    Splits a function argument string into individual components, accounting for nested
    parentheses and brackets in DSL expressions. Ensures properly matched groupings.

    Args:
        args_str (str):
            Argument string from the DSL expression, possibly nested.

    Returns:
        list[str]:
            A list of argument strings split at top-level commas.

    Raises:
        ValueError:
            If groupings are mismatched, unbalanced, or a top-level argument is empty.
    """
    args: list[str] = []
    stack: list[str] = []
    current = ''

    for i, c in enumerate(args_str):
        if c == ',' and not stack:
            stripped = current.strip()
            if not stripped:
                raise ValueError(f"Empty top-level argument at position {i}")
            args.append(stripped)
            current = ''
        else:
            if c in '([':
                stack.append(c)
            elif c == ')':
                if not stack or stack[-1] != '(':
                    raise ValueError(f"Mismatched or unmatched ')' at position {i}")
                stack.pop()
            elif c == ']':
                if not stack or stack[-1] != '[':
                    raise ValueError(f"Mismatched or unmatched ']' at position {i}")
                stack.pop()
            current += c

    if stack:
        raise ValueError("Unbalanced or mismatched grouping in DSL expression")

    if current.strip():
        args.append(current.strip())
    elif current.strip() == '' and args_str.strip().endswith(','):
        raise ValueError("Trailing empty top-level argument")

    return args
