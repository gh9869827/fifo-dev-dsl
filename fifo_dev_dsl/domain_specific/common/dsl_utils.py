import re
from typing import Callable, TypeVar

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

T = TypeVar("T")

def parse_dsl_expression(
    expr: str,
    evaluator: Callable[[str, list[str]], T],
    allow_bare_identifiers: bool = False
) -> T:
    """
    Parses a DSL expression and dispatches to an evaluator.

    Args:
        expr (str):
            The DSL expression (e.g., "OFFSET(TODAY, 2, DAY)").

        evaluator (Callable):
            A function that takes (func_name, args_list) and returns the result.

        allow_bare_identifiers (bool):
            If True, allows expressions like "TODAY" with no parentheses as valid
            function calls with zero arguments.

    Returns:
        T:
            The result of evaluating the DSL.
    """
    expr = expr.strip()
    if '(' not in expr:
        if allow_bare_identifiers:
            return evaluator(expr, [])
        raise ValueError(f"Invalid expression (bare identifier not allowed): {expr}")

    match = re.match(r'(\w+)\((.*)\)', expr)
    if not match:
        raise ValueError(f"Invalid expression: {expr}")

    func, args_str = match.groups()
    args = split_dsl_args(args_str)

    return evaluator(func, args)


def get_arg(args: list[str], index: int) -> str:
    """
    Safely retrieves an argument by index or returns 'None' as a fallback string.

    Args:
        args (list[str]):
            The list of arguments passed to the DSL function.

        index (int):
            The index to retrieve.

    Returns:
        str:
            The argument at the specified index, or 'None' if missing.
    """
    return args[index] if len(args) > index else 'None'

def extract_int(args: list[str], index: int, field: str, keyword: str) -> int:
    """
    Extracts an integer value from the args list at the specified index.

    Args:
        args (list[str]):
            The list of argument strings.

        index (int):
            The index to extract from.

        field (str):
            The name of the field (for error message context).

        keyword (str):
            The DSL keyword calling this (for error message context).

    Returns:
        int:
            The parsed integer.

    Raises:
        ValueError:
            If the index is out of range or the value cannot be converted to int.
    """
    try:
        return int(args[index])
    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Invalid or missing {field} in {keyword}: got {get_arg(args, index)!r}"
        ) from e


def extract_month(args: list[str], index: int, keyword: str) -> int:
    """
    Extracts and validates a month (1-12) from the args list.

    Args:
        args (list[str]):
            The list of argument strings.

        index (int):
            The index to extract the month from.

        keyword (str):
            The DSL keyword calling this (for error message context).

    Returns:
        int:
            The validated month as an integer.

    Raises:
        ValueError:
            If the value is missing, not an int, or out of the valid month range.
    """
    month = extract_int(args, index, "month", keyword)
    if not 1 <= month <= 12:
        raise ValueError(f"Month {month} is out of range in {keyword} (expected 1-12)")
    return month

def extract_hour_minute(
    args: list[str],
    hour_index: int,
    minute_index: int,
    keyword: str
) -> tuple[int, int]:
    """
    Extracts and validates hour and minute from arguments.

    Args:
        args:
            The list of argument strings.

        hour_index:
            Index of the hour argument.

        minute_index:
            Index of the minute argument.

        keyword:
            The DSL keyword calling this (for error message context).

    Returns:
        A tuple (hour, minute).

    Raises:
        ValueError:
            If hour or minute is missing, not a valid integer, or out of valid range.
    """
    hour = extract_int(args, hour_index, "hour", keyword)

    if not 0 <= hour <= 23:
        raise ValueError(f"Hour {hour} is out of range in {keyword} (expected 0-23)")

    minute = extract_int(args, minute_index, "minute", keyword)

    if not 0 <= minute <= 59:
        raise ValueError(f"Minute {minute} is out of range in {keyword} (expected 0-59)")

    return hour, minute


def extract_positive_int(
    args: list[str],
    index: int,
    field: str,
    keyword: str
) -> int:
    """
    Extracts and validates a positive integer from the args list.

    Args:
        args:
            The list of argument strings.

        index:
            The index to extract from.

        field:
            The name of the field (for error message context).

        keyword:
            The DSL keyword calling this (for error message context).

    Returns:
        int:
            The parsed positive integer.

    Raises:
        ValueError:
            If the value is missing, not an integer, or not positive.
    """
    value = extract_int(args, index, field, keyword)
    if value <= 0:
        raise ValueError(f"{field.capitalize()} {value} is out of range in "
                         f"{keyword} (must be positive)")
    return value
