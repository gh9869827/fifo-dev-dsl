"""
DIA: A Dialog-Interactive DSL for Intent Resolution

DIA is a lightweight, structured Domain-Specific Language (DSL) designed for
interactive agents. It represents not only function-like intent calls but also
the logic required to resolve missing information through dialog or runtime context.

Unlike traditional function call systems, DIA supports **partial, deferred evaluation**
and integrates **interaction loops** directly into the syntax using constructs like:

- `ASK("...")`: Prompt the user to provide a missing value.
- `QUERY_USER("...")`: Pull information the user has requested previously.
- `QUERY_FILL("...")`: Resolve slot values by querying runtime state (e.g., inventory).
- `PROPAGATE_SLOT(...)`: Forward values inferred from prior context or dialog.
- `ABORT()` / `ABORT_WITH_NEW_INTENTS(...)`: Interrupt and redirect execution flow.

Key Features:
- Intent calls with named arguments: `get_screw(count=4, length=QUERY_FILL("8mm to 12mm"))`
- Nested expressions and value composition: `add(v=[1, negate(v=2)])`
- Interactive evaluation and slot resolution: `ASK("What length?")`, `QUERY_FILL(...)`
- Declarative control flow modeling with aborts and rewrites

Parsing API:
- `parse_dsl_element()`: Recursively parses a single expression into a `DslBase` instance
- `parse_intent()`: Parses a structured intent call with named parameters
- `split_top_level_commas()`: Robust splitting of nested argument lists
- `parse_dsl()`: Parses a comma-separated sequence of DSL expressions

DIA is ideal for AI systems where the user may start a command without specifying
all the details. Instead of rejecting incomplete input, DIA helps agents engage,
clarify, and complete tasks collaboratively.

Example:
    >
    Give me screws with a length between 8mm and 12mm
    <
    parse_dsl('get_screw(count=ASK("How many 12mm screws do you need?"), 
                         length=QUERY_FILL("between 8mm and 12mm"))')

This produces a structured representation that can be executed incrementally
as slot values are resolved through dialog or context.
"""


from typing import List, Type, TypeVar

from common.llm.dia.dsl.elements.abort import Abort
from common.llm.dia.dsl.elements.abort_with_new_dsl import AbortWithNewDsl
from common.llm.dia.dsl.elements.ask import Ask
from common.llm.dia.dsl.elements.base import DslBase
from common.llm.dia.dsl.elements.element_list import ListElement
from common.llm.dia.dsl.elements.intent import Intent
from common.llm.dia.dsl.elements.propagate_slots import PropagateSlots
from common.llm.dia.dsl.elements.query_fill import QueryFill
from common.llm.dia.dsl.elements.query_user import QueryUser
from common.llm.dia.dsl.elements.same_as_previous import SameAsPreviousIntent
from common.llm.dia.dsl.elements.slot import Slot
from common.llm.dia.dsl.elements.value_base import DSLValueBase
from common.llm.dia.dsl.elements.value_fuzzy import FuzzyValue
from common.llm.dia.dsl.elements.value_list import ListValue
from common.llm.dia.dsl.elements.value_return import ReturnValue
from common.llm.dia.dsl.elements.value import Value
from common.typeutils.cast import strict_cast

def split_top_level_commas(param_str: str) -> List[str]:
    """
    Split a DSL argument string by top-level commas, preserving nested structures.

    This function is designed for parsing DIA's mini DSL, where function calls may
    include deeply nested lists, sub-expressions, and quoted strings. It ensures that 
    commas inside brackets, parentheses, or quotes do not trigger a split.

    Args:
        param_str (str):
            A comma-separated argument string from the DSL. Can include nested calls,
            list literals, and quoted values.

    Returns:
        List[str]:
            A list of argument substrings, each trimmed of surrounding whitespace.

    Example:
        split_top_level_commas('v=[1, negate(v=2)], x="a, b", y=(z, w)') →
            ['v=[1, negate(v=2)]', 'x="a, b"', 'y=(z, w)']
    """
    args = []
    level = 0
    in_quote = False
    i = 0
    start_idx = i
    while i < len(param_str):
        if param_str[i] == '"':
            in_quote = not in_quote
        elif not in_quote:
            if param_str[i] in '([':
                level += 1
            elif param_str[i] in ')]':
                level -= 1
            elif param_str[i] == ',' and level == 0:
                args.append(param_str[start_idx:i].strip())
                i += 1
                start_idx = i
        i += 1

    if i > start_idx:
        args.append(param_str[start_idx:i].strip())

    return args


def strip_quotes(val: str) -> str:
    """
    Remove surrounding double quotes from a string.

    Args:
        val (str): 
            The string to process. Must start and end with a double quote (`"`).

    Returns:
        str:
            The string without the outer double quotes.

    Raises:
        ValueError:
            If the input string does not start and end with a double quote.
    """
    val = val.strip()
    if not (val.startswith('"') and val.endswith('"')):
        raise ValueError(f'String must start and end with a double quote: {val!r}')
    return val[1:-1]

def parse_intent(name: str, args: str) -> Intent:
    """
    Parse a DSL intent call with named arguments into an Intent object.

    This function processes a single DIA mini DSL intent expression, extracting
    key-value arguments and recursively parsing their values. It supports both
    top-level and nested usage (e.g., as return or abort values).

    Args:
        name (str):
            The name of the intent (e.g., "move", "multiply", "schedule").

        args (str):
            A comma-separated string of named arguments (e.g., 'v=[1, negate(v=2)]').

    Returns:
        Intent:
            An Intent object with parsed parameters.

    Raises:
        ValueError:
            If any argument is missing an '=' sign.

    Example:
        parse_intent("move", 'x=add(a=1, b=2), y=-1, speed="fast"')
    """
    slots = []
    for arg in split_top_level_commas(args):
        if '=' in arg:
            k, v = arg.split('=', 1)
            slots.append(Slot(k.strip(), parse_dsl_element(v.strip(), True)))
        else:
            raise ValueError("Intent args missing =")

    return Intent(name=name, slots=slots)

T = TypeVar("T", bound=DslBase)
U = TypeVar("U", bound=DslBase)

def parse_dsl_element(text: str,
                      wrap_intent_as_value: bool,
                      list_type: Type[T] = ListValue,
                      list_content_type: Type[U] = DSLValueBase) -> DslBase:
    """
    Parse a single DSL element from a string into its corresponding object.

    This function handles literals (numbers, strings), lists, and function-like
    constructs such as intents and built-ins. It supports recursive parsing
    for nested structures and can optionally wrap top-level intents in a
    ReturnValue when used in value contexts.

    Args:
        text (str):
            The DSL expression to parse.

        wrap_intent_as_value (bool):
            If True, top-level intents will be wrapped in a ReturnValue object.

        list_type (Type[T], optional):
            The container class to use when parsing arrays, typically `ListValue`
            or a subclass. Must accept a list of `list_content_type` instances as input.

        list_content_type (Type[U], optional):
            The expected type for each element inside the list. All parsed elements
            must be instances of this type, or a `TypeError` is raised.

    Returns:
        DslBase:
            The parsed representation of the DSL element.

    Raises:
        ValueError:
            If the input is empty or arguments are malformed.

        TypeError:
            If list elements do not match the expected type.
    """

    text = text.strip()

    if not text:
        raise ValueError("Empty element.")

    if text.startswith('[') and text.endswith(']'):
        # array
        return list_type(
            [strict_cast(
                list_content_type,
                parse_dsl_element(value, wrap_intent_as_value, list_type, list_content_type))
             for value in split_top_level_commas (text[1:-1])
            ]
        )

    if text.startswith('"') and text.endswith('"'):
        # string
        return Value(text[1:-1])

    if "(" in text and text.endswith(")"):
        # intent and builtings
        open_paren = text.find("(")
        name = text[:open_paren].strip()
        args = text[open_paren+1:-1].strip()

        if name == "F":
            return FuzzyValue(strip_quotes(args))

        if name == "ASK":
            return Ask(strip_quotes(args))

        if name == "QUERY_FILL":
            return QueryFill(strip_quotes(args))

        if name == "QUERY_USER":
            return QueryUser(strip_quotes(args))

        if name == "SAME_AS_PREVIOUS_INTENT":
            return SameAsPreviousIntent()

        if name == "PROPAGATE_SLOT":
            slots = []
            for s in split_top_level_commas (args):
                if '=' in s:
                    k, v = s.split('=', 1)
                    slots.append(Slot(k.strip(), parse_dsl_element(v.strip(), True)))
                else:
                    raise ValueError("Propagated slots args missing =")
            return PropagateSlots(slots)

        if name == "ABORT_WITH_NEW_INTENTS":
            return AbortWithNewDsl(
                strict_cast(ListElement, parse_dsl_element(args, False, ListElement, DslBase))
            )

        if name == "ABORT":
            return Abort()

        if wrap_intent_as_value:
            return ReturnValue(parse_intent(name, args))

        return parse_intent(name, args)

    # numbers
    return Value(text)

def parse_dsl(dsl_input: str) -> ListElement:
    """
    Parse a top-level DSL expression containing multiple elements.

    Splits the input string by top-level commas and parses each element
    into its corresponding DSL object. This is the entry point for parsing
    a sequence of expressions.

    Args:
        dsl_input (str):
            A comma-separated DSL expression string (e.g., 'add(v=1), ABORT()').

    Returns:
        ListElement:
            The list of parsed root DSL elements, which are instances of DslBase subclasses.

    Raises:
        ValueError:
            If any element in the input is malformed.
    """
    return ListElement(
        items=[parse_dsl_element(element, False) for element in split_top_level_commas(dsl_input)]
    )
