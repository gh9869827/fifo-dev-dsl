import random
import string
from typing import Union
import argparse

Tree = tuple[str, Union[float, tuple["Tree", "Tree"]]]

def generate_balanced_random_number() -> int:
    """
    Generate a signed integer by first selecting a magnitude range
    uniformly from five fixed digit buckets, then sampling uniformly
    within that range, and randomly flipping the sign.

    Returns:
        int: Random signed integer from [-99999, 99999], 
             excluding negative zero.
    """
    ranges = [
        (0, 9),
        (10, 99),
        (100, 999),
        (1000, 9999),
        (10000, 99999)
    ]
    low, high = random.choice(ranges)
    number = random.randint(low, high)
    if number != 0 and random.choice([True, False]):
        number *= -1
    return number


def create_tree(nb_total_values: int, prev_op: str = "", side: str = "") -> Tree:
    """
    Recursively generate a random binary expression tree.

    Each tree node contains a binary operator ("+", "-", "*", "/") and two children.
    Leaf nodes are values between -99999 and 99999.

    Args:
        nb_total_values (int):
            Total number of value nodes (i.e. leaves) in the tree.

        prev_op (str):
            Operator of the parent node. Used to prevent repetition on the left child,
            ensuring unambiguous parsing and deterministic reconstruction from infix.

        side (str):
            Indicates whether this node is the left or right child.

    Returns:
        Tree:
            A nested tuple-based representation of the expression tree.
    """
    if nb_total_values == 1:
        return ("v", generate_balanced_random_number())

    if nb_total_values == 2:
        nb_left, nb_right = 1, 1
    else:
        nb_left = random.randrange(1, nb_total_values)
        nb_right = nb_total_values - nb_left

    if side == "left":
        op = random.choice([o for o in ["+", "-", "*", "/"] if o != prev_op])
    else:
        op = random.choice(["+", "-", "*", "/"])

    return (op, (create_tree(nb_left, op, "left"), create_tree(nb_right, op, "right")))


def pretty_print_user(tree: Tree) -> str:
    """
    Convert an expression tree to a user-readable infix string.

    Applies parentheses based on operator precedence and associativity.

    Args:
        tree (Tree):
            The expression tree to convert.

    Returns:
        str:
            The formatted string representation of the expression.
    """
    op_center, values = tree

    if op_center == "v":
        return str(values)

    assert isinstance(values, tuple)

    left = pretty_print_user(values[0])
    right = pretty_print_user(values[1])

    op_left = values[0][0]
    op_right = values[1][0]

    need_parenthesis_left = (
        # (1 + 2) * 4
        op_left in ("+", "-") and op_center in ("*", "/")
    )

    need_parenthesis_right = (
        # 1 * (2 + 4)
        op_center in ("*", "/") and op_right in ("+", "-")
        # 1 - (2 + 4)
        or op_center == "-" and op_right in ("+", "-")
        # 7986 / (2790 / 921)
        or op_center == "/" and op_right in ("*", "/")
    )

    result = ""

    if need_parenthesis_left:
        result += f"( {left} )"
    else:
        result += left

    result += f" {op_center} "

    if need_parenthesis_right:
        result += f"( {right} )"
    else:
        result += right

    return result


def pretty_print_dsl(tree: Tree, fct: dict[str, dict[str, str | list[str]]]) -> str:
    """
    Convert an expression tree to a structured DSL string using randomized function and argument
    names.

    Args:
        tree (Tree):
            The expression tree to convert. Each node is either:
              - A value node: ("v", number)
              - An operator node: (op, (left_subtree, right_subtree)), where `op` is one of 
                "+", "-", "*", "/"

        fct (dict[str, dict[str, str | list[str]]]):
            A mapping from canonical operators ("add", "subtract", "divide", "multiply") to 
            randomized names:
              - "name": randomized function name
              - "args": list of two argument names (e.g., ["x", "y"])

    Returns:
        str:
            A string representing the DSL expression in functional form, e.g.,
            `abc(a=123, b=xyz(...))` with randomized function and argument names.
    """
    op, values = tree

    if op == "v":
        return f"{values}"

    assert isinstance(values, tuple)

    op_to_fct = {
        "*": "multiply",
        "+": "add",
        "/": "divide",
        "-": "subtract"
    }

    return (f'{fct[op_to_fct[op]]["name"]}('
            f'{fct[op_to_fct[op]]["args"][0]}={pretty_print_dsl(values[0], fct)}, '
            f'{fct[op_to_fct[op]]["args"][1]}={pretty_print_dsl(values[1], fct)}'
            ')')


def evaluate_tree(tree: Tree) -> float:
    """
    Recursively evaluate a binary expression tree.

    Args:
        tree (Tree):
            The expression tree to evaluate.

    Returns:
        float:
            The result of evaluating the expression.
    """
    op, values = tree

    if op == "v":
        assert isinstance(values, float | int)
        return float(values)

    assert isinstance(values, tuple)
    left = evaluate_tree(values[0])
    right = evaluate_tree(values[1])

    if op == "+":
        return left + right
    elif op == "-":
        return left - right
    elif op == "*":
        return left * right
    elif op == "/":
        if right == 0:
            raise ZeroDivisionError("Division by zero in expression tree.")
        return left / right
    else:
        raise ValueError(f"Unknown operator: {op}")


def get_system_prompt(fct: dict[str, dict[str, str | list[str]]]) -> str:
    """
    Generate a system prompt that defines four arithmetic intents with randomized names.

    Each intent corresponds to one of the operations: add, subtract, multiply, or divide.
    It includes parameter names and descriptions using randomized argument identifiers.

    Args:
        fct (dict[str, dict[str, str | list[str]]]):
            A dictionary mapping canonical operation names ("add", "subtract", "divide", "multiply")
            to a sub-dictionary with:
              - "name": randomized function name (str)
              - "args": list of two randomized argument names (list[str])

    Returns:
        str:
            A formatted system prompt suitable for intent sequencing, listing each intent,
            its parameters, and return type.
    """
    return f"""You are a precise intent sequencer. You parse the user's prompt and split it into atomic intents that match one of the defined intents below:

- intent: {fct['add']['name']}
  description: Add two numbers.
  parameters:
    - name: {fct['add']['args'][0]}
      type: float
      description: first number to add
      optional: False
    - name: {fct['add']['args'][1]}
      type: float
      description: second number to add
      optional: False
  return:
    type: float
    description: the sum of {fct['add']['args'][0]} and {fct['add']['args'][1]}
- intent: {fct['subtract']['name']}
  description: Subtract two numbers.
  parameters:
    - name: {fct['subtract']['args'][0]}
      type: float
      description: the number to subtract from
      optional: False
    - name: {fct['subtract']['args'][1]}
      type: float
      description: the number to subtract
      optional: False
  return:
    type: float
    description: the result of {fct['subtract']['args'][0]} - {fct['subtract']['args'][1]}
- intent: {fct['divide']['name']}
  description: Divide two numbers.
  parameters:
    - name: {fct['divide']['args'][0]}
      type: float
      description: numerator
      optional: False
    - name: {fct['divide']['args'][1]}
      type: float
      description: denominator
      optional: False
  return:
    type: float
    description: the result of {fct['divide']['args'][0]} / {fct['divide']['args'][1]}
- intent: {fct['multiply']['name']}
  description: Multiply two numbers.
  parameters:
    - name: {fct['multiply']['args'][0]}
      type: float
      description: first number
      optional: False
    - name: {fct['multiply']['args'][1]}
      type: float
      description: second number
      optional: False
  return:
    type: float
    description: the product of {fct['multiply']['args'][0]} and {fct['multiply']['args'][1]}

QUERY_FILL cannot be used as no information can be retrieved at runtime."""


def random_function_name() -> str:
    """
    Generate a random lowercase function name with 3 to 7 letters.

    Returns:
        str: Randomly generated function name.
    """
    length = random.randint(3, 7)
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def random_argument_name() -> str:
    """
    Generate a random lowercase argument name with 1 to 2 letters.

    Returns:
        str: Randomly generated argument name.
    """
    length = random.randint(1, 2)
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def random_function_signature() -> dict[str, dict[str, str | list[str]]]:
    """
    Generate randomized mappings for basic arithmetic function names and their arguments.

    Each operation ("add", "subtract", "divide", "multiply") maps to a dictionary:
      - "name": a unique randomly generated function name (3 to 7 lowercase letters)
      - "args": a list of two distinct randomly generated argument names (1 to 2 lowercase letters).
        These names are local to each function and may repeat across different functions.

    Returns:
        dict[str, dict[str, str | list[str]]]: 
            Mapping from canonical operation names to their randomized names and arguments.
    """
    used_function_names: set[str] = set()

    def unique_function_name() -> str:
        while True:
            name = random_function_name()
            if name not in used_function_names:
                used_function_names.add(name)
                return name

    def two_unique_args() -> list[str]:
        arg1 = random_argument_name()
        while True:
            arg2 = random_argument_name()
            if arg2 != arg1:
                return [arg1, arg2]

    return {
        op: {
            "name": unique_function_name(),
            "args": two_unique_args()
        }
        for op in ["add", "subtract", "divide", "multiply"]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Number of examples to generate")
    args = parser.parse_args()

    print("---")
    for _ in range(args.n):
        fct_signature = random_function_signature()
        t = create_tree(random.randrange(3, 9))
        print("$")
        print(get_system_prompt(fct_signature))
        print(">", pretty_print_user(t))
        print("<", pretty_print_dsl(t, fct_signature))
        print("---")
