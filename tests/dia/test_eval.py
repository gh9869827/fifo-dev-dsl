import pytest
from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
from fifo_dev_common.introspection.tool_decorator import tool_handler
from fifo_dev_dsl.dia.dsl.elements.value import Value
from fifo_dev_dsl.dia.dsl.elements.value_fuzzy import FuzzyValue
from fifo_dev_dsl.dia.dsl.elements.value_list import ListValue
from fifo_dev_dsl.dia.dsl.elements.value_return import ReturnValue
from fifo_dev_dsl.dia.dsl.elements.intent import Intent
from fifo_dev_dsl.dia.dsl.elements.same_as_previous import SameAsPreviousIntent
from fifo_dev_dsl.dia.dsl.elements.slot import Slot
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext


class Demo:
    @tool_handler("add")
    def add(self, a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a (int):
                first number
            b (int):
                second number

        Returns:
            int: the sum
        """
        return a + b


def runtime_context(tools: list = None) -> LLMRuntimeContext:
    return LLMRuntimeContext(tools or [], [])


def test_value_eval_casts_and_requires_type() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("int")
    assert Value("3").eval(ctx, ty) == 3
    with pytest.raises(RuntimeError):
        Value("3").eval(ctx)


def test_fuzzy_value_eval() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("int")
    assert FuzzyValue("a few").eval(ctx, ty) == 3
    with pytest.raises(ValueError):
        FuzzyValue("lots").eval(ctx, ty)
    with pytest.raises(RuntimeError):
        FuzzyValue("few").eval(ctx)


def test_list_value_eval() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("list[int]")
    lst = ListValue([Value("1"), Value("2")])
    assert lst.eval(ctx, ty) == [1, 2]
    with pytest.raises(ValueError):
        lst.eval(ctx, MiniDocStringType("int"))
    with pytest.raises(RuntimeError):
        lst.eval(ctx)


def test_intent_and_return_value_eval() -> None:
    demo = Demo()
    ctx = runtime_context([demo.add])
    intent = Intent("add", [Slot("a", Value("2")), Slot("b", Value("3"))])
    assert intent.eval(ctx) == 5
    assert intent.eval(ctx, MiniDocStringType("float")) == 5.0
    rv = ReturnValue(intent)
    assert rv.eval(ctx, MiniDocStringType("int")) == 5
    with pytest.raises(RuntimeError):
        rv.eval(ctx)


def test_same_as_previous_intent_eval() -> None:
    with pytest.raises(NotImplementedError):
        SameAsPreviousIntent().eval(runtime_context(), MiniDocStringType("int"))
