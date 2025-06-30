import pytest
from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
from fifo_dev_common.introspection.tool_decorator import ToolHandler, tool_handler
# Importing the parser first avoids circular import issues when loading query
# elements. This is needed when running this file on its own.
from fifo_dev_dsl.dia.dsl.parser.parser import parse_dsl   # noqa, pylint: disable=unused-import, # pyright: ignore[reportUnusedImport]
from fifo_dev_dsl.dia.dsl.elements.abort import Abort
from fifo_dev_dsl.dia.dsl.elements.abort_with_new_dsl import AbortWithNewDsl
from fifo_dev_dsl.dia.dsl.elements.ask import Ask
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.elements.element_list import ListElement
from fifo_dev_dsl.dia.dsl.elements.intent_runtime_error_resolver import IntentRuntimeErrorResolver
from fifo_dev_dsl.dia.dsl.elements.query_fill import QueryFill
from fifo_dev_dsl.dia.dsl.elements.query_gather import QueryGather
from fifo_dev_dsl.dia.dsl.elements.query_user import QueryUser
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


def runtime_context(tools: list[ToolHandler] | None = None) -> LLMRuntimeContext:
    return LLMRuntimeContext(tools or [], [])


def test_value_eval() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("int")
    assert ty.cast(Value("3").eval(ctx)) == 3


@pytest.mark.asyncio
async def test_value_eval_async() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("int")
    assert ty.cast(await Value("3").eval_async(ctx)) == 3


def test_fuzzy_value_eval() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("int")
    assert ty.cast(FuzzyValue("a few").eval(ctx)) == 3
    with pytest.raises(ValueError):
        FuzzyValue("lots").eval(ctx)


@pytest.mark.asyncio
async def test_fuzzy_value_eval_async() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("int")
    assert ty.cast(await FuzzyValue("a few").eval_async(ctx)) == 3
    with pytest.raises(ValueError):
        await FuzzyValue("lots").eval_async(ctx)


def test_list_value_eval() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("list[int]")
    lst = ListValue([Value("1"), Value("2")])
    assert ty.cast(lst.eval(ctx)) == [1, 2]


@pytest.mark.asyncio
async def test_list_value_eval_async() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("list[int]")
    lst = ListValue([Value("1"), Value("2")])
    assert ty.cast(await lst.eval_async(ctx)) == [1, 2]


def test_list_element_eval_1() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("list[int]")
    lst = ListElement([Value("1"), Value("2")])
    assert ty.cast(lst.eval(ctx)) == [1, 2]


@pytest.mark.asyncio
async def test_list_element_eval_1_async() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("list[int]")
    lst = ListElement([Value("1"), Value("2")])
    assert ty.cast(await lst.eval_async(ctx)) == [1, 2]


def test_list_element_eval_2() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("list[str]")
    lst = ListElement([Value("1"), Value("2")])
    assert ty.cast(lst.eval(ctx)) == ["1", "2"]


@pytest.mark.asyncio
async def test_list_element_eval_2_async() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("list[str]")
    lst = ListElement([Value("1"), Value("2")])
    assert ty.cast(await lst.eval_async(ctx)) == ["1", "2"]


def test_list_element_eval_3() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("list[int]")
    lst = ListElement([Value(1), Value(2)])
    assert ty.cast(lst.eval(ctx)) == [1, 2]


@pytest.mark.asyncio
async def test_list_element_eval_3_async() -> None:
    ctx = runtime_context()
    ty = MiniDocStringType("list[int]")
    lst = ListElement([Value(1), Value(2)])
    assert ty.cast(await lst.eval_async(ctx)) == [1, 2]


def test_intent_and_return_value_eval() -> None:
    demo = Demo()
    ctx = runtime_context([demo.add])
    intent = Intent("add", [Slot("a", Value("2")), Slot("b", Value("3"))])
    assert intent.eval(ctx) == 5
    assert MiniDocStringType("float").cast(intent.eval(ctx)) == 5.0
    rv = ReturnValue(intent)
    assert MiniDocStringType("int").cast(rv.eval(ctx)) == 5


@pytest.mark.asyncio
async def test_intent_and_return_value_eval_async() -> None:
    demo = Demo()
    ctx = runtime_context([demo.add])
    intent = Intent("add", [Slot("a", Value("2")), Slot("b", Value("3"))])
    assert await intent.eval_async(ctx) == 5
    assert MiniDocStringType("float").cast(await intent.eval_async(ctx)) == 5.0
    rv = ReturnValue(intent)
    assert MiniDocStringType("int").cast(await rv.eval_async(ctx)) == 5


def test_same_as_previous_intent_eval() -> None:
    with pytest.raises(NotImplementedError):
        SameAsPreviousIntent().eval(runtime_context())


@pytest.mark.parametrize(
    "obj",
    [
        Abort(),
        Ask("question"),
        QueryFill("value"),
        QueryGather("orig", "info"),
        QueryUser("query"),
        AbortWithNewDsl(ListElement([])),
        IntentRuntimeErrorResolver(Intent("foo", []), "error"),
    ],
)
def test_eval_unresolved(obj: DslBase) -> None:
    with pytest.raises(RuntimeError):
        obj.eval(LLMRuntimeContext([], []))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "obj",
    [
        Abort(),
        Ask("question"),
        QueryFill("value"),
        QueryGather("orig", "info"),
        QueryUser("query"),
        AbortWithNewDsl(ListElement([])),
        IntentRuntimeErrorResolver(Intent("foo", []), "error"),
    ],
)
async def test_eval_async_unresolved(obj: DslBase) -> None:
    with pytest.raises(RuntimeError):
        await obj.eval_async(LLMRuntimeContext([], []))


@pytest.mark.asyncio
async def test_same_as_previous_intent_eval_async() -> None:
    with pytest.raises(NotImplementedError):
        await SameAsPreviousIntent().eval_async(runtime_context())
