from unittest.mock import patch
import pytest

# Importing the parser first avoids circular import issues when loading query
# elements. This is needed when running this file on its own.
from fifo_dev_dsl.dia.dsl.parser.parser import parse_dsl   # noqa, pylint: disable=unused-import, # pyright: ignore[reportUnusedImport]
from fifo_dev_dsl.dia.dsl.elements.query_fill import QueryFill
from fifo_dev_dsl.dia.dsl.elements.query_gather import QueryGather
from fifo_dev_dsl.dia.dsl.elements.query_user import QueryUser
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.context import ResolutionContext
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.dia.dsl.elements.value import Value


def _runtime_context() -> LLMRuntimeContext:
    return LLMRuntimeContext([], [])


def test_query_fill_abort_raises() -> None:
    qf = QueryFill("dummy")
    ctx = _runtime_context()
    rc = ResolutionContext()
    with patch(
        "fifo_dev_dsl.dia.dsl.elements.query_fill.call_airlock_model_server",
        return_value="reasoning: r\nvalue: v\nabort: nope",
    ):
        with pytest.raises(RuntimeError, match="QueryFill failed: abort message was returned"):
            qf.do_resolution(ctx, rc, None)


def test_query_fill_unknown_value() -> None:
    qf = QueryFill("dummy")
    ctx = _runtime_context()
    rc = ResolutionContext()
    with patch(
        "fifo_dev_dsl.dia.dsl.elements.query_fill.call_airlock_model_server",
        return_value="unexpected",
    ):
        outcome = qf.do_resolution(ctx, rc, None)
    assert outcome.result is ResolutionResult.NEW_DSL_NODES
    assert outcome.nodes is not None
    assert isinstance(outcome.nodes[0], Value)
    assert outcome.nodes[0].value == "unknown"


def test_query_gather_unknown_value() -> None:
    qg = QueryGather("intent", "question")
    ctx = _runtime_context()
    rc = ResolutionContext()
    mock_outcome = object()
    with patch(
        "fifo_dev_dsl.dia.dsl.elements.query_gather.call_airlock_model_server",
        return_value="bad answer",
    ), patch(
        "fifo_dev_dsl.dia.dsl.elements.query_gather.ask_helper_no_interaction",
        return_value=mock_outcome
    ) as helper:
        outcome = qg.do_resolution(ctx, rc, None)
    helper.assert_called_once()
    assert helper.call_args.args[-1] == "unknown"
    assert outcome is mock_outcome


def test_query_user_unknown_value() -> None:
    qu = QueryUser("question")
    ctx = _runtime_context()
    rc = ResolutionContext()
    with patch(
        "fifo_dev_dsl.dia.dsl.elements.query_user.call_airlock_model_server",
        return_value="irrelevant",
    ):
        outcome = qu.do_resolution(ctx, rc, None)
    assert outcome.result is ResolutionResult.INTERACTION_REQUESTED
    assert outcome.interaction is not None
    assert outcome.interaction.message == "unknown"
