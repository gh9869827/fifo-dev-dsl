import builtins
from unittest.mock import patch

from pytest import CaptureFixture

from fifo_dev_common.introspection.mini_docstring import MiniDocStringType

from fifo_dev_dsl.dia.dsl.parser.parser import parse_dsl, parse_dsl_element
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.interaction import (
    Interaction,
    InteractionRequest,
)
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome
from fifo_dev_dsl.dia.resolution.resolver import Resolver
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext


def test_fully_resolve_in_text_mode(capsys: CaptureFixture[str]):
    runtime_context = LLMRuntimeContext([], [])
    resolver = Resolver(runtime_context=runtime_context, dsl=parse_dsl("[]"))

    requester = parse_dsl_element("foo()", False)
    interaction_req = InteractionRequest(
        message="how many?",
        expected_type=MiniDocStringType(int),
        requester=requester,
        slot=None,
    )

    outcomes = [
        ResolutionOutcome(
            result=ResolutionResult.INTERACTION_REQUESTED,
            interaction=interaction_req,
        ),
        ResolutionOutcome(result=ResolutionResult.UNCHANGED),
    ]
    calls: list[Interaction | None] = []

    def fake_call(reply: Interaction | None) -> ResolutionOutcome:
        calls.append(reply)
        return outcomes.pop(0)

    with patch("fifo_dev_dsl.dia.resolution.resolver.Resolver.__call__", side_effect=fake_call):
        with patch.object(builtins, "input", side_effect=["42"]):
            resolver.fully_resolve_in_text_mode()

    captured = capsys.readouterr().out.strip().splitlines()
    assert captured == ["< how many?"]
    assert calls[0] is None
    assert isinstance(calls[1], Interaction)
    assert calls[1].request is interaction_req
    assert calls[1].answer.content == "42"
