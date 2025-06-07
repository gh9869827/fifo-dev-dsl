import pytest

from fifo_dev_dsl.dia.resolution.context import ResolutionContext
from fifo_dev_dsl.dia.dsl.elements.ask import Ask
from fifo_dev_dsl.dia.resolution.llm_call_log import LLMCallLog


def test_format_previous_qna_block_empty() -> None:
    ctx = ResolutionContext()

    assert ctx.format_previous_qna_block() == "  previous_questions_and_answers: []"


def test_format_previous_qna_block_multiple() -> None:
    ctx = ResolutionContext(
        questions_being_clarified=[
            (Ask("q1"), "q1", "a1"),
            (Ask("q2"), "q2", "a2"),
        ]
    )

    expected = (
        "  previous_questions_and_answers:\n"
        "    - question: q1\n"
        "      answer: a1\n"
        "    - question: q2\n"
        "      answer: a2"
    )

    assert ctx.format_previous_qna_block() == expected


def test_format_call_log_empty() -> None:
    ctx = ResolutionContext()

    assert ctx.format_call_log() == ""


def test_format_call_log_multiple() -> None:
    ctx = ResolutionContext(
        llm_call_logs=[
            LLMCallLog("d1", "sys1", "asst1", "ans1"),
            LLMCallLog("d2", "sys2", "asst2", "ans2"),
        ]
    )

    expected = (
        "---\n$\nsys1\n>\nasst1\n<\nans1\n---\n$\nsys2\n>\nasst2\n<\nans2\n---"
    )

    assert ctx.format_call_log() == expected


def test_format_other_slots_yaml_empty_none() -> None:
    ctx = ResolutionContext()

    assert ctx.format_other_slots_yaml() == "other_slots: {}"


def test_format_other_slots_yaml_empty_dict() -> None:
    ctx = ResolutionContext(other_slots={})

    assert ctx.format_other_slots_yaml() == "other_slots: {}"


def test_format_other_slots_yaml_contents_and_padding() -> None:
    ctx = ResolutionContext(other_slots={"foo": "bar", "baz": "qux"})

    assert ctx.format_other_slots_yaml() == (
        "other_slots:\n  foo: bar\n  baz: qux"
    )

    padded = ctx.format_other_slots_yaml("  ")

    assert padded == (
        "  other_slots:\n    foo: bar\n    baz: qux"
    )
