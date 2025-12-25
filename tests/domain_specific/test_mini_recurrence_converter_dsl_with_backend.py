"""Tests for parse_natural_recurrence_expression_with_backend function."""
import pytest
from fifo_dev_dsl.common.llm_abstraction import LlmBackend, LlmRequest
from fifo_dev_dsl.domain_specific.mini_recurrence_converter_dsl.core import (
    parse_natural_recurrence_expression_with_backend,
    parse_natural_recurrence_expression,
    RecurrenceUnit
)


class MockLlmBackend:
    """Mock LLM backend for testing."""
    
    def __init__(self, response: str):
        self.response = response
        self.last_request = None
    
    def complete(self, req: LlmRequest) -> str:
        """Return the configured response."""
        self.last_request = req
        return self.response


def test_parse_natural_recurrence_expression_with_backend_daily():
    """Test basic daily recurrence."""
    backend = MockLlmBackend("DAILY(1)")
    
    dsl_code, rule = parse_natural_recurrence_expression_with_backend(
        "every day",
        backend=backend
    )
    
    assert dsl_code == "DAILY(1)"
    assert rule.unit == RecurrenceUnit.DAILY
    assert rule.frequency == 1


def test_parse_natural_recurrence_expression_with_backend_daily_with_time():
    """Test daily recurrence with time."""
    backend = MockLlmBackend("DAILY(2, TIME(9, 30))")
    
    dsl_code, rule = parse_natural_recurrence_expression_with_backend(
        "every 2 days at 9:30am",
        backend=backend
    )
    
    assert dsl_code == "DAILY(2, TIME(9, 30))"
    assert rule.unit == RecurrenceUnit.DAILY
    assert rule.frequency == 2
    assert rule.hour == 9
    assert rule.minute == 30


def test_parse_natural_recurrence_expression_with_backend_weekly():
    """Test weekly recurrence."""
    backend = MockLlmBackend("WEEKLY(1, [MO, WE], TIME(10, 0))")
    
    dsl_code, rule = parse_natural_recurrence_expression_with_backend(
        "every Monday and Wednesday at 10am",
        backend=backend
    )
    
    assert dsl_code == "WEEKLY(1, [MO, WE], TIME(10, 0))"
    assert rule.unit == RecurrenceUnit.WEEKLY
    assert rule.frequency == 1
    assert rule.days == [0, 2]  # MO=0, WE=2
    assert rule.hour == 10
    assert rule.minute == 0


def test_parse_natural_recurrence_expression_with_backend_monthly():
    """Test monthly recurrence."""
    backend = MockLlmBackend("MONTHLY(1, 15, TIME(8, 30))")
    
    dsl_code, rule = parse_natural_recurrence_expression_with_backend(
        "every month on the 15th at 8:30am",
        backend=backend
    )
    
    assert dsl_code == "MONTHLY(1, 15, TIME(8, 30))"
    assert rule.unit == RecurrenceUnit.MONTHLY
    assert rule.frequency == 1
    assert rule.day_of_month == 15
    assert rule.hour == 8
    assert rule.minute == 30


def test_parse_natural_recurrence_expression_with_backend_monthly_by_weekday():
    """Test monthly by weekday recurrence."""
    backend = MockLlmBackend("MONTHLY_BY_WEEKDAY(1, FR, 1, TIME(17, 0))")
    
    dsl_code, rule = parse_natural_recurrence_expression_with_backend(
        "every first Friday at 5pm",
        backend=backend
    )
    
    assert dsl_code == "MONTHLY_BY_WEEKDAY(1, FR, 1, TIME(17, 0))"
    assert rule.unit == RecurrenceUnit.MONTHLY_BY_WEEKDAY
    assert rule.frequency == 1
    assert rule.days == [4]  # FR=4
    assert rule.occurrence == 1
    assert rule.hour == 17
    assert rule.minute == 0


def test_parse_natural_recurrence_expression_with_backend_yearly():
    """Test yearly recurrence."""
    backend = MockLlmBackend("YEARLY(1, 12, 25, TIME(18, 0))")
    
    dsl_code, rule = parse_natural_recurrence_expression_with_backend(
        "every Christmas at 6pm",
        backend=backend
    )
    
    assert dsl_code == "YEARLY(1, 12, 25, TIME(18, 0))"
    assert rule.unit == RecurrenceUnit.YEARLY
    assert rule.frequency == 1
    assert rule.month == 12
    assert rule.day_of_month == 25
    assert rule.hour == 18
    assert rule.minute == 0


def test_parse_natural_recurrence_expression_with_backend_hourly():
    """Test hourly recurrence."""
    backend = MockLlmBackend("HOURLY(1, 30)")
    
    dsl_code, rule = parse_natural_recurrence_expression_with_backend(
        "every 1 hour and 30 minutes",
        backend=backend
    )
    
    assert dsl_code == "HOURLY(1, 30)"
    assert rule.unit == RecurrenceUnit.HOURLY
    assert rule.frequency == 1
    assert rule.minute == 30


def test_parse_natural_recurrence_expression_with_backend_max_new_tokens():
    """Test that max_new_tokens parameter is passed to LlmRequest."""
    backend = MockLlmBackend("DAILY(1)")
    
    parse_natural_recurrence_expression_with_backend(
        "every day",
        backend=backend,
        max_new_tokens=512
    )
    
    assert backend.last_request.max_new_tokens == 512


def test_parse_natural_recurrence_expression_with_backend_temperature():
    """Test that temperature parameter is passed to LlmRequest."""
    backend = MockLlmBackend("DAILY(1)")
    
    parse_natural_recurrence_expression_with_backend(
        "every day",
        backend=backend,
        temperature=0.5
    )
    
    assert backend.last_request.temperature == 0.5


def test_parse_natural_recurrence_expression_with_backend_invalid_dsl():
    """Test that invalid DSL raises ValueError."""
    backend = MockLlmBackend("INVALID_FUNCTION()")
    
    with pytest.raises(ValueError) as exc_info:
        parse_natural_recurrence_expression_with_backend(
        "some invalid input",
        backend=backend
    )
    
    assert "INVALID_FUNCTION" in str(exc_info.value)


def test_parse_natural_recurrence_expression_with_backend_system_prompt():
    """Test that system prompt is correctly set."""
    backend = MockLlmBackend("DAILY(1)")
    
    parse_natural_recurrence_expression_with_backend(
        "every day",
        backend=backend
    )
    
    # System prompt should mention parser and DSL functions
    assert "parser" in backend.last_request.system_prompt.lower()
    assert "WEEKLY" in backend.last_request.system_prompt
    assert "MONTHLY_BY_WEEKDAY" in backend.last_request.system_prompt


def test_parse_natural_recurrence_expression_with_backend_user_prompt():
    """Test that user prompt is the question."""
    backend = MockLlmBackend("DAILY(1)")
    question = "every Tuesday"
    
    parse_natural_recurrence_expression_with_backend(
        question,
        backend=backend
    )
    
    assert backend.last_request.user_prompt == question


def test_parse_natural_recurrence_expression_with_backend_reasoning_level():
    """Test that reasoning_level parameter is passed to LlmRequest."""
    backend = MockLlmBackend("DAILY(1)")
    
    parse_natural_recurrence_expression_with_backend(
        "every day",
        backend=backend,
        reasoning_level="high"
    )
    
    assert backend.last_request.reasoning_level == "high"


def test_parse_natural_recurrence_expression_with_backend_default_reasoning_level():
    """Test that default reasoning_level is 'low'."""
    backend = MockLlmBackend("DAILY(1)")
    
    parse_natural_recurrence_expression_with_backend(
        "every day",
        backend=backend
    )
    
    assert backend.last_request.reasoning_level == "low"


def test_parse_natural_recurrence_expression_deprecated_warning():
    """Test that the old function emits a deprecation warning."""
    with pytest.warns(DeprecationWarning, match="parse_natural_recurrence_expression is deprecated"):
        # This will fail because we don't have a real backend, but we're just
        # checking for the warning
        try:
            parse_natural_recurrence_expression(
                "every day",
                container_name="test",
                adapter="test-adapter"
            )
        except Exception:
            # Expected to fail without real backend
            pass
