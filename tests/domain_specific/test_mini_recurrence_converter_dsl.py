import pytest
from datetime import datetime
from fifo_dev_dsl.domain_specific.mini_recurrence_converter_dsl.core import MiniRecurrenceConverterDSL, RecurrenceRule, RecurrenceUnit

@pytest.mark.parametrize(
    "expr,expected",
    [
        # DAILY

        # DAILY with required frequency only
        (
            "DAILY(1)",
            RecurrenceRule(
                unit=RecurrenceUnit.DAILY,
                frequency=1
            )
        ),

        # DAILY with frequency and TIME
        (
            "DAILY(2, TIME(9, 30))",
            RecurrenceRule(
                unit=RecurrenceUnit.DAILY,
                frequency=2,
                hour=9,
                minute=30
            )
        ),

        # DAILY with TIME set to midnight (explicit)
        (
            "DAILY(3, TIME(0, 0))",
            RecurrenceRule(
                unit=RecurrenceUnit.DAILY,
                frequency=3,
                hour=0,
                minute=0
            )
        ),

        # WEEKLY

        # WEEKLY with frequency, days, and time
        (
            "WEEKLY(1, [MO, WE], TIME(8, 0))",
            RecurrenceRule(
                unit=RecurrenceUnit.WEEKLY,
                frequency=1,
                days=[0, 2],
                hour=8,
                minute=0
            )
        ),

        # WEEKLY with frequency only
        (
            "WEEKLY(2)",
            RecurrenceRule(
                unit=RecurrenceUnit.WEEKLY,
                frequency=2
            )
        ),

        # WEEKLY with frequency and days only
        (
            "WEEKLY(3, [FR])",
            RecurrenceRule(
                unit=RecurrenceUnit.WEEKLY,
                frequency=3,
                days=[4]
            )
        ),

        # WEEKLY with frequency and time only
        (
            "WEEKLY(1, TIME(10, 15))",
            RecurrenceRule(
                unit=RecurrenceUnit.WEEKLY,
                frequency=1,
                hour=10,
                minute=15
            )
        ),

        # WEEKLY with frequency, days, and time (single day)
        (
            "WEEKLY(1, [TU], TIME(6, 30))",
            RecurrenceRule(
                unit=RecurrenceUnit.WEEKLY,
                frequency=1,
                days=[1],
                hour=6,
                minute=30
            )
        ),

        # MONTHLY

        # MONTHLY with frequency, specific day of month, and time
        (
            "MONTHLY(1, 15, TIME(8, 30))",
            RecurrenceRule(
                unit=RecurrenceUnit.MONTHLY,
                frequency=1,
                day_of_month=15,
                hour=8,
                minute=30
            )
        ),

        # MONTHLY with frequency and specific day of month (no time specified)
        (
            "MONTHLY(2, 1)",
            RecurrenceRule(
                unit=RecurrenceUnit.MONTHLY,
                frequency=2,
                day_of_month=1
            )
        ),

        # MONTHLY with frequency, last day of the month, and time
        (
            "MONTHLY(3, -1, TIME(23, 59))",
            RecurrenceRule(
                unit=RecurrenceUnit.MONTHLY,
                frequency=3,
                day_of_month=-1,
                hour=23,
                minute=59
            )
        ),

        # MONTHLY with frequency only (defaults to day 1 at 00:00)
        (
            "MONTHLY(1)",
            RecurrenceRule(
                unit=RecurrenceUnit.MONTHLY,
                frequency=1
            )
        ),

        # MONTHLY_BY_WEEKDAY

        # MONTHLY_BY_WEEKDAY with weekday+occurrence only
        (
            "MONTHLY_BY_WEEKDAY(3, WE, 2)",
            RecurrenceRule(
                unit=RecurrenceUnit.MONTHLY_BY_WEEKDAY,
                frequency=3,
                days=[2],
                occurrence=2
            )
        ),

        # MONTHLY_BY_WEEKDAY with weekday+occurrence and time
        (
            "MONTHLY_BY_WEEKDAY(1, FR, 1, TIME(17, 0))",
            RecurrenceRule(
                unit=RecurrenceUnit.MONTHLY_BY_WEEKDAY,
                frequency=1,
                days=[4],
                occurrence=1,
                hour=17,
                minute=0
            )
        ),

        # MONTHLY_BY_WEEKDAY with single digit weekday occurrence
        (
            "MONTHLY_BY_WEEKDAY(4, MO, 5, TIME(23, 59))",
            RecurrenceRule(
                unit=RecurrenceUnit.MONTHLY_BY_WEEKDAY,
                frequency=4,
                days=[0],
                occurrence=5,
                hour=23,
                minute=59
            )
        ),

        # YEARLY

        # YEARLY with frequency only
        (
            "YEARLY(1)",
            RecurrenceRule(
                unit=RecurrenceUnit.YEARLY,
                frequency=1
            )
        ),

        # YEARLY with frequency, month, and day
        (
            "YEARLY(1, 7, 4)",
            RecurrenceRule(
                unit=RecurrenceUnit.YEARLY,
                frequency=1,
                month=7,
                day_of_month=4
            )
        ),

        # YEARLY with all components including time
        (
            "YEARLY(3, 11, 15, TIME(14, 30))",
            RecurrenceRule(
                unit=RecurrenceUnit.YEARLY,
                frequency=3,
                month=11,
                day_of_month=15,
                hour=14,
                minute=30
            )
        ),

        # HOURLY

        # HOURLY: every 1 hour and 30 minutes
        (
            "HOURLY(1, 30)",
            RecurrenceRule(
                unit=RecurrenceUnit.HOURLY,
                frequency=1,
                minute=30
            )
        ),

        # HOURLY: every 0 hours and 20 minutes
        (
            "HOURLY(0, 20)",
            RecurrenceRule(
                unit=RecurrenceUnit.HOURLY,
                frequency=0,
                minute=20
            )
        ),

        # HOURLY: every 2 hours and 0 minutes
        (
            "HOURLY(2, 0)",
            RecurrenceRule(
                unit=RecurrenceUnit.HOURLY,
                frequency=2,
                minute=0
            )
        ),
    ]
)
def test_parse_valid_recurrence(expr: str, expected: RecurrenceRule):
    result = MiniRecurrenceConverterDSL().parse(expr)
    assert result == expected


@pytest.mark.parametrize(
    "expr, expected_exception",
    [
        # DAILY: missing required frequency
        ("DAILY()", ValueError),

        # WEEKLY: malformed weekday list
        ("WEEKLY(1, [MO, WE]", ValueError),
        ("WEEKLY(1, MO, TIME(8, 0))", ValueError),  # not a list
        ("WEEKLY(x, [MO])", ValueError),            # non-numeric frequency

        # MONTHLY: invalid day_of_month
        ("MONTHLY(1, x)", ValueError),
        ("MONTHLY()", ValueError),

        # MONTHLY_BY_WEEKDAY: invalid weekday string
        ("MONTHLY_BY_WEEKDAY(1, XX, 2)", ValueError),
        ("MONTHLY_BY_WEEKDAY(1, MO)", ValueError),  # missing occurrence
        ("MONTHLY_BY_WEEKDAY()", ValueError),

        # YEARLY: invalid month or day
        ("YEARLY(1, x, 10)", ValueError),
        ("YEARLY(1, 3, y)", ValueError),

        # HOURLY: non-integer args
        ("HOURLY(abc, def)", ValueError),
        ("HOURLY(1)", ValueError),            # missing minute arg
        ("HOURLY()", ValueError),             # missing both

        # Unknown function
        ("FOO(1, 2)", ValueError),

        # General syntax errors
        ("OFFSET(TODAY, , DAY)", ValueError),   # trailing comma
        ("a(b, c", ValueError),                 # unbalanced parentheses
        ("WEEKLY(1, [MO, WE], TIME())", ValueError),  # malformed TIME
    ]
)
def test_parse_invalid_recurrence(expr: str, expected_exception: type[Exception]):
    with pytest.raises(expected_exception):
        MiniRecurrenceConverterDSL().parse(expr)


@pytest.mark.parametrize(
    "rule,dt,expected",
    [
        # DAILY every 1 day at 00:00
        (
            RecurrenceRule(unit=RecurrenceUnit.DAILY, frequency=1),
            datetime(2025, 1, 1, 15, 30),
            datetime(2025, 1, 2, 0, 0)
        ),

        # WEEKLY every 1 week without specific days
        (
            RecurrenceRule(unit=RecurrenceUnit.WEEKLY, frequency=1),
            datetime(2025, 1, 1, 10, 0),  # Wednesday
            datetime(2025, 1, 8, 0, 0)
        ),

        # WEEKLY every 1 week on [MO, WE] (next is WE = same day, so 1 week later)
        (
            RecurrenceRule(unit=RecurrenceUnit.WEEKLY, frequency=1, days=[0, 2]),
            datetime(2025, 1, 1),  # Wed
            datetime(2025, 1, 6, 0, 0)  # next Monday
        ),

        # MONTHLY on the 15th
        (
            RecurrenceRule(unit=RecurrenceUnit.MONTHLY, frequency=1, day_of_month=15),
            datetime(2025, 1, 10),
            datetime(2025, 2, 15, 0, 0)
        ),

        # MONTHLY on last day (-1)
        (
            RecurrenceRule(unit=RecurrenceUnit.MONTHLY, frequency=1, day_of_month=-1),
            datetime(2025, 1, 30),
            datetime(2025, 2, 28, 0, 0)
        ),

        # YEARLY on July 4
        (
            RecurrenceRule(unit=RecurrenceUnit.YEARLY, frequency=1, month=7, day_of_month=4),
            datetime(2025, 1, 1),
            datetime(2026, 7, 4, 0, 0)
        ),

        # YEARLY clamped day (e.g., Feb 30 -> Feb 28/29)
        (
            RecurrenceRule(unit=RecurrenceUnit.YEARLY, frequency=1, month=2, day_of_month=30),
            datetime(2025, 1, 1),
            datetime(2026, 2, 28, 0, 0)  # 2026 is not leap
        ),

        # HOURLY every 1h30m
        (
            RecurrenceRule(unit=RecurrenceUnit.HOURLY, frequency=1, minute=30),
            datetime(2025, 1, 1, 10, 0),
            datetime(2025, 1, 1, 11, 30)
        ),
    ]
)
def test_recurrence_next(rule: RecurrenceRule, dt: datetime, expected: datetime):
    assert rule.next(dt) == expected


@pytest.mark.parametrize(
    "expr",
    [
        # Keyword arguments (not allowed, should use positional only)
        "DAILY(frequency=1)",
        "WEEKLY(1, days=[MO, WE])",
        "MONTHLY(1, day_of_month=15)",

        # Misnamed functions
        "daily(1)",       # Case sensitivity
        "WeekLy(1, [MO])",

        # Invalid DSL structures
        "DAILY(1, TIME(hour=9, minute=0))",    # Wrong TIME syntax
        "WEEKLY(1, [MO], TIME=9, 0)",          # Assignment in wrong place

        # Non-callable DSL
        "DAILY",           # Missing parentheses
        "TIME(9)",         # Missing second arg

        # Function-like garbage
        "INVALID(1, 2)",
        "DAILY(1, , TIME(9, 0))",              # Extra comma
        "WEEKLY(, [MO])",                      # Missing first arg
        "DAILY(1, [MO])",                      # Invalid second arg
    ]
)
def test_invalid_dsl_syntax(expr: str):
    with pytest.raises((ValueError, KeyError, IndexError)):
        MiniRecurrenceConverterDSL().parse(expr)



@pytest.mark.parametrize(
    "expr, expected_message",
    [
        # Lowercase function name
        ("dummy(1)", "DSL function names must be uppercase"),

        # unknown function
        ("DUMMY(1)", "Unknown recurrence function: DUMMY"),

        # DAILY
        ("DAILY()", "Invalid or missing frequency in DAILY: got 'None'"),
        ("DAILY(dummy)", "Invalid or missing frequency in DAILY: got 'dummy'"),
        ("DAILY(0)", "Frequency 0 is out of range in DAILY (must be positive)"),
        ("DAILY(-2)", "Frequency -2 is out of range in DAILY (must be positive)"),
        ("DAILY(1, 10:00)", "Invalid TIME argument: '10:00'"),
        ("DAILY(1, TIME10,00)", "Invalid TIME argument: 'TIME10'"),
        ("DAILY(1, TIME(24, 0))", "Hour 24 is out of range in TIME (expected 0-23)"),
        ("DAILY(1, TIME(10, 60))", "Minute 60 is out of range in TIME (expected 0-59)"),
        ("DAILY(1, TIME(dummy, 0))", "Invalid or missing hour in TIME: got 'dummy'"),
        ("DAILY(1, TIME(12))", "Invalid or missing minute in TIME: got 'None'"),

        # WEEKLY
        ("WEEKLY()", "Invalid or missing frequency in WEEKLY: got 'None'"),
        ("WEEKLY(dummy)", "Invalid or missing frequency in WEEKLY: got 'dummy'"),
        ("WEEKLY(0)", "Frequency 0 is out of range in WEEKLY (must be positive)"),
        ("WEEKLY(1, 10:00)", "Second argument to WEEKLY must be [days] or TIME(...), got '10:00'"),
        ("WEEKLY(1, [INVALID])", "Invalid day list in WEEKLY: [INVALID]"),
        ("WEEKLY(1, TIME(25, 0))", "Hour 25 is out of range in TIME (expected 0-23)"),
        ("WEEKLY(1, [MO], TIME(8))", "Invalid or missing minute in TIME: got 'None'"),
        ("WEEKLY(1, [MO], TIME(8, 61))", "Minute 61 is out of range in TIME (expected 0-59)"),
        ("WEEKLY(1, [MO], TIME(dummy, 0))", "Invalid or missing hour in TIME: got 'dummy'"),
        ("WEEKLY(1, [MO], 10:00)", "Invalid TIME argument: '10:00'"),
        ("WEEKLY(1, [MO], TIME10,00)", "Too many arguments passed to WEEKLY"),
        ("WEEKLY(1, [MO], TIME(10, 00), extra)", "Too many arguments passed to WEEKLY"),

        # MONTHLY
        ("MONTHLY()", "Invalid or missing frequency in MONTHLY: got 'None'"),
        ("MONTHLY(dummy)", "Invalid or missing frequency in MONTHLY: got 'dummy'"),
        ("MONTHLY(0)", "Frequency 0 is out of range in MONTHLY (must be positive)"),
        ("MONTHLY(1, dummy)", "Invalid or missing day_of_month in MONTHLY: got 'dummy'"),
        ("MONTHLY(1, 32)", "day_of_month 32 is out of range in MONTHLY (expected 1-31 or -1 to -31)"),
        ("MONTHLY(1, -33)", "day_of_month -33 is out of range in MONTHLY (expected 1-31 or -1 to -31)"),
        ("MONTHLY(1, 5, TIME(24, 0))", "Hour 24 is out of range in TIME (expected 0-23)"),
        ("MONTHLY(1, 5, TIME(10, 60))", "Minute 60 is out of range in TIME (expected 0-59)"),
        ("MONTHLY(1, 5, TIME(10, 60), extra)", "Too many arguments passed to MONTHLY"),

        # MONTHLY_BY_WEEKDAY
        ("MONTHLY_BY_WEEKDAY(3)", "MONTHLY_BY_WEEKDAY requires a weekday and occurrence as second and third arguments"),
        ("MONTHLY_BY_WEEKDAY(dummy)", "Invalid or missing frequency in MONTHLY_BY_WEEKDAY: got 'dummy'"),
        ("MONTHLY_BY_WEEKDAY(0)", "Frequency 0 is out of range in MONTHLY_BY_WEEKDAY (must be positive)"),
        ("MONTHLY_BY_WEEKDAY(1, INVALID, 2)", "Invalid weekday in MONTHLY_BY_WEEKDAY: 'INVALID'"),
        ("MONTHLY_BY_WEEKDAY(1, MO, dummy)", "Invalid or missing occurrence in MONTHLY_BY_WEEKDAY: got 'dummy'"),
        ("MONTHLY_BY_WEEKDAY(1, MO, 2, TIME(25, 0))", "Hour 25 is out of range in TIME (expected 0-23)"),
        ("MONTHLY_BY_WEEKDAY(1, MO, 2, TIME(10))", "Invalid or missing minute in TIME: got 'None'"),
        ("MONTHLY_BY_WEEKDAY(1, MO, 2, TIME(10, 60))", "Minute 60 is out of range in TIME (expected 0-59)"),
        ("MONTHLY_BY_WEEKDAY(1, MO, 2, TIME(dummy, 0))", "Invalid or missing hour in TIME: got 'dummy'"),
        ("MONTHLY_BY_WEEKDAY(1, MO, 2, 10:00)", "Invalid TIME argument: '10:00'"),

        # YEARLY
        ("YEARLY()", "Invalid or missing frequency in YEARLY: got 'None'"),
        ("YEARLY(dummy)", "Invalid or missing frequency in YEARLY: got 'dummy'"),
        ("YEARLY(0)", "Frequency 0 is out of range in YEARLY (must be positive)"),
        ("YEARLY(1, 12)", "YEARLY requires month and day as second and third arguments"),
        ("YEARLY(1, dummy)", "YEARLY requires month and day as second and third arguments"),
        ("YEARLY(1, 13)", "YEARLY requires month and day as second and third arguments"),
        ("YEARLY(1, 1, dummy)", "Invalid or missing day in YEARLY: got 'dummy'"),
        ("YEARLY(1, 1, 32)", "Day 32 is out of range in YEARLY (expected 1-31)"),
        ("YEARLY(1, 1, 1, 10:00)", "Invalid TIME argument: '10:00'"),
        ("YEARLY(1, 1, 1, TIME(10,00), extra)", "Too many arguments passed to YEARLY"),

        #HOURLY
        ("HOURLY()", "HOURLY requires exactly 2 arguments: frequency_hour, frequency_minute"),
        ("HOURLY(1)", "HOURLY requires exactly 2 arguments: frequency_hour, frequency_minute"),
        ("HOURLY(dummy, 15)", "Invalid or missing hour in HOURLY: got 'dummy'"),
        ("HOURLY(1, dummy)", "Invalid or missing minute in HOURLY: got 'dummy'"),
        ("HOURLY(-1, 10)", "Hour in HOURLY must be zero or positive"),
        ("HOURLY(1, -1)", "Minute in HOURLY must be zero or positive"),
        ("HOURLY(0, 0)", "HOURLY frequency must be greater than zero"),
    ]
)
def test_invalid_daily_recurrence(expr: str, expected_message: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        MiniRecurrenceConverterDSL().parse(expr)

    assert expected_message in str(exc_info.value)