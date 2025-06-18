from typing import Callable
import pytest
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, MO, TH, FR, WE, SU
from fifo_dev_dsl.domain_specific.mini_date_converter_dsl.core import MiniDateConverterDSL

# Helper
def next_month_day(month: int, day: int) -> datetime:
    today = datetime.now().date()
    try_this_year = datetime(today.year, month, day)
    return try_this_year if try_this_year.date() >= today else datetime(today.year + 1, month, day)

def last_day_of_month(dt: datetime) -> int:
    next_month = dt.replace(day=28) + timedelta(days=4)
    return (next_month.replace(day=1) - timedelta(days=1)).day

def next_month_weekday(month: int, weekday_func, occurrence: int) -> datetime:
    today = datetime.now()
    for offset in range(10):
        anchor = datetime(today.year + offset, month, 1)
        if occurrence < 0:
            anchor += relativedelta(months=1, days=-1)
        candidate = anchor + relativedelta(weekday=weekday_func(occurrence))
        if candidate >= today:
            return candidate


@pytest.mark.parametrize(
    "expr, expected_fn",
    [
        # TODAY

        # TODAY returns current date with time set to 00:00
        (
            "TODAY",
            lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ),

        # OFFSET

        # OFFSET by +1 day
        (
            "OFFSET(TODAY, 1, DAY)",
            lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    + timedelta(days=1)
        ),

        # OFFSET by -2 days
        (
            "OFFSET(TODAY, -2, DAY)",
            lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    - timedelta(days=2)
        ),

        # OFFSET by 1 week
        (
            "OFFSET(TODAY, 1, WEEK)",
            lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    + timedelta(weeks=1)
        ),

        # OFFSET by 1 month
        (
            "OFFSET(TODAY, 1, MONTH)",
            lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    + relativedelta(months=1)
        ),

        # OFFSET by 2 year
        (
            "OFFSET(TODAY, 2, YEAR)",
            lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    + relativedelta(years=2)
        ),

        # OFFSET by 2 weeks
        (
            "OFFSET(TODAY, 2, WEEK)",
            lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    + timedelta(weeks=2)
        ),

        # OFFSET by 0 days (no change)
        (
            "OFFSET(TODAY, 0, DAY)",
            lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ),

        # OFFSET to next Monday
        (
            "OFFSET(TODAY, 1, WEEKDAY=0)",
            lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    + timedelta((0 - datetime.now().weekday() + 7) % 7 or 7)
        ),

        # OFFSET to next Sunday
        (
            "OFFSET(TODAY, 1, WEEKDAY=6)",
            lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    + timedelta((6 - datetime.now().weekday() + 7) % 7 or 7)
        ),

        # DATE_FROM_MONTH_DAY

        # DATE_FROM_MONTH_DAY for Jan 1
        (
            "DATE_FROM_MONTH_DAY(1, 1)",
            lambda: next_month_day(1, 1)
        ),

        # DATE_FROM_MONTH_DAY for July 4
        (
            "DATE_FROM_MONTH_DAY(7, 4)",
            lambda: next_month_day(7, 4)
        ),

        # DATE_FROM_MONTH_DAY for Dec 31
        (
            "DATE_FROM_MONTH_DAY(12, 31)",
            lambda: next_month_day(12, 31)
        ),

        # DATE_FROM_MONTH_DAY for Feb 29 – always returns the next valid Feb 29 after today
        (
            "DATE_FROM_MONTH_DAY(2, 29)",
            lambda: next(
                datetime(y, 2, 29)
                for y in range(datetime.now().year, datetime.now().year + 10)
                if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
                and datetime(y, 2, 29).date() >= datetime.now().date()
            )
        ),

        # DATE_FROM_YEAR_MONTH_DAY

        # DATE_FROM_YEAR_MONTH_DAY for Jan 1, 2025
        (
            "DATE_FROM_YEAR_MONTH_DAY(2025, 1, 1)",
            lambda: datetime(2025, 1, 1)
        ),

        # DATE_FROM_YEAR_MONTH_DAY for July 4, 2026
        (
            "DATE_FROM_YEAR_MONTH_DAY(2026, 7, 4)",
            lambda: datetime(2026, 7, 4)
        ),

        # DATE_FROM_YEAR_MONTH_DAY for Dec 31, 2030
        (
            "DATE_FROM_YEAR_MONTH_DAY(2030, 12, 31)",
            lambda: datetime(2030, 12, 31)
        ),

        # DATE_FROM_YEAR_MONTH_DAY for Feb 29, 2024 (leap year)
        (
            "DATE_FROM_YEAR_MONTH_DAY(2024, 2, 29)",
            lambda: datetime(2024, 2, 29)
        ),

        # DATE_FROM_MONTH_WEEKDAY

        # 4th Thursday of November (US Thanksgiving)
        (
            "DATE_FROM_MONTH_WEEKDAY(11, 3, 4)",
            lambda: next_month_weekday(11, TH, 4)
        ),

        # 1st Monday of January
        (
            "DATE_FROM_MONTH_WEEKDAY(1, 0, 1)",
            lambda: next_month_weekday(1, MO, 1)
        ),

        # 3rd Friday of March
        (
            "DATE_FROM_MONTH_WEEKDAY(3, 4, 3)",
            lambda: next_month_weekday(3, FR, 3)
        ),

        # 5th Wednesday of May (may fall late in the month)
        (
            "DATE_FROM_MONTH_WEEKDAY(5, 2, 5)",
            lambda: next_month_weekday(5, WE, 5)
        ),

        # 2nd Sunday of October
        (
            "DATE_FROM_MONTH_WEEKDAY(10, 6, 2)",
            lambda: next_month_weekday(10, SU, 2)
        ),

        # Last Friday of October
        (
            "DATE_FROM_MONTH_WEEKDAY(10, 4, -1)",
            lambda: next_month_weekday(10, FR, -1)
        ),

        # Last Friday of July (next month starts on same weekday)
        (
            "DATE_FROM_MONTH_WEEKDAY(7, 4, -1)",
            lambda: next_month_weekday(7, FR, -1)
        ),

        # DATE_FROM_YEAR_MONTH_WEEKDAY

        # 2nd Monday of February 2026
        (
            "DATE_FROM_YEAR_MONTH_WEEKDAY(2026, 2, 0, 2)",
            lambda: datetime(2026, 2, 1) + relativedelta(weekday=MO(2))
        ),

        # 1st Sunday of January 2030
        (
            "DATE_FROM_YEAR_MONTH_WEEKDAY(2030, 1, 6, 1)",
            lambda: datetime(2030, 1, 1) + relativedelta(weekday=SU(1))
        ),

        # 3rd Friday of March 2025
        (
            "DATE_FROM_YEAR_MONTH_WEEKDAY(2025, 3, 4, 3)",
            lambda: datetime(2025, 3, 1) + relativedelta(weekday=FR(3))
        ),

        # 5th Wednesday of May 2027
        (
            "DATE_FROM_YEAR_MONTH_WEEKDAY(2027, 5, 2, 5)",
            lambda: datetime(2027, 5, 1) + relativedelta(weekday=WE(5))
        ),

        # 4th Thursday of November 2028 (Thanksgiving)
        (
            "DATE_FROM_YEAR_MONTH_WEEKDAY(2028, 11, 3, 4)",
            lambda: datetime(2028, 11, 1) + relativedelta(weekday=TH(4))
        ),

        # Last Friday of October 2026
        (
            "DATE_FROM_YEAR_MONTH_WEEKDAY(2026, 10, 4, -1)",
            lambda: datetime(2026, 10, 1)
                    + relativedelta(months=1, weekday=FR(-1))
        ),

        # Last Friday of July 2025 (next month starts on same weekday)
        (
            "DATE_FROM_YEAR_MONTH_WEEKDAY(2025, 7, 4, -1)",
            lambda: datetime(2025, 7, 1) + relativedelta(months=1, days=-1, weekday=FR(-1))
        ),

        # SET_MONTH_DAY

        # First day of next month
        (
            "SET_MONTH_DAY(OFFSET(TODAY, 1, MONTH), 1)",
            lambda: (
                datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                + relativedelta(months=1)
            ).replace(day=1)
        ),

        # Last day of this month
        (
            "SET_MONTH_DAY(TODAY, -1)",
            lambda: datetime.now().replace(day=last_day_of_month(datetime.now()), hour=0, minute=0, second=0, microsecond=0)
        ),

        # SET_TIME and Composition

        # Set time on today to 9:30
        (
            "SET_TIME(TODAY, 9, 30)",
            lambda: datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        ),

        # Set time on today to midnight
        (
            "SET_TIME(TODAY, 0, 0)",
            lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ),

        # Set time on OFFSET(TODAY, 1, DAY) to 23:59
        (
            "SET_TIME(OFFSET(TODAY, 1, DAY), 23, 59)",
            lambda: (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                     + timedelta(days=1)).replace(hour=23, minute=59)
        ),

        # Set time on a fixed date
        (
            "SET_TIME(DATE_FROM_YEAR_MONTH_DAY(2025, 1, 1), 6, 45)",
            lambda: datetime(2025, 1, 1, 6, 45)
        ),

        # OFFSET_TIME and composition

        # Add 1 hour and 30 minutes to today at 00:00
        (
            "OFFSET_TIME(TODAY, 1, 30)",
            lambda: datetime.now()
                    + timedelta(hours=1, minutes=30)
        ),

        # Subtract 2 hours and 15 minutes from today at 00:00
        (
            "OFFSET_TIME(TODAY, -2, -15)",
            lambda: datetime.now()
                    - timedelta(hours=2, minutes=15)
        ),

        # Add 30 minutes to today at 12:00
        (
            "OFFSET_TIME(SET_TIME(TODAY, 12, 0), 0, 30)",
            lambda: datetime.now().replace(hour=12, minute=30, second=0, microsecond=0)
        ),

        # Add 30 minutes to today at 23:45 (crosses midnight)
        (
            "OFFSET_TIME(SET_TIME(TODAY, 23, 45), 0, 30)",
            lambda: datetime.now().replace(hour=23, minute=45, second=0, microsecond=0)
                    + timedelta(minutes=30)
        ),

        # Add 1 hour to tomorrow
        (
            "OFFSET_TIME(OFFSET(TODAY, 1, DAY), 1, 0)",
            lambda: datetime.now()
                    + timedelta(days=1, hours=1)
        ),

    ]
)
def test_today_expression(expr: str, expected_fn: Callable[[], datetime]):
    result = MiniDateConverterDSL().parse(expr)

    # Allow for minor timing differences between expected and actual datetime evaluation
    assert abs(result - expected_fn()) < timedelta(milliseconds=100)


def test_nested_set_time_offset_weekday():
    dsl = MiniDateConverterDSL(datetime(2025, 1, 1))
    result = dsl.parse(
        "SET_TIME(OFFSET(DATE_FROM_MONTH_WEEKDAY(11, 3, 4), 1, WEEKDAY=0), 9, 30)"
    )
    # Thanksgiving 2025 is Thursday, Nov 27 → next Monday is Dec 1
    assert result == datetime(2025, 12, 1, 9, 30)


def test_date_from_month_weekday_future_rollover():
    # When the target this year is in the past, DATE_FROM_MONTH_WEEKDAY should
    # roll over to the next year.
    dsl = MiniDateConverterDSL(datetime(2025, 5, 1))
    result = dsl.parse("DATE_FROM_MONTH_WEEKDAY(3, 4, -1)")
    assert result == datetime(2026, 3, 27)


@pytest.mark.parametrize(
    "expr, expected_message",
    [
        ("TODAY(1)", r"TODAY takes no arguments"),
        ("offset(TODAY, 1, DAY)", r"DSL function names must be uppercase"),
        ("OFFSET(TODAY, 1, DUMMY)", r"Unknown unit in OFFSET: DUMMY"),
        (
            "OFFSET(DATE_FROM_YEAR_MONTH_DAY(2026, 1, 1), 4, WEEK, 5, DAY)",
            r"OFFSET requires exactly 3 arguments",
        ),
        ("DUMMY(1, 41)", r"Unknown function: DUMMY"),

        # DATE_FROM_MONTH_DAY
        ("DATE_FROM_MONTH_DAY(1, 1, 1)", r"DATE_FROM_MONTH_DAY requires exactly 2 arguments"),
        ("DATE_FROM_MONTH_DAY()", r"Invalid or missing month in DATE_FROM_MONTH_DAY: got 'None'"),
        ("DATE_FROM_MONTH_DAY(error)", r"Invalid or missing month in DATE_FROM_MONTH_DAY: got 'error'"),
        ("DATE_FROM_MONTH_DAY(13)", r"Month 13 is out of range in DATE_FROM_MONTH_DAY (expected 1-12)"),
        ("DATE_FROM_MONTH_DAY(1)", r"Invalid or missing day in DATE_FROM_MONTH_DAY: got 'None'"),
        ("DATE_FROM_MONTH_DAY(12, error)", r"Invalid or missing day in DATE_FROM_MONTH_DAY: got 'error'"),
        ("DATE_FROM_MONTH_DAY(2, 30)", r"DATE_FROM_MONTH_DAY(2, 30) is invalid"),
        ("DATE_FROM_MONTH_DAY(4, 31)", r"DATE_FROM_MONTH_DAY(4, 31) is invalid"),
        ("DATE_FROM_MONTH_DAY(11, 31)", r"DATE_FROM_MONTH_DAY(11, 31) is invalid"),

        # DATE_FROM_YEAR_MONTH_DAY
        ("DATE_FROM_YEAR_MONTH_DAY(2025, 1, 1, 0)", r"DATE_FROM_YEAR_MONTH_DAY requires exactly 3 arguments"),
        ("DATE_FROM_YEAR_MONTH_DAY()", r"Invalid or missing year in DATE_FROM_YEAR_MONTH_DAY: got 'None'"),
        ("DATE_FROM_YEAR_MONTH_DAY(error)", r"Invalid or missing year in DATE_FROM_YEAR_MONTH_DAY: got 'error'"),
        ("DATE_FROM_YEAR_MONTH_DAY(2025)", r"Invalid or missing month in DATE_FROM_YEAR_MONTH_DAY: got 'None'"),
        ("DATE_FROM_YEAR_MONTH_DAY(2025, error)", r"Invalid or missing month in DATE_FROM_YEAR_MONTH_DAY: got 'error'"),
        ("DATE_FROM_YEAR_MONTH_DAY(2025, 14)", r"Month 14 is out of range in DATE_FROM_YEAR_MONTH_DAY (expected 1-12)"),
        ("DATE_FROM_YEAR_MONTH_DAY(2025, 12)", r"Invalid or missing day in DATE_FROM_YEAR_MONTH_DAY: got 'None'"),
        ("DATE_FROM_YEAR_MONTH_DAY(2025, 12, error)", r"Invalid or missing day in DATE_FROM_YEAR_MONTH_DAY: got 'error'"),
        ("DATE_FROM_YEAR_MONTH_DAY(2025, 11, 31)", r"DATE_FROM_YEAR_MONTH_DAY(2025, 11, 31) is invalid"),

        # DATE_FROM_MONTH_WEEKDAY
        ("DATE_FROM_MONTH_WEEKDAY(11, 1, 2, 0)", r"DATE_FROM_MONTH_WEEKDAY requires exactly 3 arguments"),
        ("DATE_FROM_MONTH_WEEKDAY()", r"Invalid or missing month in DATE_FROM_MONTH_WEEKDAY: got 'None'"),
        ("DATE_FROM_MONTH_WEEKDAY(error)", r"Invalid or missing month in DATE_FROM_MONTH_WEEKDAY: got 'error'"),
        ("DATE_FROM_MONTH_WEEKDAY(14)", r"Month 14 is out of range in DATE_FROM_MONTH_WEEKDAY (expected 1-12)"),
        ("DATE_FROM_MONTH_WEEKDAY(11, error)", r"Invalid or missing weekday in DATE_FROM_MONTH_WEEKDAY: got 'error'"),
        ("DATE_FROM_MONTH_WEEKDAY(11, 7)", r"Invalid weekday in DATE_FROM_MONTH_WEEKDAY: got 7"),
        ("DATE_FROM_MONTH_WEEKDAY(11, 4)", r"Invalid or missing occurrence in DATE_FROM_MONTH_WEEKDAY: got 'None'"),
        ("DATE_FROM_MONTH_WEEKDAY(11, 4, error)", r"Invalid or missing occurrence in DATE_FROM_MONTH_WEEKDAY: got 'error'"), 

        # DATE_FROM_YEAR_MONTH_WEEKDAY
        ("DATE_FROM_YEAR_MONTH_WEEKDAY(2025, 11, 1, 2, 0)", r"DATE_FROM_YEAR_MONTH_WEEKDAY requires exactly 4 arguments"),
        ("DATE_FROM_YEAR_MONTH_WEEKDAY()", r"Invalid or missing year in DATE_FROM_YEAR_MONTH_WEEKDAY: got 'None'"),
        ("DATE_FROM_YEAR_MONTH_WEEKDAY(error)", r"Invalid or missing year in DATE_FROM_YEAR_MONTH_WEEKDAY: got 'error'"),
        ("DATE_FROM_YEAR_MONTH_WEEKDAY(2025)", r"Invalid or missing month in DATE_FROM_YEAR_MONTH_WEEKDAY: got 'None'"),
        ("DATE_FROM_YEAR_MONTH_WEEKDAY(2025, error)", r"Invalid or missing month in DATE_FROM_YEAR_MONTH_WEEKDAY: got 'error'"),
        ("DATE_FROM_YEAR_MONTH_WEEKDAY(2025, 14)", r"Month 14 is out of range in DATE_FROM_YEAR_MONTH_WEEKDAY (expected 1-12)"),
        ("DATE_FROM_YEAR_MONTH_WEEKDAY(2025, 11)", r"Invalid or missing weekday in DATE_FROM_YEAR_MONTH_WEEKDAY: got 'None'"),
        ("DATE_FROM_YEAR_MONTH_WEEKDAY(2025, 11, error)", r"Invalid or missing weekday in DATE_FROM_YEAR_MONTH_WEEKDAY: got 'error'"),
        ("DATE_FROM_YEAR_MONTH_WEEKDAY(2025, 11, 7)", r"Invalid weekday in DATE_FROM_YEAR_MONTH_WEEKDAY: got 7"),
        ("DATE_FROM_YEAR_MONTH_WEEKDAY(2025, 11, 4)", r"Invalid or missing occurrence in DATE_FROM_YEAR_MONTH_WEEKDAY: got 'None'"),
        ("DATE_FROM_YEAR_MONTH_WEEKDAY(2025, 11, 4, error)", r"Invalid or missing occurrence in DATE_FROM_YEAR_MONTH_WEEKDAY: got 'error'"),

        # SET_MONTH_DAY
        ("SET_MONTH_DAY(TODAY, 1, 1)", r"SET_MONTH_DAY requires exactly 2 arguments"),
        ("SET_MONTH_DAY()", r"Invalid or missing base expression in SET_MONTH_DAY: got 'None'"),
        ("SET_MONTH_DAY(error, 1)", r"Invalid or missing base expression in SET_MONTH_DAY: got 'error'"),
        ("SET_MONTH_DAY(TODAY)", r"Invalid or missing day in SET_MONTH_DAY: got 'None'"),
        ("SET_MONTH_DAY(TODAY, error)", r"Invalid or missing day in SET_MONTH_DAY: got 'error'"),
        ("SET_MONTH_DAY(TODAY, 0)", r"SET_MONTH_DAY(0) is invalid"),
        ("SET_MONTH_DAY(TODAY, 32)", r"SET_MONTH_DAY(32) is invalid"),
        ("SET_MONTH_DAY(TODAY, -32)", r"SET_MONTH_DAY(-32) is invalid"),

        # SET_TIME
        ("SET_TIME(TODAY, 10, 30, 1)", r"SET_TIME requires exactly 3 arguments"),
        ("SET_TIME()", r"Invalid or missing base expression in SET_TIME: got 'None'"),
        ("SET_TIME(error)", r"Invalid or missing base expression in SET_TIME: got 'error'"),
        ("SET_TIME(TODAY)", r"Invalid or missing hour in SET_TIME: got 'None'"),
        ("SET_TIME(TODAY, error)", r"Invalid or missing hour in SET_TIME: got 'error'"),
        ("SET_TIME(TODAY, 24)", r"Hour 24 is out of range in SET_TIME (expected 0-23)"),
        ("SET_TIME(TODAY, 10)", r"Invalid or missing minute in SET_TIME: got 'None'"),
        ("SET_TIME(TODAY, 10, error)", r"Invalid or missing minute in SET_TIME: got 'error'"),
        ("SET_TIME(TODAY, 10, 60)", r"Minute 60 is out of range in SET_TIME (expected 0-59)"),

        # OFFSET_TIME
        ("OFFSET_TIME(TODAY, 1, 30, 1)", r"OFFSET_TIME requires exactly 3 arguments"),
        ("OFFSET_TIME()", r"Invalid or missing base expression in OFFSET_TIME: got 'None'"),
        ("OFFSET_TIME(error)", r"Invalid or missing base expression in OFFSET_TIME: got 'error'"),
        ("OFFSET_TIME(TODAY)", r"Invalid or missing hour in OFFSET_TIME: got 'None'"),
        ("OFFSET_TIME(TODAY, error)", r"Invalid or missing hour in OFFSET_TIME: got 'error'"),
        ("OFFSET_TIME(TODAY, 1)", r"Invalid or missing minute in OFFSET_TIME: got 'None'"),
        ("OFFSET_TIME(TODAY, 1, error)", r"Invalid or missing minute in OFFSET_TIME: got 'error'"),
        ("OFFSET(DATE_FROM_MONTH_WEEKDAY(6, 3, 2), 2, WEEK) 2 WEEKDAY",
         r"Invalid expression"),
    ]
)
def test_invalid_dsl_expressions(expr: str, expected_message: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        MiniDateConverterDSL().parse(expr)

    assert expected_message in str(exc_info.value)
