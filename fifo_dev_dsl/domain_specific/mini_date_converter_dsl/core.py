from __future__ import annotations
from datetime import datetime, timedelta
from typing import Tuple
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU
from fifo_tool_airlock_model_env.common.models import (
    GenerationParameters,
    Message
)
from fifo_tool_airlock_model_env.sdk.client_sdk import (
    call_airlock_model_server,
    Model
)
from fifo_dev_dsl.domain_specific.common.dsl_utils import (
    extract_hour_minute,
    extract_int,
    extract_month,
    get_arg,
    parse_dsl_expression
)

SYSTEM_PROMPT = ("You are a precise temporal parser. Your only job is to translate natural language"
                 " date expressions into structured DSL function calls such as OFFSET(...) or"
                 " DATE_FROM_MONTH_DAY(...). Do not explain or elaborate. Only return the code.")

def parse_natural_date_expression(
        question: str,
        container_name: str,
        adapter: str = "mini-date-converter-dsl",
        now: datetime | None = None,
        host: str = "http://127.0.0.1:8000") -> Tuple[str, datetime]:
    """
    Given a natural language date expression, this function uses the LLM model to translate it
    to the DSL, then parses and returns the corresponding datetime.

    Args:
        question (str):
            The natural language question, e.g., "in one day and two hours"

        container_name (str):
            Container for the model server.

        adapter (str, optional):
            Adapter name used when calling `call_airlock_model_server`. Defaults to
            `"mini-date-converter-dsl"`.

        now (datetime | None, optional):
            Overrides the current datetime for evaluation. Passed to
            `MiniDateConverterDSL`.

        host (str, optional):
            URL of the airlock model server.

    Returns:
        Tuple[str, datetime]:
            (the DSL code, the parsed datetime object)
    """
    answer = call_airlock_model_server(
        model=Model.Phi4MiniInstruct,
        adapter=adapter,
        messages=[
            Message.system(SYSTEM_PROMPT),
            Message.user(question)
        ],
        parameters=GenerationParameters(
            max_new_tokens=1024,
            do_sample=False
        ),
        container_name=container_name,
        host=host
    )

    try:
        dt = MiniDateConverterDSL(now=now).parse(answer)
    except ValueError as e:
        raise ValueError(f"{e} (dsl='{answer}')") from e

    return answer, dt


class MiniDateConverterDSL:
    """
    MiniDateConverterDSL is a lightweight interpreter for a symbolic, domain-specific language (DSL)
    that converts structured date and time expressions into executable Python datetime values.

    Supported DSL Functions:
    ------------------------

    - TODAY
        Returns the current date (with time set to 00:00).

    - OFFSET(base_expr, value, unit)
        Adds or subtracts a time offset to a base date expression.
        Unit must be one of: DAY, WEEK, MONTH, YEAR, WEEKDAY=<weekday_index 0-6>.

        Example:
            OFFSET(TODAY, 2, DAY)
            OFFSET(DATE_FROM_MONTH_DAY(12, 25), 1, YEAR)

    - DATE_FROM_MONTH_DAY(month, day)
        Constructs a date using this year with the given month and day. If that
        date has already passed, the same month/day of the next year is used.

        Example:
            DATE_FROM_MONTH_DAY(12, 25)

    - DATE_FROM_YEAR_MONTH_DAY(year, month, day)
        Constructs a specific date.

        Example:
            DATE_FROM_YEAR_MONTH_DAY(2025, 5, 1)

    - DATE_FROM_MONTH_WEEKDAY(month, weekday_index, occurrence)
        Finds the nth occurrence of a weekday in the given month of the current
        year. If the resulting date is in the past, the next year's occurrence
        is returned instead.
        Weekday must be an integer from 0 (Monday) to 6 (Sunday).
        `occurrence` may be negative to count from the end of the month
        (`-1` is the last weekday, `-2` the second to last, etc.).

        Example:
            DATE_FROM_MONTH_WEEKDAY(11, 3, 4)   # 4th Thursday of November (Thanksgiving in US)
            DATE_FROM_MONTH_WEEKDAY(10, 4, -1)  # last Friday of October

    - DATE_FROM_YEAR_MONTH_WEEKDAY(year, month, weekday_index, occurrence)
        Same as above, but with an explicit year.
        `occurrence` may be negative to count from the end of the month
        (`-1` is the last weekday, `-2` the second to last, etc.).

        Example:
            DATE_FROM_YEAR_MONTH_WEEKDAY(2026, 1, 0, 2)   # 2nd Monday of January 2026
            DATE_FROM_YEAR_MONTH_WEEKDAY(2026, 10, 4, -1) # last Friday of October 2026

    - SET_MONTH_DAY(date_expr, day)
        Sets the day-of-month on `date_expr`. `day` may be negative to count
        backwards from the end of the month (-1 is the last day).

        Example:
            SET_MONTH_DAY(OFFSET(TODAY, 1, MONTH), 1)  # first of next month
            SET_MONTH_DAY(TODAY, -1)  # last day of this month

    - SET_TIME(date_expr, hour, minute)
        Sets the hour and minute for a given date expression, returning a datetime with the
        specified time. Hour uses 24-hour format (0-23). Minute is 0-59.

        Example:
            SET_TIME(TODAY, 17, 30)  # today at 5:30pm

    - OFFSET_TIME(date_expr, hours, minutes)
        Adds or subtracts a time offset (hours and minutes) to a given date or datetime expression.
        Both hours and minutes can be positive or negative.

        Example:
            OFFSET_TIME(TODAY, 2, 45)  # in 2 hours and 45 minutes
            OFFSET_TIME(SET_TIME(TODAY, 12, 0), 0, 30)  # today at 12:30pm

    Notes:
    ------
    - All DSL expressions evaluate to Python `datetime` objects.
    - Functions can be nested to create complex temporal expressions.
    - All inputs must be numeric and valid — out-of-range or malformed values will raise ValueError.
    """

    WEEKDAY_MAP = [MO, TU, WE, TH, FR, SA, SU]

    def __init__(self, now: datetime | None = None):
        self.input_now = now or datetime.now()

    def parse(self, expr: str):
        """
        Parses a DSL expression and returns a datetime object.

        Args:
            expr (str):
                The DSL expression to parse and evaluate.

        Returns:
            datetime:
                The resulting datetime object from evaluating the expression.
        """
        dt, time_modified = self._parse(expr)
        if not time_modified:
            dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return dt

    def _parse(self, expr: str) -> Tuple[datetime, bool]:
        return parse_dsl_expression(
            expr=expr,
            evaluator=self._evaluate,
            allow_bare_identifiers=True
        )

    def _evaluate(self, func: str, args: list[str]) -> Tuple[datetime, bool]:
        """
        Evaluates a DSL function with the given arguments.
        Args:
            func (str):
                The name of the DSL function.

            args (list[str]):
                The list of arguments (as strings) to evaluate.

        Returns:
            Tuple[datetime, bool]:
                (The evaluated datetime object, bool indicating if the time has been set)
        """
        if func != func.upper():
            raise ValueError("DSL function names must be uppercase")

        if func == "TODAY":
            if len(args) != 0:
                raise ValueError("TODAY takes no arguments")
            return self.input_now, False

        if func == "OFFSET":
            if len(args) > 3:
                raise ValueError("OFFSET requires exactly 3 arguments")
            base, base_time_mod = self._parse(args[0])
            value = int(args[1])
            unit = args[2].upper()
            if unit == "DAY":
                return base + timedelta(days=value), base_time_mod
            if unit == "WEEK":
                return base + timedelta(weeks=value), base_time_mod
            if unit == "MONTH":
                return base + relativedelta(months=value), base_time_mod
            if unit == "YEAR":
                return base + relativedelta(years=value), base_time_mod
            if unit.startswith("WEEKDAY="):
                target_day = int(unit.split("=")[-1])
                weekday_func = self.WEEKDAY_MAP[target_day]

                # `dateutil` returns the same day when the base already falls
                # on the requested weekday. The DSL expects "next" or
                # "previous" weekday depending on the sign of `value`.
                # Adjust the search start by one day in that case so that
                # `OFFSET(TODAY, 1, WEEKDAY=x)` always moves away from today
                # when today is already the target weekday.
                extra_day = 0
                if base.weekday() == target_day and value != 0:
                    extra_day = 1 if value > 0 else -1

                return base + relativedelta(days=extra_day,
                                            weekday=weekday_func(value)), base_time_mod

            raise ValueError(f"Unknown unit in OFFSET: {unit}")

        if func == "DATE_FROM_MONTH_DAY":
            if len(args) > 2:
                raise ValueError("DATE_FROM_MONTH_DAY requires exactly 2 arguments")
            month = extract_month(args, 0, func)
            day = extract_int(args, 1, "day", func)
            year = self.input_now.year

            for offset in range(10):  # search up to 10 years ahead
                try:
                    target = datetime(year + offset, month, day)
                    if target >= self.input_now:
                        return target, False
                except ValueError:
                    continue  # skip invalid dates like Feb 29 on non-leap years

            raise ValueError(f"DATE_FROM_MONTH_DAY({month}, {day}) is invalid")

        if func == "DATE_FROM_YEAR_MONTH_DAY":
            if len(args) > 3:
                raise ValueError("DATE_FROM_YEAR_MONTH_DAY requires exactly 3 arguments")
            year = extract_int(args, 0, "year", func)
            month = extract_month(args, 1, func)
            day = extract_int(args, 2, "day", func)

            try:
                return datetime(year, month, day), False
            except ValueError as e:
                raise ValueError(f"{func}({year}, {month}, {day}) is invalid") from e

        if func == "DATE_FROM_MONTH_WEEKDAY":
            if len(args) > 3:
                raise ValueError("DATE_FROM_MONTH_WEEKDAY requires exactly 3 arguments")
            month = extract_month(args, 0, func)
            weekday_index = extract_int(args, 1, "weekday", func)

            if not 0 <= weekday_index < len(self.WEEKDAY_MAP):
                raise ValueError(f"Invalid weekday in {func}: got {weekday_index}")

            weekday = self.WEEKDAY_MAP[weekday_index]
            occurrence = extract_int(args, 2, "occurrence", func)

            for offset in range(10):  # search up to 10 years ahead
                try:
                    anchor = datetime(self.input_now.year + offset, month, 1)
                    if occurrence < 0:
                        anchor += relativedelta(months=1, days=-1)
                    candidate = anchor + relativedelta(weekday=weekday(occurrence))
                    if candidate >= self.input_now:
                        return candidate, False
                except ValueError:
                    continue

            raise ValueError(
                f"Failed to compute {func}({month}, {weekday_index}, {occurrence})"
            )

        if func == "DATE_FROM_YEAR_MONTH_WEEKDAY":
            if len(args) > 4:
                raise ValueError("DATE_FROM_YEAR_MONTH_WEEKDAY requires exactly 4 arguments")
            year = extract_int(args, 0, "year", func)
            month = extract_month(args, 1, func)
            weekday_index = extract_int(args, 2, "weekday", func)

            if not 0 <= weekday_index < len(self.WEEKDAY_MAP):
                raise ValueError(f"Invalid weekday in {func}: got {weekday_index}")

            weekday = self.WEEKDAY_MAP[weekday_index]
            occurrence = extract_int(args, 3, "occurrence", func)

            try:
                anchor = datetime(year, month, 1)
                if occurrence < 0:
                    anchor += relativedelta(months=1, days=-1)
                return anchor + relativedelta(weekday=weekday(occurrence)), False
            except ValueError as e:
                raise ValueError(
                    f"Failed to compute {func}({year}, {month}, {weekday_index}, {occurrence}): {e}"
                ) from e

        if func == "SET_MONTH_DAY":
            if len(args) > 2:
                raise ValueError("SET_MONTH_DAY requires exactly 2 arguments")
            try:
                base, base_time_mod = self._parse(args[0])
            except (IndexError, ValueError) as e:
                raise ValueError(
                    "Invalid or missing base expression in SET_MONTH_DAY: "
                    f"got {get_arg(args, 0)!r}"
                ) from e

            day_val = extract_int(args, 1, "day", func)

            last_of_month = (
                base.replace(day=1)
                + relativedelta(months=1)
                - timedelta(days=1)
            ).day

            if day_val > 0:
                new_day = day_val
            else:
                new_day = last_of_month + 1 + day_val

            if not 1 <= new_day <= last_of_month:
                raise ValueError(f"SET_MONTH_DAY({day_val}) is invalid")

            return base.replace(day=new_day), base_time_mod

        if func == "SET_TIME":
            if len(args) > 3:
                raise ValueError("SET_TIME requires exactly 3 arguments")
            try:
                base, _ = self._parse(args[0])
            except (IndexError, ValueError) as e:
                raise ValueError("Invalid or missing base expression in SET_TIME: "
                                 f"got {get_arg(args, 0)!r}") from e

            hour, minute = extract_hour_minute(args, 1, 2, func)
            return base.replace(hour=hour, minute=minute, second=0, microsecond=0), True

        if func == "OFFSET_TIME":
            if len(args) > 3:
                raise ValueError("OFFSET_TIME requires exactly 3 arguments")
            try:
                base, _ = self._parse(args[0])
            except (IndexError, ValueError) as e:
                raise ValueError("Invalid or missing base expression in OFFSET_TIME: "
                                 f"got {get_arg(args, 0)!r}") from e

            # No range validation needed - only type/structure
            hours = extract_int(args, 1, "hour", func)
            minutes = extract_int(args, 2, "minute", func)

            return base + timedelta(hours=hours, minutes=minutes), True

        raise ValueError(f"Unknown function: {func}")
