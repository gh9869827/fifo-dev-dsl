from __future__ import annotations
import warnings
from datetime import datetime, timedelta
from typing import Tuple
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU
from fifo_dev_dsl.common.llm_abstraction import LlmBackend, LlmRequest
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
        adapter: str = "mini-date-converter-dsl-adapter",
        now: datetime | None = None,
        host: str = "http://127.0.0.1:8000") -> Tuple[str, datetime]:
    """
    Given a natural language date expression, this function uses the LLM model to translate it
    to the DSL, then parses and returns the corresponding datetime.

    Deprecated:
        Use `parse_natural_date_expression_with_backend` instead, which supports
        pluggable LLM backends through the LlmBackend protocol.

    Args:
        question (str):
            The natural language question, e.g., "in one day and two hours"

        container_name (str):
            Container for the model server.

        adapter (str, optional):
            Adapter name used when calling `call_airlock_model_server`. Defaults to
            `"mini-date-converter-dsl-adapter"`.

        now (datetime | None, optional):
            Overrides the current datetime for evaluation. Passed to
            `MiniDateConverterDSL`.

        host (str, optional):
            URL of the airlock model server.

    Returns:
        Tuple[str, datetime]:
            (the DSL code, the parsed datetime object)
    """
    warnings.warn(
        "parse_natural_date_expression is deprecated. "
        "Use parse_natural_date_expression_with_backend instead, "
        "which supports pluggable LLM backends through the LlmBackend protocol.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import AirlockBackend and AirlockModelEnv only when needed (for backward compatibility)
    from fifo_dev_dsl.common.llm_abstraction import AirlockBackend # pylint: disable=import-outside-toplevel
    from fifo_tool_airlock_model_env.common.models import Model # pylint: disable=import-outside-toplevel

    backend = AirlockBackend(
        container_name=container_name,
        adapter=adapter,
        host=host,
        base_model=Model.Phi4MiniInstruct
    )

    return parse_natural_date_expression_with_backend(
        question,
        now,
        backend=backend,
        max_new_tokens=1024,
        temperature=0.0
    )


def parse_natural_date_expression_with_backend(
        question: str,
        now: datetime | None = None,
        *,
        backend: LlmBackend,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        reasoning_effort: str | None = None) -> Tuple[str, datetime]:
    """
    Given a natural language date expression, this function uses an LLM backend to translate it
    to the DSL, then parses and returns the corresponding datetime.

    This function uses the LlmBackend protocol, allowing you to use any compatible backend
    implementation (e.g., AirlockBackend, OpenAICompatibleBackend).

    Args:
        question (str):
            The natural language question, e.g., "in one day and two hours"

        now (datetime | None, optional):
            Overrides the current datetime for evaluation. Passed to
            `MiniDateConverterDSL`. Defaults to None (uses current time).

        backend (LlmBackend):
            LLM backend implementing the LlmBackend protocol. This can be an AirlockBackend,
            OpenAICompatibleBackend, or any other compatible backend.

        max_new_tokens (int, optional):
            Maximum number of tokens to generate. Defaults to 1024.

        temperature (float, optional):
            Sampling temperature (higher = more random). When 0.0, use greedy decoding.
            Defaults to 0.0.

        reasoning_effort (str | None, optional):
            Reasoning effort level for reasoning models. Only applicable when using
            reasoning-capable models. When None, the parameter is not passed to the
            backend. Defaults to None.

    Returns:
        Tuple[str, datetime]:
            (the DSL code, the parsed datetime object)

    Examples:
        >>> from fifo_dev_dsl.common.llm_abstraction import AirlockBackend
        >>> backend = AirlockBackend(
        ...     container_name="my-container",
        ...     adapter="mini-date-converter-dsl-adapter",
        ...     host="http://127.0.0.1:8000"
        ... )
        >>> dsl_code, dt = parse_natural_date_expression_with_backend(
        ...     "next Tuesday at 5pm",
        ...     backend=backend
        ... )
    """
    request = LlmRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=question,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort
    )

    answer = backend.complete(request)

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
        `day` may be negative to count backward from the end of the month
        (`-1` is the last day, `-2` is the second-to-last day, etc.).

        Example:
            DATE_FROM_MONTH_DAY(12, 25)
            DATE_FROM_MONTH_DAY(1, -1)   # last day of January

    - DATE_FROM_YEAR_MONTH_DAY(year, month, day)
        Constructs a specific date.
        `day` may be negative to count backward from the end of the month
        (`-1` is the last day, `-2` is the second-to-last day, etc.).

        Example:
            DATE_FROM_YEAR_MONTH_DAY(2025, 5, 1)
            DATE_FROM_YEAR_MONTH_DAY(2026, 1, -1)  # January 31st, 2026

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

            # Validate day parameter
            if day == 0:
                raise ValueError(f"DATE_FROM_MONTH_DAY({month}, {day}) is invalid")

            year = self.input_now.year

            for offset in range(10):  # search up to 10 years ahead
                try:
                    # Handle negative day values (count from end of month)
                    actual_day = day
                    if day < 0:
                        # Calculate last day of the month
                        last_of_month = (
                            datetime(year + offset, month, 1)
                            + relativedelta(months=1)
                            - timedelta(days=1)
                        ).day
                        # Convert negative index to positive day number
                        # E.g., -1 becomes last_of_month, -2 becomes last_of_month - 1
                        actual_day = last_of_month + 1 + day

                    target = datetime(year + offset, month, actual_day)
                    if target.date() >= self.input_now.date():
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

            # Validate day parameter
            if day == 0:
                raise ValueError(f"DATE_FROM_YEAR_MONTH_DAY({year}, {month}, {day}) is invalid")

            # Handle negative day values (count from end of month)
            if day < 0:
                # Calculate last day of the month
                last_of_month = (
                    datetime(year, month, 1)
                    + relativedelta(months=1)
                    - timedelta(days=1)
                ).day
                # Convert negative index to positive day number
                # E.g., -1 becomes last_of_month, -2 becomes last_of_month - 1
                day = last_of_month + 1 + day

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
                    if candidate.date() >= self.input_now.date():
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
