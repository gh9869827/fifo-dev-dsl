from __future__ import annotations
from datetime import datetime, timedelta
from calendar import monthrange
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Tuple
from enum import Enum, auto
from fifo_tool_airlock_model_env.common.models import (
    GenerationParameters,
    Message
)
from fifo_tool_airlock_model_env.sdk.client_sdk import (
    call_airlock_model_server,
    Model
)
from fifo_dev_dsl.domain_specific.common.dsl_utils import (
    extract_int,
    extract_month,
    extract_positive_int,
    get_arg,
    parse_dsl_expression
)

SYSTEM_PROMPT = ("You are a precise parser of recurring schedule expressions. Your only job is to"
                 " translate natural language recurrence expressions into structured DSL function"
                 " calls such as WEEKLY(...) or MONTHLY_BY_WEEKDAY(...). Do not explain or"
                 " elaborate. Only return the code.")

def parse_natural_recurrence_expression(
        question: str,
        container_name: str,
        adapter: str="mini-recurrence-converter-dsl-adapter",
        host: str = "http://127.0.0.1:8000") -> Tuple[str, RecurrenceRule]:
    """
    Given a natural language recurrence expression, this function uses the LLM model to translate it
    to the DSL, then parses and returns the corresponding RecurrenceRule object.

    Args:
        question (str):
            The natural language question, e.g., "in one day and two hours"

        container_name (str):
            Container for the model server.

        adapter (str, optional):
            Adapter name used when calling `call_airlock_model_server`. Defaults to
            `"mini-recurrence-converter-dsl-adapter"`.

        host (str, optional):
            URL of the airlock model server.

    Returns:
        Tuple[str, RecurrenceRule]: 
            (the DSL code, the parsed RecurrenceRule object)
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
        dt = MiniRecurrenceConverterDSL().parse(answer)
    except ValueError as e:
        raise ValueError(f"{e} (dsl='{answer}')") from e

    return answer, dt


class RecurrenceUnit(Enum):
    """
    Enum for supported recurrence pattern types in MiniRecurrenceConverterDSL.

    Attributes:
        DAILY:   Repeat every N days (e.g., every day, every 3 days).
        WEEKLY:  Repeat every N weeks on specific weekdays (e.g., every week on Monday and 
                 Thursday).
        MONTHLY: Repeat every N months on a specific day of the month (e.g., every month on the 
                 15th).
        MONTHLY_BY_WEEKDAY: Repeat every N months on the K-th weekday (e.g., every month on the 
                 2nd Tuesday).
        YEARLY:  Repeat every N years on a specific month and day (e.g., every year on July 4th).
        HOURLY:  Repeat every N hours and M minutes (e.g., every 90 minutes).
    """
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    MONTHLY_BY_WEEKDAY = auto()
    YEARLY = auto()
    HOURLY = auto()

@dataclass
class RecurrenceRule:
    """
    Represents a structured recurrence rule parsed from the MiniRecurrenceConverterDSL.

    Attributes:
        unit (RecurrenceUnit):
            The recurrence pattern type (e.g., DAILY, WEEKLY, etc.).
        frequency (int):
            Main frequency of recurrence (e.g., every 2 weeks = 2).
        days (list[int] | None):
            List of weekday indices (0=Monday to 6=Sunday) for WEEKLY and MONTHLY_BY_WEEKDAY
            patterns.
        day_of_month (int | None):
            Specific day of the month for MONTHLY or YEARLY patterns (use -1 for last day).
        month (int | None):
            Month number (1-12) for YEARLY patterns.
        occurrence (int | None):
            Occurrence number for MONTHLY_BY_WEEKDAY (e.g., 2 for second Monday).
        hour (int | None):
            Time of day (hour, 0-23) for patterns with a time component. Defaults to 0.
        minute (int | None):
            Time of day (minute, 0-59) or interval minutes for HOURLY patterns. Defaults to 0.

    Class Attributes:
        WEEKDAY_STR_TO_INT (dict[str, int]):
            Mapping from weekday strings (e.g., 'MO', 'FR') to integers (0=Monday, ..., 6=Sunday).
        WEEKDAY_INT_TO_STR (dict[int, str]):
            Mapping from integers (0=Monday, ..., 6=Sunday) back to strings (e.g., 'MO', 'FR').

    Methods:
        days_from_strings(days_strs: list[str]) -> list[int]:
            Converts a list of weekday strings (['MO', 'FR']) to a list of integers ([0, 4]).

        days_to_strings(days_ints: list[int]) -> list[str]:
            Converts a list of weekday integers ([0, 4]) to a list of strings (['MO', 'FR']).

        to_dict():
            Serialize the RecurrenceRule instance to a plain dictionary, converting any
            non-JSON-serializable fields (such as Enums) to a serializable representation.

        from_dict(d):
            Construct a RecurrenceRule instance from a dictionary (as produced by to_dict()).
            This handles re-converting any serialized fields (such as Enums) back to the correct
            type.
    """

    unit: RecurrenceUnit
    frequency: int = 1
    days: list[int] | None = None
    day_of_month: int | None = None
    month: int | None = None
    occurrence: int | None = None
    hour: int | None = None
    minute: int | None = None

    WEEKDAY_STR_TO_INT: ClassVar[dict[str, int]] = {
        "MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6
    }
    WEEKDAY_INT_TO_STR: ClassVar[dict[int, str]] = {
        v: k for k, v in WEEKDAY_STR_TO_INT.items()
    }

    @classmethod
    def days_from_strings(cls, days_strs: list[str]) -> list[int]:
        """
        Convert a list of weekday string codes to their corresponding integer indices.

        Args:
            days_strs (list[str]):
                List of two-letter weekday codes, e.g., ['MO', 'WE', 'FR'].

        Returns:
            list[int]:
                List of weekday indices, where 0=Monday, ..., 6=Sunday.
                Example: ['MO', 'WE'] -> [0, 2]
        """
        return [cls.WEEKDAY_STR_TO_INT[d.upper()] for d in days_strs]

    @classmethod
    def days_to_strings(cls, days_ints: list[int]) -> list[str]:
        """
        Convert a list of weekday integer indices to their corresponding two-letter codes.

        Args:
            days_ints (list[int]):
                List of weekday indices, where 0=Monday, ..., 6=Sunday.

        Returns:
            list[str]:
                List of two-letter weekday codes.
                Example: [0, 2] -> ['MO', 'WE']
        """
        return [cls.WEEKDAY_INT_TO_STR[i] for i in days_ints]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this RecurrenceRule into a plain dictionary suitable for JSON serialization.

        Converts Enum fields (e.g., unit) to their string names.

        Returns:
            dict[str, Any]:
                Dictionary with the same fields as RecurrenceRule, but:
                - Enums (like `unit`) are converted to string
                - Optional values are preserved (e.g., None)
                - Only standard JSON types used (str, int, list, None)
        """
        d = asdict(self)
        # Convert Enum to string for JSON
        d['unit'] = self.unit.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RecurrenceRule:
        """
        Create a RecurrenceRule instance from a dictionary, reversing the conversion
        done by to_dict().

        Args:
            d (dict[str, Any]):
                Dictionary representing a RecurrenceRule, typically produced by to_dict().

        Returns:
            RecurrenceRule:
                The reconstructed recurrence rule instance.
        """
        d = dict(d)  # copy
        d['unit'] = RecurrenceUnit[d['unit']]
        return cls(**d)

    def next(self, dt: datetime) -> datetime:
        """
        Compute the next occurrence of this recurrence after the given datetime.
        If no next recurrence, raises ValueError.
        """
        if self.unit == RecurrenceUnit.DAILY:
            next_dt = dt + timedelta(days=self.frequency)
            next_dt = next_dt.replace(
                hour=self.hour or 0, minute=self.minute or 0, second=0, microsecond=0
            )
            return next_dt

        if self.unit == RecurrenceUnit.WEEKLY:
            if not self.days:
                # No days specified: just advance by frequency weeks
                next_dt = dt + timedelta(weeks=self.frequency)
                next_dt = next_dt.replace(
                    hour=self.hour or 0, minute=self.minute or 0, second=0, microsecond=0
                )
                return next_dt
            # Find the next matching weekday in self.days after dt
            days_sorted = sorted(self.days)
            current_weekday = dt.weekday()
            # List of (offset_days, weekday) for each allowed day
            candidates: list[tuple[int, int]] = []
            for day in days_sorted:
                offset = (day - current_weekday) % 7
                # If today is the day, move to next occurrence
                if offset == 0:
                    offset = 7 * self.frequency
                candidates.append((offset, day))
            min_offset = min(offset for offset, _ in candidates)
            next_dt = dt + timedelta(days=min_offset)
            next_dt = next_dt.replace(
                hour=self.hour or 0, minute=self.minute or 0, second=0, microsecond=0
            )
            return next_dt

        if self.unit == RecurrenceUnit.MONTHLY:
            # Day of month specified
            year, month = dt.year, dt.month
            for _ in range(100):  # Prevent infinite loop
                # Compute the next month (frequency-based)
                month += self.frequency
                y_inc, month = divmod(month - 1, 12)
                year += y_inc
                month = month + 1  # 1-based months
                if self.day_of_month == -1:
                    day = monthrange(year, month)[1]  # Last day of month
                else:
                    day = self.day_of_month or dt.day
                    last_day = monthrange(year, month)[1]
                    if day > last_day:
                        day = last_day  # Clamp
                try:
                    next_dt = datetime(year, month, day, self.hour or 0, self.minute or 0)
                    if next_dt > dt:
                        return next_dt
                except ValueError:
                    continue
            raise ValueError("Couldn't find next monthly occurrence")

        if self.unit == RecurrenceUnit.YEARLY:
            year = dt.year + self.frequency
            month = self.month or dt.month
            day = self.day_of_month or dt.day
            # Clamp day if necessary
            last_day = monthrange(year, month)[1]
            if day > last_day:
                day = last_day
            next_dt = datetime(year, month, day, self.hour or 0, self.minute or 0)
            return next_dt

        if self.unit == RecurrenceUnit.HOURLY:
            next_dt = dt + timedelta(hours=self.frequency)
            next_dt = next_dt.replace(minute=self.minute or 0, second=0, microsecond=0)
            return next_dt

        # Add more cases as needed (MONTHLY_BY_WEEKDAY, etc.)
        raise NotImplementedError(f"next() not implemented for unit: {self.unit}")

class MiniRecurrenceConverterDSL:
    """
    MiniRecurrenceConverterDSL is a lightweight domain-specific language (DSL) for expressing
    recurring date/time patterns, suitable for reminders, calendar events, and scheduling systems.

    Supported Functions (use positional arguments only)
    ---------------------------------------------------

    - DAILY(frequency, [TIME(hour, minute)])
        Every `frequency` days, optionally at a specific time.

        - `frequency`: required integer.
        - `TIME(hour, minute)`: optional, defaults to 00:00.

        Examples:
            DAILY(1, TIME(9, 0))     # every day at 09:00
            DAILY(3)                 # every 3 days at 00:00

    - WEEKLY(frequency, [days], [TIME(hour, minute)])
        Every `frequency` weeks, optionally on specific days and at a specific time.

        - `frequency`: required integer.
        - `days`: optional list of weekday codes (e.g., [MO, TU, FR]).
        - `TIME(hour, minute)`: optional, defaults to 00:00.

        Examples:
            WEEKLY(1, [MO, WE], TIME(10, 0))    # every Monday and Wednesday at 10:00
            WEEKLY(2, [FR])                     # every other Friday at 00:00
            WEEKLY(3)                           # every 3 weeks at 00:00 (no weekday constraint)

    - MONTHLY(frequency, [day_of_month], [TIME(hour, minute)])
        Every `frequency` months, optionally on a specific day and time.

        - `frequency`: required integer.
        - `day_of_month`: optional integer (1-31 or -1 for the last day); defaults to 1.
        - `TIME(hour, minute)`: optional, defaults to 00:00.

        Examples:
            MONTHLY(1, 1, TIME(8, 0))    # every month on the 1st at 08:00
            MONTHLY(2)                   # every 2 months on the 1st at 00:00
            MONTHLY(1, -1)               # every month on the last day at 00:00

    - MONTHLY_BY_WEEKDAY(frequency, [weekday, occurrence], [TIME(hour, minute)])
        Every `frequency` months, optionally on the Nth weekday (e.g., [MO, 2]) and time.

        - `frequency`: required integer.
        - `weekday`: required weekday; `weekday` is MO-SU
        - `occurrence`: required occurence; `occurrence` is e.g. 2 for 2nd.
        - `TIME(hour, minute)`: optional, defaults to 00:00.

        Examples:
            MONTHLY_BY_WEEKDAY(1, MO, 2, TIME(15, 0))   # 2nd Monday at 15:00 every month

    - YEARLY(frequency, [month, day], [TIME(hour, minute)])
        Every `frequency` years, optionally on a specific month/day pair and time.

        - `frequency`: required integer.
        - `month, day`: optional month (1-12) and day (1-31); both must be provided together.
        - `TIME(hour, minute)`: optional, defaults to 00:00.

        Examples:
            YEARLY(1, 12, 25, TIME(18, 0))    # every Christmas at 18:00
            YEARLY(2, 7, 4)                   # every other July 4 at 00:00
            YEARLY(3)                         # every 3 years on today's date

    - HOURLY(frequency_hour, frequency_minute)
        Every `frequency_hour` hours and `frequency_minute` minutes.

        - `frequency_hour`: required integer (can be 0).
        - `frequency_minute`: required integer (can be 0).

        Examples:
            HOURLY(1, 30)    # every 1 hour and 30 minutes
            HOURLY(0, 20)    # every 20 minutes

    Notes:
    ------
    - Do **not** use keyword arguments (e.g., `frequency=1`). All arguments must be positional.
    - `TIME(hour, minute)` can appear anywhere in the argument list but must follow this format.
    """

    def parse(self, expr: str) -> RecurrenceRule:
        """
        Parses a recurring DSL expression and returns a RecurrenceRule object.

        Args:
            expr (str):
                The recurring DSL expression to parse.

        Returns:
            RecurrenceRule: 
                The parsed recurrence rule.
        """
        return parse_dsl_expression(
            expr=expr,
            evaluator=self._evaluate,
            allow_bare_identifiers=False
        )

    def _parse_time_arg(self, arg: str) -> tuple[int, int]:
        """
        Parses a TIME(hour, minute) expression string and validates the range.

        Args:
            arg (str):
                A string in the form 'TIME(HH, MM)'.

        Returns:
            tuple[int, int]:
                The validated (hour, minute) values.

        Raises:
            ValueError:
                If the format is invalid or values are out of range.
        """
        if not arg.startswith("TIME(") or not arg.endswith(")"):
            raise ValueError(f"Invalid TIME argument: {arg!r}")

        inner = arg[5:-1]  # strip 'TIME(' and ')'
        parts = [p.strip() for p in inner.split(",")]

        hour = extract_int(parts, 0, "hour", "TIME")
        minute = extract_int(parts, 1, "minute", "TIME")

        if not 0 <= hour <= 23:
            raise ValueError(f"Hour {hour} is out of range in TIME (expected 0-23)")

        if not 0 <= minute <= 59:
            raise ValueError(f"Minute {minute} is out of range in TIME (expected 0-59)")

        return hour, minute

    def _evaluate(self, func: str, args: list[str]) -> RecurrenceRule:
        if func != func.upper():
            raise ValueError("DSL function names must be uppercase")

        try:
            unit_enum = RecurrenceUnit[func]
        except KeyError as e:
            raise ValueError(f"Unknown recurrence function: {func}") from e

        if unit_enum == RecurrenceUnit.DAILY:
            frequency = extract_positive_int(args, 0, "frequency", "DAILY")

            if len(args) > 1:
                hour, minute = self._parse_time_arg(args[1])
            else:
                hour, minute = None, None

            return RecurrenceRule(
                unit=unit_enum,
                frequency=frequency,
                hour=hour,
                minute=minute
            )

        if unit_enum == RecurrenceUnit.WEEKLY:
            frequency = extract_positive_int(args, 0, "frequency", "WEEKLY")

            days = None
            hour, minute = None, None

            if len(args) > 3:
                raise ValueError("Too many arguments passed to WEEKLY")

            if len(args) >= 2:
                arg = get_arg(args, 1)
                if arg.startswith("[") and arg.endswith("]"):
                    try:
                        day_strs = [d.strip().strip('"\'') for d in arg[1:-1].split(',')]
                        days = RecurrenceRule.days_from_strings(day_strs)
                    except Exception as e:
                        raise ValueError(f"Invalid day list in WEEKLY: {arg}") from e
                elif arg.startswith("TIME("):
                    hour, minute = self._parse_time_arg(arg)
                else:
                    raise ValueError("Second argument to WEEKLY must be [days] or TIME(...), "
                                     f"got {arg!r}")

            if len(args) == 3:
                time_arg = get_arg(args, 2)
                hour, minute = self._parse_time_arg(time_arg)

            return RecurrenceRule(
                unit=unit_enum,
                frequency=frequency,
                days=days,
                hour=hour,
                minute=minute
            )

        if unit_enum == RecurrenceUnit.MONTHLY:
            frequency = extract_positive_int(args, 0, "frequency", "MONTHLY")

            day_of_month = None
            hour, minute = None, None

            if len(args) > 3:
                raise ValueError("Too many arguments passed to MONTHLY")

            if len(args) >= 2:
                day_of_month = extract_int(args, 1, "day_of_month", "MONTHLY")

                if not (1 <= day_of_month <= 31 or -31 <= day_of_month <= -1):
                    raise ValueError(f"day_of_month {day_of_month} is out of range in MONTHLY "
                                     "(expected 1-31 or -1 to -31)")

            if len(args) == 3:
                hour, minute = self._parse_time_arg(args[2])

            return RecurrenceRule(
                unit=unit_enum,
                frequency=frequency,
                day_of_month=day_of_month,
                hour=hour,
                minute=minute
            )

        if unit_enum == RecurrenceUnit.MONTHLY_BY_WEEKDAY:
            frequency = extract_positive_int(args, 0, "frequency", "MONTHLY_BY_WEEKDAY")

            if len(args) < 3:
                raise ValueError("MONTHLY_BY_WEEKDAY requires a weekday and occurrence as second "
                                 "and third arguments")

            weekday_str = get_arg(args, 1).upper()
            if weekday_str not in RecurrenceRule.WEEKDAY_STR_TO_INT:
                raise ValueError(f"Invalid weekday in MONTHLY_BY_WEEKDAY: {weekday_str!r}")
            weekday = RecurrenceRule.WEEKDAY_STR_TO_INT[weekday_str]

            occurrence = extract_int(args, 2, "occurrence", "MONTHLY_BY_WEEKDAY")

            hour, minute = None, None
            if len(args) >= 4:
                hour, minute = self._parse_time_arg(args[3])

            return RecurrenceRule(
                unit=unit_enum,
                frequency=frequency,
                days=[weekday],
                occurrence=occurrence,
                hour=hour,
                minute=minute
            )

        if unit_enum == RecurrenceUnit.YEARLY:
            frequency = extract_positive_int(args, 0, "frequency", "YEARLY")

            month = None
            day = None
            hour, minute = None, None

            if len(args) > 4:
                raise ValueError("Too many arguments passed to YEARLY")

            if len(args) == 2:
                raise ValueError("YEARLY requires month and day as second and third arguments")

            if len(args) >= 3:
                month = extract_month(args, 1, "YEARLY")
                day = extract_int(args, 2, "day", "YEARLY")
                if not 1 <= day <= 31:
                    raise ValueError(f"Day {day} is out of range in YEARLY (expected 1-31)")

            if len(args) == 4:
                hour, minute = self._parse_time_arg(args[3])

            return RecurrenceRule(
                unit=unit_enum,
                frequency=frequency,
                month=month,
                day_of_month=day,
                hour=hour,
                minute=minute
            )

        if unit_enum == RecurrenceUnit.HOURLY:
            if len(args) != 2:
                raise ValueError(
                    "HOURLY requires exactly 2 arguments: frequency_hour, frequency_minute"
                )

            freq_hour = extract_int(args, 0, "hour", "HOURLY")
            freq_minute = extract_int(args, 1, "minute", "HOURLY")

            if freq_hour < 0:
                raise ValueError("Hour in HOURLY must be zero or positive")

            if freq_minute < 0:
                raise ValueError("Minute in HOURLY must be zero or positive")

            if freq_hour == 0 and freq_minute == 0:
                raise ValueError("HOURLY frequency must be greater than zero")

            return RecurrenceRule(
                unit=unit_enum,
                frequency=freq_hour,
                minute=freq_minute
            )

        # we should never reach this statement are all RecurrenceUnit value are covered
        assert False
