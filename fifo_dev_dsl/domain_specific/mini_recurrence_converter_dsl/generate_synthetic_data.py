import random
from typing import Iterator, cast
from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter

WEEKDAYS = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

WEEKDAY_FULL_NAME = {
    "MO": "Monday", "TU": "Tuesday", "WE": "Wednesday",
    "TH": "Thursday", "FR": "Friday", "SA": "Saturday", "SU": "Sunday"
}

def ordinal(n: int) -> str:
    """
    Converts an integer to its ordinal string representation.

    Args:
        n (int):
            The integer to convert. For values 1-5, returns '1st', '2nd', '3rd', '4th', or '5th'
            respectively. If the input is -1, returns 'last'. For all other values, returns the
            number followed by 'th' (e.g., '6th').
    
    Returns:
        str:
            The ordinal string representation of the input integer.
    """
    if n == -1:
        return "last"
    return {
        1: "1st",
        2: "2nd",
        3: "3rd",
        4: "4th",
        5: "5th"
    }.get(n, f"{n}th")

def month_name(m: int) -> str:
    """
    Return the English name for a month index.

    Args:
        m (int):
            Month number where `1` is January and `12` is December.

    Returns:
        str:
            The month name.
    """
    return MONTH_NAMES[m]

def gen_days() -> list[str]:
    """
    Randomly select one to three distinct weekdays.

    Returns:
        list[str]:
            Sorted weekday abbreviations such as `["MO", "WE"]`.
    """
    return sorted(random.sample(WEEKDAYS, k=random.randint(1, 3)))

def gen_frequency_hour() -> int:
    """
    Generate a random hour component for a frequency.

    Returns:
        int:
            Hour increment used in HOURLY expressions. `0` represents less than one hour.
    """
    return random.choice([0, 1, 2, 3, 6, 12])

def gen_frequency_minute() -> int:
    """
    Generate a random minute for a frequency component.

    Returns:
        int:
            Minute increment used for HOURLY expressions.
    """
    return random.choice([0, 5, 10, 15, 30, 45])

def gen_frequency() -> int:
    """
    Generate a generic frequency value.

    Returns:
        int:
            An integer between 1 and 24.
    """
    return random.randint(1, 24)

def gen_time_hh_mm() -> tuple[int, int]:
    """
    Generate a random time of day.

    Returns:
        tuple[int, int]:
            `(hour, minute)` in 24-hour format.
    """
    hour = random.choice([0, 6, 8, 9, 12, 15, 18, 21])
    minute = random.choice([0, 15, 30, 45])
    return hour, minute

def gen_month() -> int:
    """
    Pick a random month index.

    Returns:
        int:
            Value from `1` to `12` where `1` is January and `12` is December.
    """
    return random.randint(1, 12)

def gen_day() -> int:
    """
    Pick a random day of the month.

    Returns:
        int:
            Value between `1` and `31`.
    """
    return random.randint(1, 31)

def gen_day_of_month() -> int:
    """
    Choose a day of month or `-1` for the last day.

    Returns:
        int:
            Number from `1` to `31` or `-1`.
    """
    return random.choice([*range(1, 32), -1])

def gen_occurrence() -> int:
    """
    Generate an occurrence index for weekday-based rules.

    Returns:
        int:
            One of `[1-5]` or `-1` for the last occurrence.
    """
    return random.choice([1, 2, 3, 4, 5, -1])

def gen_weekday() -> str:
    """
    Select a random weekday abbreviation.

    Returns:
        str:
            One of `"MO"` through `"SU"`.
    """
    return random.choice(WEEKDAYS)

def format_TIME(hour: int, minute: int) -> str:
    """
    Convert a 24-hour time to a human-friendly string.

    Args:
        hour (int):
            Hour of day in 24-hour format.

        minute (int):
            Minute of hour.

    Returns:
        str:
            Formatted time such as `"8:30 AM"` or `"midnight"`.
    """
    if hour == 0 and minute == 0:
        return "midnight"

    if hour == 12 and minute == 0:
        return "noon"

    suffix = "AM" if hour < 12 else "PM"
    hour_12 = hour if 1 <= hour <= 12 else (hour - 12 if hour > 12 else 12)
    return f"{hour_12}:{minute:02d} {suffix}"

def format_HOURLY(frequency_hour: int, frequency_minute: int) -> tuple[str, str]:
    """
    Build natural text and DSL code for an `HOURLY` expression.

    Args:
        frequency_hour (int):
            Hour component of the frequency.

        frequency_minute (int):
            Minute component of the frequency.

    Returns:
        tuple[str, str]:
            `(readable text, DSL code)`.
    """
    if frequency_hour == 0 and frequency_minute == 0:
        # Regenerate if invalid combo
        return format_HOURLY(gen_frequency_hour(), gen_frequency_minute())

    parts: list[str] = []
    if frequency_hour > 0:
        parts.append(f"{frequency_hour} hour{'s' if frequency_hour > 1 else ''}")
    if frequency_minute > 0:
        parts.append(f"{frequency_minute} minute{'s' if frequency_minute > 1 else ''}")

    text = f"every {' and '.join(parts)}"
    dsl_code = f"HOURLY({frequency_hour}, {frequency_minute})"

    return text, dsl_code

def format_WEEKLY(
    frequency: int,
    days: list[str] | None = None,
    time_hh_mm: tuple[int, int] | None = None
) -> tuple[str, str]:
    """
    Generate natural language and DSL code for a `WEEKLY(...)` expression.

    Args:
        frequency (int):
            Number of weeks between occurrences.

        days (list[str] | None):
            Weekday abbreviations (e.g. ["MO", "WE"]).

        time_hh_mm (tuple[int, int] | None):
            Time of day (hour, minute) in 24-hour format.

    Returns:
        tuple[str, str]:
            (readable text, DSL code)
    """
    if frequency > 1:
        base = f"every {frequency} weeks"
    else:
        base = "every week"

    if days:

        names = [WEEKDAY_FULL_NAME[d] for d in days]
        if len(names) == 1:
            readable_days = names[0]
        elif len(names) == 2:
            readable_days = f"{names[0]} and {names[1]}"
        else:
            readable_days = ", ".join(names[:-1]) + f", and {names[-1]}"

        base += f" on {readable_days}"

    if time_hh_mm:
        assert days
        hour, minute = time_hh_mm
        base += f" at {format_TIME(hour, minute)}"
        code = f"WEEKLY({frequency}, [{', '.join(days)}], TIME({hour}, {minute}))"
    elif days:
        code = f"WEEKLY({frequency}, [{', '.join(days)}])"
    else:
        code = f"WEEKLY({frequency})"

    return base, code

def format_DAILY(frequency: int, time_hh_mm: tuple[int, int] | None = None) -> tuple[str, str]:
    """
    Generate natural language and DSL code for a `DAILY(...)` expression.

    Args:
        frequency (int):
            Number of days between occurrences.

        time_hh_mm (tuple[int, int] | None):
            Time of day (hour, minute) in 24-hour format.

    Returns:
        tuple[str, str]:
            (readable text, DSL code)
    """
    if frequency > 1:
        base = f"every {frequency} days"
    else:
        base = "every day"

    if time_hh_mm:
        hour, minute = time_hh_mm
        text = f"{base} at {format_TIME(hour, minute)}"
        code = f"DAILY({frequency}, TIME({hour}, {minute}))"
    else:
        text = base
        code = f"DAILY({frequency})"

    return text, code

def format_YEARLY(
    frequency: int,
    month: int | None = None,
    day: int | None = None,
    time_hh_mm: tuple[int, int] | None = None
) -> tuple[str, str]:
    """
    Generate natural language and DSL code for a `YEARLY(...)` expression.

    Args:
        frequency (int):
            Number of years between occurrences.

        month (int | None):
            Month value (1–12).

        day (int | None):
            Day of month (1–31).

        time_hh_mm (tuple[int, int] | None):
            Time of day (hour, minute) in 24-hour format.

    Returns:
        tuple[str, str]:
            (readable text, DSL code)
    """
    if month and day:
        base = f"every {frequency} year{'s' if frequency > 1 else ''} on {month_name(month)} {day}"
    else:
        base = f"every {frequency} year{'s' if frequency > 1 else ''}"

    if time_hh_mm:
        hour, minute = time_hh_mm
        text = f"{base} at {format_TIME(hour, minute)}"
        code = f"YEARLY({frequency}, {month}, {day}, TIME({hour}, {minute}))"
    elif month and day:
        text = base
        code = f"YEARLY({frequency}, {month}, {day})"
    else:
        text = base
        code = f"YEARLY({frequency})"

    return text, code

def format_MONTHLY(
    frequency: int,
    day_of_month: int | None = None,
    time_hh_mm: tuple[int, int] | None = None
) -> tuple[str, str]:
    """
    Generate natural language and DSL code for a `MONTHLY(...)` expression.

    Args:
        frequency (int):
            Number of months between occurrences.

        day_of_month (int | None):
            Day of month (1–31) or -1 for "last day".

        time_hh_mm (tuple[int, int] | None):
            Time of day (hour, minute) in 24-hour format.

    Returns:
        tuple[str, str]:
            (readable text, DSL code)
    """
    if frequency > 1:
        base = f"every {frequency} months"
    else:
        base = "every month"

    if day_of_month is not None:
        day_desc = (
            "last day" if day_of_month == -1
            else f"the {ordinal(day_of_month)}"
        )
        base = f"{base} on {day_desc}"

    if time_hh_mm:
        hour, minute = time_hh_mm
        base = f"{base} at {format_TIME(hour, minute)}"
        code = f"MONTHLY({frequency}, {day_of_month}, TIME({hour}, {minute}))"
    elif day_of_month is not None:
        code = f"MONTHLY({frequency}, {day_of_month})"
    else:
        code = f"MONTHLY({frequency})"

    return base, code

def format_MONTHLY_BY_WEEKDAY(
    frequency: int,
    weekday: str,
    occurrence: int,
    time_hh_mm: tuple[int, int] | None = None
) -> tuple[str, str]:
    """
    Generate natural language and DSL code for a `MONTHLY_BY_WEEKDAY(...)` expression.

    Args:
        frequency (int):
            Number of months between occurrences.

        weekday (str):
            Weekday abbreviation (e.g. "MO").

        occurrence (int):
            Ordinal occurrence (1–5 or -1 for "last").

        time_hh_mm (tuple[int, int] | None):
            Time of day (hour, minute) in 24-hour format.

    Returns:
        tuple[str, str]:
            (readable text, DSL code)
    """
    occ_desc = ordinal(occurrence)
    weekday_name = WEEKDAY_FULL_NAME[weekday]

    if frequency == 1:
        base = f"every {occ_desc} {weekday_name} of the month"
    else:
        base = f"every {frequency} months on the {occ_desc} {weekday_name}"

    if time_hh_mm:
        hour, minute = time_hh_mm
        base += f" at {format_TIME(hour, minute)}"
        code = f"MONTHLY_BY_WEEKDAY({frequency}, {weekday}, {occurrence}, TIME({hour}, {minute}))"
    else:
        code = f"MONTHLY_BY_WEEKDAY({frequency}, {weekday}, {occurrence})"

    return base, code

examples: set[tuple[str, str]] = set()

adapter_obj = DSLAdapter()

dataset_dict = adapter_obj.from_hub_to_dataset_wide_dict(
    "a6188466/mini-recurrence-converter-dsl-dataset"
)

existing_examples = set(
    entry["in"]
    for x in ["train", "test", "validation"]
    for entry in list(cast(Iterator[dict[str, str]], dataset_dict[x]))
)

# Helper to conditionally add only novel examples
def try_add(pair: tuple[str, str]) -> None:
    """
    Add a generated example if it is not already present in the dataset.

    Args:
        pair (tuple[str, str]):
            `(input text, DSL code)` pair to consider.
    """
    if pair[0] not in existing_examples:
        examples.add(pair)

# Generate 200 examples
while len(examples) < 200:

    # HOURLY
    try_add(format_HOURLY(gen_frequency_hour(), gen_frequency_minute()))

    # DAILY without time
    try_add(format_DAILY(gen_frequency()))

    # DAILY with time
    try_add(format_DAILY(gen_frequency(), gen_time_hh_mm()))

    # YEARLY — just frequency
    try_add(format_YEARLY(gen_frequency()))

    # YEARLY — with month and day
    try_add(format_YEARLY(gen_frequency(), gen_month(), gen_day()))

    # YEARLY — with month, day, and time
    try_add(format_YEARLY(gen_frequency(), gen_month(), gen_day(), gen_time_hh_mm()))

    # MONTHLY — base
    try_add(format_MONTHLY(gen_frequency()))

    # MONTHLY — with day_of_month
    try_add(format_MONTHLY(gen_frequency(), gen_day_of_month()))

    # MONTHLY — with day_of_month and time
    try_add(format_MONTHLY(gen_frequency(), gen_day_of_month(), gen_time_hh_mm()))

    # MONTHLY_BY_WEEKDAY — base
    try_add(format_MONTHLY_BY_WEEKDAY(gen_frequency(), gen_weekday(), gen_occurrence()))

    # MONTHLY_BY_WEEKDAY — with time
    try_add(format_MONTHLY_BY_WEEKDAY(
        gen_frequency(), gen_weekday(), gen_occurrence(), gen_time_hh_mm()
    ))

    # WEEKLY — frequency only
    try_add(format_WEEKLY(gen_frequency()))

    # WEEKLY — frequency + days
    try_add(format_WEEKLY(gen_frequency(), gen_days()))

    # WEEKLY — frequency + days + time
    try_add(format_WEEKLY(gen_frequency(), gen_days(), gen_time_hh_mm()))

# Print results
for input_text, dsl in sorted(examples):
    print("---")
    print("$ You are a precise parser of recurring schedule expressions. Your only job is to translate natural language recurrence expressions into structured DSL function calls such as WEEKLY(...) or MONTHLY_BY_WEEKDAY(...). Do not explain or elaborate. Only return the code.")
    print(f"> {input_text}\n< {dsl}")

print("---")
