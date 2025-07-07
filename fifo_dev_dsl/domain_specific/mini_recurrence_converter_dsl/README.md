# `fifo_dev_dsl.domain_specific.mini_recurrence_converter_dsl`

This module provides a symbolic DSL (domain-specific language) for representing **recurring schedule expressions**, such as "every Monday at 9am" or "every last day of the month." It includes a parser, evaluation logic, and recurrence rule representation that supports computation of the next occurrence.

---

## 📦 Pretrained model

The `parse_natural_recurrence_expression` function in the `core` module uses a language model to translate natural recurrence expressions into structured DSL function calls, as defined in this module.

A fine-tuned LoRA adapter is available here:  
👉 [**mini-recurrence-converter-dsl-adapter**](https://huggingface.co/a6188466/mini-recurrence-converter-dsl-adapter)

It was trained on this dataset of paired natural language and DSL examples:  
📚 [**mini-recurrence-converter-dsl-dataset**](https://huggingface.co/datasets/a6188466/mini-recurrence-converter-dsl-dataset)

---

## 📘 DSL Syntax and Examples

The DSL supports symbolic functions to express common recurrence patterns using positional arguments only — no keyword arguments.

---

### `DAILY(frequency, [TIME(hour, minute)])`

Every `frequency` days, optionally at a specific time.

- `frequency`: required integer.
- `TIME(hour, minute)`: optional, defaults to `00:00`.

**Examples:**
```
DAILY(1, TIME(9, 0))     # every day at 09:00  
DAILY(3)                 # every 3 days at 00:00  
```

---

### `WEEKLY(frequency, [days], [TIME(hour, minute)])`

Every `frequency` weeks, optionally on specific days and at a specific time.

- `frequency`: required integer.
- `days`: optional list of weekday codes (e.g., `[MO, TU, FR]`).
- `TIME(hour, minute)`: optional, defaults to `00:00`.

**Examples:**
```
WEEKLY(1, [MO, WE], TIME(10, 0))    # every Monday and Wednesday at 10:00  
WEEKLY(2, [FR])                     # every other Friday at 00:00  
WEEKLY(3)                           # every 3 weeks at 00:00 (no weekday constraint)  
```

---

### `MONTHLY(frequency, [day_of_month], [TIME(hour, minute)])`

Every `frequency` months, optionally on a specific day and time.

- `frequency`: required integer.
- `day_of_month`: optional integer (1–31 or -1 for last day); defaults to `1`.
- `TIME(hour, minute)`: optional, defaults to `00:00`.

**Examples:**
```
MONTHLY(1, 1, TIME(8, 0))    # every month on the 1st at 08:00  
MONTHLY(2)                   # every 2 months on the 1st at 00:00  
MONTHLY(1, -1)               # every month on the last day at 00:00  
```

---

### `MONTHLY_BY_WEEKDAY(frequency, weekday, occurrence, [TIME(hour, minute)])`

Repeats every `frequency` months on the specified weekday occurrence.

- `frequency`: required integer > 0.
- `weekday`: required string — one of `MO`, `TU`, `WE`, `TH`, `FR`, `SA`, `SU`.
- `occurrence`: required integer (e.g. `2` for "2nd weekday of the month").
- `TIME(hour, minute)`: optional time, defaults to `00:00`.

**Examples:**

```
MONTHLY_BY_WEEKDAY(1, MO, 2, TIME(15, 0))   # every month, 2nd Monday at 15:00  
MONTHLY_BY_WEEKDAY(3, FR, 1)               # every 3 months, 1st Friday at 00:00
```

---

### `YEARLY(frequency, [month, day], [TIME(hour, minute)])`

Every `frequency` years, optionally on a specific month/day pair and time.

- `frequency`: required integer.
- `month, day`: optional month (1–12) and day (1–31); both must be provided together.
- `TIME(hour, minute)`: optional, defaults to `00:00`.

**Examples:**
```
YEARLY(1, 12, 25, TIME(18, 0))    # every Christmas at 18:00  
YEARLY(2, 7, 4)                   # every other July 4 at 00:00  
YEARLY(3)                         # every 3 years on today’s date  
```

---

### `HOURLY(frequency_hour, frequency_minute)`

Every `frequency_hour` hours and `frequency_minute` minutes.

- `frequency_hour`: required integer (can be 0).
- `frequency_minute`: required integer (can be 0).

**Examples:**
```
HOURLY(1, 30)    # every 1 hour and 30 minutes  
HOURLY(0, 20)    # every 20 minutes  
```

---

## 🔁 Main Components

### `MiniRecurrenceConverterDSL`

A parser that converts symbolic DSL expressions into `RecurrenceRule` objects, e.g.:

```python
rule = MiniRecurrenceConverterDSL().parse("WEEKLY(1, [MO, WE], TIME(9, 0))")
```

### `RecurrenceRule`

A data class that holds structured recurrence information and implements `.next(datetime)` to compute the next occurrence.

#### Example:

```python
from datetime import datetime
rule = RecurrenceRule(unit=RecurrenceUnit.DAILY, frequency=1, hour=9, minute=0)
next_time = rule.next(datetime.now())
```

---

## 📦 Model-Assisted Natural Language Conversion

You can also generate DSL expressions from natural language using an LLM adapter:

```python
dsl_code, rule = parse_natural_recurrence_expression(
    "every second Tuesday at 5pm", container_name="my_container"
)
```

---

## 🧪 Testing & Coverage

The entire module is covered with unit tests:
- DSL parsing correctness (`test_parse_*`)
- Invalid argument handling
- Edge cases (e.g. leap years, last day of month)
- `.next()` computation for all units

Coverage as reported by `pytest --cov` exceeds 90%.

---

## ✅ Validation Rules

- Function names must be UPPERCASE (`weekly(...)` is invalid)
- All arguments are positional (no `frequency=1`)
- Invalid day, month, or weekday strings raise `ValueError`
- Empty/malformed DSL (e.g. `DAILY()`, `WEEKLY(1, foo)`) is rejected

---

## 🔄 Example DSL Inputs & Outputs

### Input DSL:
```
WEEKLY(2, [MO, FR], TIME(7, 0))
```

### Parsed Object:
```python
RecurrenceRule(
    unit=RecurrenceUnit.WEEKLY,
    frequency=2,
    days=[0, 4],
    hour=7,
    minute=0
)
```

### Next Occurrence:
```python
rule.next(datetime(2025, 5, 1))  # Computes the next valid datetime
```

---

## 🧱 Directory Structure

```
mini_recurrence_converter_dsl/
├── __init__.py
├── core.py
├── evaluate_mini_recurrence_converter_dsl_model.py
├── generate_synthetic_data.py
├── README.md
tests/
├── test_mini_recurrence_converter_dsl.py
```

---

## 📜 License

MIT — See [LICENSE](../../../LICENSE)
