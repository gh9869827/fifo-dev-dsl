# `fifo_dev_dsl.domain_specific.mini_recurrence_converter_dsl`

This module defines a compact domain-specific language (DSL) for representing **recurring schedule references** (such as "every Monday at 9am") as symbolic function calls, which are evaluated into structured recurrence rules.

It includes:
- Functionality to call a fine-tuned LLM adapter that translates free-form natural language into DSL syntax (as text)
- A Python parser and evaluation engine that parses DSL text into symbolic function trees and computes the next occurrence

Together, these components support both human-readable input and structured recurrence logic.

---

## 🎯 Project Status & Audience

🚧 **Work in Progress** — Part of the **`fifo-dev-dsl`** project, currently in **early development**. 🚧

This is a personal project developed and maintained by a solo developer.  
Contributions, ideas, and feedback are welcome, but development is driven by personal time and priorities.

`fifo-dev-dsl` is designed to support other `fifo-*` projects developed by the author.  
It is provided for **individual developers** interested in experimenting with DSL-driven natural language interpretation.

No official release or pre-release has been published yet. The code is provided for **preview and experimentation**.  
**Use at your own risk.**

---

## 🎯 Purpose

Turn expressions like:

> "every Monday at 9am"  
> "every last day of the month"  
> "every second Friday of the month at 8:45am"

into structured recurrence rules that compute the next matching datetime using a symbolic, composable DSL.

---

## 🧠 Natural Language to DSL Conversion

The `mini_recurrence_converter_dsl` system supports translation of free-form recurrence expressions into symbolic DSL syntax through a fine-tuned language model.

The `parse_natural_recurrence_expression` function (in the `core` module) provides functionality to call a LoRA adapter that performs this conversion. The resulting DSL string is then parsed and evaluated into a structured recurrence rule, which can compute the next matching datetime.

The following pretrained adapter can be used with the `parse_natural_recurrence_expression` function:

### 📦 Pretrained Model  
[**mini-recurrence-converter-dsl-adapter**](https://huggingface.co/a6188466/mini-recurrence-converter-dsl-adapter)  
A **demonstration** fine-tuned LoRA adapter trained to convert English recurrence expressions into DSL syntax.

### 📚 Training Dataset  
[**mini-recurrence-converter-dsl-dataset**](https://huggingface.co/datasets/a6188466/mini-recurrence-converter-dsl-dataset)  
A **demonstration** dataset containing English recurrence expressions paired with DSL targets used for training and evaluation.

---

## 🚀 How to Use

### Evaluate a DSL string into a `RecurrenceRule` object:

```python
from datetime import datetime
from fifo_dev_dsl.domain_specific.mini_recurrence_converter_dsl.core import MiniRecurrenceConverterDSL

rule = MiniRecurrenceConverterDSL().parse("WEEKLY(1, [MO, WE], TIME(9, 0))")
print(rule.next(datetime(2025, 6, 2)))  # e.g., 2025-06-04 09:00:00 (next Monday or Wednesday)
```

### Translate natural language into DSL using an LLM adapter:

```python
from datetime import datetime
from fifo_dev_dsl.domain_specific.mini_recurrence_converter_dsl.core import parse_natural_recurrence_expression

dsl_code, rule = parse_natural_recurrence_expression("every other Tuesday at 5pm", model="phi")

print(dsl_code)
# Output: WEEKLY(2, [TU], TIME(17, 0))

# Start: Monday, June 2, 2025
print(rule.next(datetime(2025, 6, 2)))
# Output: 2025-06-03 17:00:00

# Start: Tuesday, June 3, 2025 (already scheduled day)
print(rule.next(datetime(2025, 6, 3)))
# Output: 2025-06-17 17:00:00
```

---

## 📘 DSL Syntax and Examples

This DSL uses symbolic, composable function calls to represent recurring schedules.  
All arguments are **positional only**. Keyword arguments are not supported.

---

### `DAILY(frequency, [TIME(hour, minute)])`

Repeats every `frequency` days, optionally at a specific time.

- `frequency`: Required integer.
- `TIME(hour, minute)`: Optional time (defaults to `00:00`).

**Examples:**
```
DAILY(1, TIME(9, 0))   # every day at 09:00  
DAILY(3)               # every 3 days at 00:00  
```

---

### `WEEKLY(frequency, [days], [TIME(hour, minute)])`

Repeats every `frequency` weeks, optionally on specific weekdays and at a specific time.

- `frequency`: Required integer.
- `days`: Optional list of weekday codes (e.g., `[MO, TU, FR]`).
- `TIME(hour, minute)`: Optional time (defaults to `00:00`).

**Examples:**
```
WEEKLY(1, [MO, WE], TIME(10, 0))   # every Monday and Wednesday at 10:00  
WEEKLY(2, [FR])                    # every other Friday at 00:00  
WEEKLY(3)                          # every 3 weeks with no weekday constraint  
```

---

### `MONTHLY(frequency, [day_of_month], [TIME(hour, minute)])`

Repeats every `frequency` months, optionally on a specific day and time.

- `frequency`: Required integer.
- `day_of_month`: Optional integer (1–31 or -1 for last day). Defaults to `1`.
- `TIME(hour, minute)`: Optional time (defaults to `00:00`).

**Examples:**
```
MONTHLY(1, 1, TIME(8, 0))   # every month on the 1st at 08:00  
MONTHLY(2)                  # every 2 months on the 1st at 00:00  
MONTHLY(1, -1)              # every month on the last day at 00:00  
```

---

### `MONTHLY_BY_WEEKDAY(frequency, weekday, occurrence, [TIME(hour, minute)])`

Repeats every `frequency` months on the specified weekday occurrence.

- `frequency`: Required integer.
- `weekday`: Required string — one of `MO`, `TU`, `WE`, `TH`, `FR`, `SA`, `SU`.
- `occurrence`: Required integer (e.g., `2` for "2nd weekday of the month").
- `TIME(hour, minute)`: Optional time (defaults to `00:00`).

**Examples:**
```
MONTHLY_BY_WEEKDAY(1, MO, 2, TIME(15, 0))   # every month, 2nd Monday at 15:00  
MONTHLY_BY_WEEKDAY(3, FR, 1)                # every 3 months, 1st Friday at 00:00  
```

---

### `YEARLY(frequency, [month, day], [TIME(hour, minute)])`

Repeats every `frequency` years, optionally on a specific month and day.

- `frequency`: Required integer.
- `month`, `day`: Optional (both must be provided). Month: 1–12, Day: 1–31.
- `TIME(hour, minute)`: Optional time (defaults to `00:00`).

**Examples:**
```
YEARLY(1, 12, 25, TIME(18, 0))   # every Christmas at 18:00  
YEARLY(2, 7, 4)                  # every other July 4 at 00:00  
YEARLY(3)                        # every 3 years on today’s date  
```

---

### `HOURLY(frequency_hour, frequency_minute)`

Repeats every `frequency_hour` hours and `frequency_minute` minutes.

- `frequency_hour`: Required integer (can be 0).
- `frequency_minute`: Required integer (can be 0).

**Examples:**
```
HOURLY(1, 30)   # every 1 hour and 30 minutes  
HOURLY(0, 20)   # every 20 minutes  
```

---

## ✅ Validation Rules

- Function names must be **UPPERCASE** (`weekly(...)` is invalid)
- All arguments are **positional only** (no `frequency=1`)
- Invalid values (e.g., unknown weekday codes, invalid days/months) raise `ValueError`
- Empty or malformed DSL (e.g., `DAILY()`, `WEEKLY(1, foo)`) is rejected

---

## 🧪 Testing & Coverage

To run tests:

```bash
pytest --cov
```

---

## 🔄 Example DSL Inputs & Outputs

### DSL Input:
```
WEEKLY(2, [MO, FR], TIME(7, 0))
```

### Parsed Python Object:
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
rule.next(datetime(2025, 5, 1))  # Computes the next matching datetime
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

## ✅ License

MIT — See [LICENSE](../../../LICENSE)
