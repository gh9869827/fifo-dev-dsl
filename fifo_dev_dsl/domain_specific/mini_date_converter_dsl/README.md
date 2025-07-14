# `mini_date_converter_dsl`

This module defines a compact domain-specific language (DSL) for representing natural language **date and time references** as symbolic function calls, which are evaluated into Python `datetime` objects.

It includes:
- Functionality to call a fine-tuned LLM adapter that translates free-form natural language into DSL syntax (as text)
- A Python parser and evaluation engine that parses DSL text into symbolic function trees and evaluates them into Python `datetime` objects

Together, these components support both human-readable input and structured execution.

---

## 🎯 Purpose

Turn expressions like:

> "next Monday at 5pm"  
> "one day and two hours from now"  
> "the 4th Thursday of November"

into valid Python `datetime` objects using a structured, composable DSL.

---

## 🧠 Natural Language to DSL Conversion

The `mini_date_converter_dsl` system supports translation of free-form date expressions into symbolic DSL syntax through a fine-tuned language model.

The `parse_natural_date_expression` function (in the `core` module) provides functionality to call a LoRA adapter that performs this conversion. The resulting DSL string is then parsed and evaluated into a Python `datetime` object by this module.

The following pretrained adapter can be used with the `parse_natural_date_expression` function:

### 📦 Pretrained Model  
[**mini-date-converter-dsl-adapter**](https://huggingface.co/a6188466/mini-date-converter-dsl-adapter)  
A **demonstration** fine-tuned LoRA adapter trained to convert English date expressions into DSL syntax.

### 📚 Training Dataset  
[**mini-date-converter-dsl-dataset**](https://huggingface.co/datasets/a6188466/mini-date-converter-dsl-dataset)  
A **demonstration** dataset containing English date expressions paired with DSL targets used for training and evaluation.

---

## 🚀 How to Use

### Evaluate a DSL string into a Python `datetime`:

```python
from datetime import datetime
from fifo_dev_dsl.domain_specific.mini_date_converter_dsl.core import MiniDateConverterDSL

result = MiniDateConverterDSL(datetime(2025, 6, 2)).parse("SET_TIME(TODAY, 9, 30)")
print(result)
# Output: 2025-06-02 09:30:00
```

### Translate natural language into DSL using an LLM adapter:

```python
from fifo_dev_dsl.domain_specific.mini_date_converter_dsl.core import parse_natural_date_expression

dsl_code, date_time_object = parse_natural_date_expression("next Tuesday at 5pm", model="phi")

print(dsl_code)
# Output: SET_TIME(OFFSET(TODAY, 1, WEEKDAY=1), 17, 0)

print(date_time_object)
# Output: 2025-06-03 17:00:00
```

---

## 📘 Supported DSL Functions

This DSL defines symbolic, composable functions for expressing complex temporal logic derived from natural language. All arguments are positional. Nesting is fully supported, allowing expressions such as:

```dsl
SET_TIME(
  OFFSET(
    DATE_FROM_MONTH_WEEKDAY(11, 4, 4),
    1,
    DAY
  ),
  9, 30
)
```

This evaluates to 9:30 AM on the day after the 4th Thursday of November — e.g., the Friday after U.S. Thanksgiving.

---

### `TODAY`

Returns the current date (time set to `00:00`).

**Example:**

```dsl
TODAY
```

---

### `OFFSET(base_expr, value, unit)`

Adds or subtracts a time offset from a base expression.

- `base_expr`: Required DSL expression that resolves to a date.
- `value`: Required integer offset (can be negative).
- `unit`: One of:
  - `DAY`, `WEEK`, `MONTH`, `YEAR`
  - or `WEEKDAY=<0-6>` (where `0 = Monday`, ..., `6 = Sunday`)

**Examples:**

```dsl
OFFSET(TODAY, 2, DAY)
OFFSET(DATE_FROM_MONTH_DAY(12, 25), 1, YEAR)
```

---

### `DATE_FROM_MONTH_DAY(month, day)`

Builds a date using the current year and provided `month` and `day`.  
If the date has passed, uses the next available year.

- `month`: integer (1–12)
- `day`: integer (1–31)

**Example:**

```dsl
DATE_FROM_MONTH_DAY(12, 25)
```

---

### `DATE_FROM_YEAR_MONTH_DAY(year, month, day)`

Constructs a specific date.

- `year`: four-digit year
- `month`: integer (1–12)
- `day`: integer (1–31)

**Example:**

```dsl
DATE_FROM_YEAR_MONTH_DAY(2026, 1, 1)
```

---

### `DATE_FROM_MONTH_WEEKDAY(month, weekday_index, occurrence)`

Returns the Nth weekday of a given month this year. `occurrence` can be
negative to count backward from the end of the month (`-1` is the last weekday,
`-2` the second to last, etc.). If the resulting date has already
passed this year, the same occurrence of that weekday in the following
year is returned.

- `month`: integer (1–12)
- `weekday_index`: integer (0=Monday, ..., 6=Sunday)
- `occurrence`: integer index of the weekday (positive or negative)

**Example:**

```dsl
DATE_FROM_MONTH_WEEKDAY(11, 3, 4)   # 4th Thursday of November
DATE_FROM_MONTH_WEEKDAY(10, 4, -1)  # last Friday of October
```

---

### `DATE_FROM_YEAR_MONTH_WEEKDAY(year, month, weekday_index, occurrence)`

Same as above, with an explicit year.

- `year`: four-digit year
- `month`: integer (1–12)
- `weekday_index`: integer (0=Monday, ..., 6=Sunday)
- `occurrence`: integer index of the weekday (positive or negative)

**Example:**

```dsl
DATE_FROM_YEAR_MONTH_WEEKDAY(2027, 5, 0, 2)   # 2nd Monday of May 2027
DATE_FROM_YEAR_MONTH_WEEKDAY(2026, 10, 4, -1) # last Friday of October 2026
```

---

### `SET_MONTH_DAY(date_expr, day)`

Sets the day of the month on a base expression. Negative values index from the
end of the month (`-1` is the last day, `-2` two days before the last, etc.).

- `date_expr`: DSL expression returning a date
- `day`: integer (supports negative indexing)

**Examples:**

```dsl
SET_MONTH_DAY(OFFSET(TODAY, 1, MONTH), 1)
SET_MONTH_DAY(TODAY, -1)
```

---

### `SET_TIME(date_expr, hour, minute)`

Sets the time on a date expression.

- `date_expr`: DSL expression returning a date
- `hour`: integer (0–23)
- `minute`: integer (0–59)

**Example:**

```dsl
SET_TIME(TODAY, 17, 30)
```

---

### `OFFSET_TIME(date_expr, hours, minutes)`

Applies a relative time shift to an existing datetime.

- `date_expr`: DSL expression returning a date or datetime
- `hours`: integer (positive or negative)
- `minutes`: integer (positive or negative)

**Examples:**

```dsl
OFFSET_TIME(TODAY, 2, 45)
OFFSET_TIME(SET_TIME(TODAY, 12, 0), 0, 30)
```

---

## ✅ Validation Rules

- Function names must be **uppercase** (e.g., `OFFSET`, not `offset`)
- All arguments are **positional** and must follow the documented order
- Argument values must be valid (e.g., `month=13`, `weekday_index=7`, `minute=75` will raise `ValueError`)
- Invalid or non-existent dates (e.g., `February 30`, `April 31`) raise `ValueError`
- Missing arguments or malformed expressions raise `ValueError` with a detailed message
- Nested expressions are fully supported and evaluated recursively
- Function names and units (e.g., `DAY`, `WEEKDAY=1`) are **case-sensitive**

---

## 🧪 Testing & Coverage

To run tests:

```bash
pytest --cov
```

## 🧱 Directory Structure

```
mini_date_converter_dsl/
├── __init__.py
├── core.py
├── evaluate_mini_date_converter_dsl_model.py
├── README.md
tests/
├── test_mini_date_converter_dsl.py
```

---

## ✅ License

MIT — See [LICENSE](../../../LICENSE)
