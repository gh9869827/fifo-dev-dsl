# `mini_date_converter_dsl`

This module defines a minimal domain-specific language (DSL) for expressing natural language **date and time references** as symbolic function calls, which are then interpreted into `datetime` objects.

It includes a Python parser and evaluation engine, as well as a function to translate free-form expressions into DSL via an LLM adapter (`mini-date-converter-dsl`).

---

## 🧠 Purpose

Turn expressions like:

> "next Monday at 5pm"  
> "one day and two hours from now"  
> "the 4th Thursday of November"

into valid Python `datetime` objects using a structured and composable DSL.

---

## 🧩 Components

The `mini_date_converter_dsl` system consists of three main components:

1. **📦 Python Module**  
   The core DSL engine for parsing, resolving, and evaluating structured DSL trees.

2. **🧠 LoRA Adapter for Natural Language to DSL Conversion**  
   A fine-tuned language model that translates natural language date expressions into DSL expressions.  
   👉 [View Model on Hugging Face Hub](https://huggingface.co/your-model-link)

3. **📊 Training & Evaluation Dataset**  
   A curated dataset of English date expressions mapped to DSL syntax for model training and testing.  
   👉 [View Dataset on Hugging Face Hub](https://huggingface.co/datasets/a6188466/mini-date-converter-dsl-dataset)

---

## 🚀 How to Use

Evaluate a DSL string into a Python `datetime`:

```python
from fifo_dev_dsl.domain_specific.mini_date_converter_dsl import MiniDateConverterDSL

result = MiniDateConverterDSL().parse("SET_TIME(TODAY, 9, 30)")
print(result)  # e.g., 2025-06-02 09:30:00
```

Translate natural language into DSL using an LLM adapter:

```python
from fifo_dev_dsl.domain_specific.mini_date_converter_dsl.core import parse_natural_date_expression

dsl_code, date_time_object = parse_natural_date_expression("next Tuesday at 5pm", model="phi")
print(dsl_code)  # e.g., SET_TIME(OFFSET(TODAY, 1, WEEKDAY=1), 17, 0)
print(date_time_object)  # e.g., 2025-06-03 17:00:00
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
  - or `WEEKDAY=<0-6>` (where `0 = Monday`, `6 = Sunday`)

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

Returns the Nth weekday of a given month this year.

- `month`: integer (1–12)
- `weekday_index`: integer (0=Monday, ..., 6=Sunday)
- `occurrence`: integer (e.g., `4` for "4th weekday")

**Example:**

```dsl
DATE_FROM_MONTH_WEEKDAY(11, 3, 4)  # 4th Thursday of November
```

---

### `DATE_FROM_YEAR_MONTH_WEEKDAY(year, month, weekday_index, occurrence)`

Same as above, with an explicit year.

- `year`: four-digit year
- `month`: integer (1–12)
- `weekday_index`: integer (0=Monday, ..., 6=Sunday)
- `occurrence`: integer (e.g., `2` for "2nd weekday")

**Example:**

```dsl
DATE_FROM_YEAR_MONTH_WEEKDAY(2027, 5, 0, 2)  # 2nd Monday of May 2027
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
- Invalid argument values (e.g., `month=13`, `weekday_index=7`, `minute=75`) raise `ValueError`
- Invalid or non-existent dates (e.g., `February 30`, `April 31`) raise `ValueError`
- Missing arguments or malformed expressions also raise `ValueError` with a detailed message
- Nested expressions are fully supported and evaluated recursively
- Function names and units (e.g., `DAY`, `WEEKDAY=1`) are **case-sensitive**

---

## 🧪 Testing & Coverage

To run tests:

```bash
pytest --cov
```
