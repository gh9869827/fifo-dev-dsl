# 🧠 Domain-Specific DSLs

This directory hosts modular DSL interpreters used to parse symbolic expressions into structured runtime objects. Each subdirectory targets a specific domain:

- `mini_date_converter_dsl/`: Interpret symbolic date/time expressions into Python `datetime` values.
- `mini_recurrence_converter_dsl/`: Interpret recurring schedule logic into structured recurrence rules.

Each DSL is self-contained and comes with its own parser, test suite, and documentation.

> Shared utilities used by both DSLs can be found in `common/`.
