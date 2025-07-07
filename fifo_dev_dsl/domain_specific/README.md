# 🧠 Domain-Specific DSLs

This directory hosts modular DSL interpreters used to parse symbolic expressions into structured runtime objects. Each DSL supports a two-stage process:  
  1. natural language is translated into a DSL string using a fine-tuned LLM adapter, and  
  2. the DSL string is parsed into a structured Python object.

The parsing stage can also be used independently when a DSL expression is already available, with no need to invoke an LLM.

Each subdirectory targets a specific domain:

- [`mini_date_converter_dsl/`](mini_date_converter_dsl/README.md): converts natural language date and time expressions into symbolic DSL strings (e.g.,  
  `OFFSET(DATE_FROM_MONTH_WEEKDAY(11, 4, 4), 1, DAY)`), which are then parsed and evaluated into Python `datetime` objects.

- [`mini_recurrence_converter_dsl/`](mini_recurrence_converter_dsl/README.md): converts natural language recurrence patterns into symbolic DSL strings (e.g.,  
  `WEEKLY(2, [TU], TIME(17, 0))`), which are then parsed and evaluated into structured recurrence rule objects.

Each DSL is self-contained and comes with its own parser, test suite, and documentation.

> Shared utilities used by both DSLs can be found in [`common/`](common/).
