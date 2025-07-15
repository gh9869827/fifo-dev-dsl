# 🧠 `fifo_dev_dsl.domain_specific`: Domain-Specific DSLs

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

## ✅ License

MIT — see [LICENSE](../../LICENSE).
