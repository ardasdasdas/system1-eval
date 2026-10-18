# Code-to-DMN Token Saver (`code2dmn`) Design Specification

## 1. Intent and Why

AI coding assistants (Cursor, Codex, Antigravity) consume large amounts of input and output tokens when repeatedly reading, reasoning about, and modifying deeply nested business rules (`if-else` chains, pricing tiers, tax/commission calculations, eligibility rules) embedded in legacy or sprawling source files.

This project builds an open-source, deterministic **Code-to-DMN Converter and MCP/CLI Toolkit** (`code2dmn`). It scans source code `if-else` structures using Abstract Syntax Tree (AST) analysis—spending **zero LLM tokens**—and transforms them into:
1. Standard **OMG DMN 1.3 XML (`.dmn`)** files, which can be executed directly by a lightweight DMN runner or published to future Open-Source Community Rule Repositories.
2. **Ultra-Compact Decision Tables (Markdown/CSV-like matrix)** served via CLI and Model Context Protocol (MCP) so AI coding agents can understand and verify complex branching logic using up to 70–90% fewer context tokens than reading raw source code.

Starting with the converter first solves the cold-start problem: developers can immediately reduce token costs on their own codebases today while generating standardized `.dmn` artifacts to seed shared community rule repositories tomorrow.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
1. **Zero-Token Deterministic Extraction:** Parse nested `if / elif / else`, `for` loops, `while` loops, and `try / except` structures from source code into an intermediate representation (IR) of decision rules without calling any LLM.
2. **Standard DMN 1.3 XML Generation:** Export extracted rules as valid OMG DMN 1.3 XML (`FIRST`, `COLLECT` / `C+`, iterative execution annotations, and exception-fallback rules) compatible with standard DMN viewers/engines and our local runtime evaluator.
3. **Agent Context Compression (MCP & CLI):** Provide an MCP server and CLI tool that returns an ultra-compact decision table representation alongside token-saving metrics (original code token count vs. compact table token count).
4. **Codebase Offloading & Runtime Evaluation:** Provide a lightweight, zero-LLM local DMN evaluator (`evaluate_dmn_file`) and refactoring stub generator so complex functions containing `if-else`, `for`, `while`, and `try/except` logic can be replaced by a `.dmn` file plus a 3-line evaluation call, eliminating future output-token regeneration costs.

### Non-Goals (v1)
1. **Natural Language / Legislation Text Parsing:** Extracting DMN tables from unstructured PDF/text regulations is deferred to Phase 2.
2. **Cloud Registry / Web Hosting:** The "Community Rule Repository" web platform is out of scope for v1; v1 focuses on generating clean, portable `.dmn` files and local rule catalogs that can live in Git.
3. **External I/O Side Effects:** Network socket calls, filesystem writes, and database commits cannot be executed inside pure DMN tables and will be flagged as external side effects.

### Anti-Goals (Failure Conditions Even If Technically Working)
1. **Using LLMs for the AST-to-DMN Conversion:** Calling an LLM to parse code into DMN defeats the core purpose of deterministic, zero-cost extraction.
2. **Dumping Raw DMN XML into Agent Context:** Standard DMN XML is verbose. Feeding raw XML to Cursor/Antigravity by default would *increase* token usage. Agents must receive the compact Markdown/matrix representation in context, while `.dmn` XML is written to disk.
3. **Silent Semantic Drift:** Never silently drop an `else` branch, loop accumulation, or `except` handler in a way that changes evaluation outcomes. Unbounded `while` loops must enforce a deterministic `max_iterations` guard at evaluation time.

---

## 3. Constraints

- **Zero External API Cost:** Must run 100% offline and locally with no API keys required.
- **Latency:** Parsing a source file and generating both `.dmn` XML and compact agent output must complete in `< 100ms` for typical files (< 2,000 LOC) so MCP tool calls feel instantaneous.
- **Minimal Dependencies:** Keep the dependency footprint small so developers can install and run the CLI or register the MCP server in Cursor/Antigravity in one command.

---

## 4. User-Decided Architecture & Workflow

1. **Converter First Strategy:** Build the extractor/converter first before attempting a community template repository, using the converter as the foundation.
2. **Primary Input:** Existing source code branching and control-flow structures (`if-else`, `for`, `while`, and `try/except`).
3. **Dual Token-Saving Workflow for Cursor / Codex / Antigravity:**
   - **Workflow A — Context Compression (Read Path):** Instead of reading a large source file into the LLM prompt, the agent invokes the MCP tool (or CLI) on a target file/function and receives a compact decision matrix plus token savings report.
   - **Workflow B — Codebase Refactoring (Write Path):** The tool writes a `.dmn` file to disk, evaluates inputs deterministically against that `.dmn` file at runtime, and generates a minimal replacement function that delegates control-flow logic to the `.dmn` table.

---

## 5. Technical Decisions (Builder's Calls / Sensible Defaults)

*(Note: The following technical defaults were selected by the architect to enable immediate implementation without ambiguity.)*

1. **Language & Stack (`Builder's Call`):**
   - **Python 3.11+** package named `code2dmn`.
   - Uses Python's standard library `ast` module for Python source parsing and `xml.etree.ElementTree` for DMN 1.3 XML generation and parsing.
2. **Control-Flow Mapping for `if-else`, `for`, `while`, and `try/except` (`Builder's Call`):**
   - **`if / elif / else` Chains:**
     - Mapped to a Decision Table with `hitPolicy="FIRST"`.
     - Nested `if` statements combine parent and child conditions via logical `AND`.
     - Compound `or` conditions across different input variables split into sequential rows with the same output; final `else` maps to a catch-all row (`-` for all inputs).
   - **`for` Loops (`for item in iterable:`):**
     - Mapped to an **Iterated Decision Table** (`iterationVariable="item"`, `collectionExpression="iterable"` stored in DMN `<extensionElements>` and compact header metadata):
       - **Accumulation loops** (`total += expr` inside `if/elif/else`): Mapped to `hitPolicy="COLLECT" aggregation="SUM"` (`C+`), where each matching item's rule output is summed (initial accumulator value preserved in metadata).
       - **Collection/filter loops** (`result.append(expr)` inside `if/elif/else`): Mapped to `hitPolicy="COLLECT"` (`C`), returning the list of matching rule outputs across all elements in `iterable`.
       - **Early-return search loops** (`if cond: return expr` inside `for`): Mapped to `hitPolicy="FIRST"` over the iterated collection, with any post-loop `return default` mapped as the fallback result if no item matches.
   - **`while` Loops (`while condition:`):**
     - Mapped to a **Stateful Iterative Decision Table** (`executionMode="WHILE"`, `whileCondition="<expr>"`, `maxIterations=1000` stored in DMN `<extensionElements>` and compact header metadata).
     - State variables initialized before the `while` loop (`state = ...`) and updated inside `if/elif/else` branches of the loop body become both input columns and output columns of the table. The DMN evaluator repeatedly applies the decision table to the state dictionary while `whileCondition` holds true, then evaluates the post-loop `return` expression.
   - **`try / except / else / finally` Blocks:**
     - The `try` suite (and optional `else` suite) is extracted as the primary rules of the Decision Table.
     - Each `except <ExceptionType>:` handler (e.g., `except ZeroDivisionError:`, `except (KeyError, ValueError):`, `except Exception:`) is extracted as an **Exception Fallback Rule** with a special input column `__error__` matching the exception type name (e.g., `"ZeroDivisionError"`, `"KeyError", "ValueError"`, or `"*"` for bare `except:` / `Exception`).
     - During runtime evaluation (`evaluate_dmn_file`), if evaluating a primary rule's condition or output expression raises an exception (or if an explicit `raise ExcType(...)` branch is hit), the evaluator catches the exception, binds `__error__ = type(exc).__name__`, and matches the corresponding exception fallback rule! If a `finally` block assigns/overrides a state variable, it is recorded as a `finallyExpression` applied before returning.
   - **Impurity Handling:** If a block contains unsupported external I/O (`await`, `yield`, `with open(...)`), the parser marks that function as `skipped_impure` with exact reason and line range.
3. **Compact Agent Format & Token Estimator (`Builder's Call`):**
   - The compact format uses a tight Markdown table with FEEL conditions and control-flow annotations (`HitPolicy`, `Iterates`, `While`, `ErrorFallback`):
     ```markdown
     # Decision: calculate_shipping (HitPolicy: FIRST, Mode: STANDARD)
     # Savings: 312 -> 58 tokens (81.4% saved)
     | # | country | order_total | is_vip | __error__ | => shipping_fee |
     |---|---------|-------------|--------|-----------|-----------------|
     | 1 | "TR"    | >= 500      | -      | -         | 0               |
     | 2 | "TR"    | < 500       | true   | -         | 19.99           |
     | 3 | "TR"    | < 500       | false  | -         | 49.99           |
     | 4 | -       | -           | -      | "ValueError" | -1           |
     | 5 | -       | -           | -      | -         | 149.99          |
     ```
   - Token counting uses a deterministic heuristic (`ceil(len(text) / 4)` tokens, plus word/symbol boundary weighting) with zero external network calls, reporting `original_tokens`, `compact_tokens`, `dmn_xml_tokens`, and `savings_pct`.
4. **DMN Runtime Evaluator & Refactoring Generator (`Builder's Call`):**
   - Includes a pure-Python DMN 1.3 table evaluator (`evaluate_dmn_file(dmn_path, decision_id, context_dict)`) that supports:
     - Standard `FIRST` and `UNIQUE` hit policies
     - `COLLECT` (`C`) and `COLLECT+SUM` (`C+`) hit policies with `for` collection iteration
     - Stateful `WHILE` loop iteration up to `maxIterations`
     - `try/except` runtime exception catching and `__error__` fallback rule routing
   - Includes a refactoring helper (`generate_refactored_code(file_path, function_name, dmn_path)`) that produces the replacement Python function calling `evaluate_dmn_file(...)`, verified to return identical outputs to the original function across test inputs.
5. **CLI & MCP Interface (`Builder's Call`):**
   - **CLI (`code2dmn`):**
     - `code2dmn inspect <file>` — Prints compact decision tables and token savings for all extractable functions in `<file>`.
     - `code2dmn export <file> -o <dir>` — Exports `.dmn` XML files and optionally refactored Python wrapper code.
     - `code2dmn eval <dmn_file> --input '{"country": "TR", "order_total": 600, "is_vip": false}'` — Evaluates a DMN file directly from CLI.
   - **MCP Server (`code2dmn-mcp` via JSON-RPC stdio):**
     - Exposes `inspect_decisions(file_path)`, `export_dmn(file_path, output_dir)`, and `evaluate_decision(dmn_path, inputs)` tools so Cursor, Codex, and Antigravity can invoke them natively.

---

## 6. What Is Left to the Builder

- Internal module layout within `src/code2dmn/` (e.g., separating `models.py`, `parser_py.py`, `dmn_xml.py`, `compact.py`, `evaluator.py`, `refactor.py`, `cli.py`, `mcp_server.py`).
- Exact XML element ID naming conventions and `<extensionElements>` schema for storing loop (`for`/`while`) and `try/except` metadata inside valid DMN 1.3 XML.
- Handling of complex `or` expressions in conditions: when an `if` condition has top-level `a or b` across different input variables, splitting into two sequential `FIRST`-hit rules with identical output is preferred because DMN table cells test one input variable per column.
