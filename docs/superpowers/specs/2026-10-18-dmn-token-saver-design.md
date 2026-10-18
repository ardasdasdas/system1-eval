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
1. **Zero-Token Deterministic Extraction:** Parse nested `if / elif / else` structures from source code into an intermediate representation (IR) of flattened decision rules without calling any LLM.
2. **Standard DMN 1.3 XML Generation:** Export extracted rules as valid OMG DMN 1.3 XML (`hitPolicy="FIRST"`, matching sequential `if-else` evaluation order) compatible with standard DMN viewers/engines (e.g., Camunda, `dmn-js`).
3. **Agent Context Compression (MCP & CLI):** Provide an MCP server and CLI tool that returns an ultra-compact decision table representation alongside token-saving metrics (original code token count vs. compact table token count).
4. **Codebase Offloading & Runtime Evaluation:** Provide a lightweight, zero-LLM local DMN evaluator (`evaluate_dmn`) and refactoring stub generator so a 150-line `if-else` function can be replaced by a `.dmn` file plus a 3-line evaluation call, eliminating future output-token regeneration costs.

### Non-Goals (v1)
1. **Natural Language / Legislation Text Parsing:** Extracting DMN tables from unstructured PDF/text regulations is deferred to Phase 2.
2. **Cloud Registry / Web Hosting:** The "Community Rule Repository" web platform is out of scope for v1; v1 focuses on generating clean, portable `.dmn` files and local rule catalogs that can live in Git.
3. **Full Imperative Program Equivalence:** Loops (`for`/`while`), async network calls, database mutations, and exception handlers inside `if` branches will not be converted into DMN.

### Anti-Goals (Failure Conditions Even If Technically Working)
1. **Using LLMs for the AST-to-DMN Conversion:** Calling an LLM to parse `if-else` code defeats the core purpose of deterministic, zero-cost extraction.
2. **Dumping Raw DMN XML into Agent Context:** Standard DMN XML is verbose. Feeding raw XML to Cursor/Antigravity by default would *increase* token usage. Agents must receive the compact Markdown/matrix representation in context, while `.dmn` XML is written to disk.
3. **Silent Semantic Drift:** Never silently drop an `else` branch or flatten conditions in a way that changes evaluation outcomes. Sequential `if / elif / else` must use `FIRST` hit policy (or explicitly negated prior conditions), and impure/unsupported branches must raise explicit diagnostics with line numbers.

---

## 3. Constraints

- **Zero External API Cost:** Must run 100% offline and locally with no API keys required.
- **Latency:** Parsing a source file and generating both `.dmn` XML and compact agent output must complete in `< 100ms` for typical files (< 2,000 LOC) so MCP tool calls feel instantaneous.
- **Minimal Dependencies:** Keep the dependency footprint small so developers can install and run the CLI or register the MCP server in Cursor/Antigravity in one command.

---

## 4. User-Decided Architecture & Workflow

1. **Converter First Strategy:** Build the extractor/converter first before attempting a community template repository, using the converter as the foundation.
2. **Primary Input:** Existing source code `if-else` branching structures.
3. **Dual Token-Saving Workflow for Cursor / Codex / Antigravity:**
   - **Workflow A — Context Compression (Read Path):** Instead of reading a large source file into the LLM prompt, the agent invokes the MCP tool (or CLI) on a target file/function and receives a compact decision matrix plus token savings report.
   - **Workflow B — Codebase Refactoring (Write Path):** The tool writes a `.dmn` file to disk, evaluates inputs deterministically against that `.dmn` file at runtime, and generates a minimal replacement function that delegates branching logic to the `.dmn` table.

---

## 5. Technical Decisions (Builder's Calls / Sensible Defaults)

*(Note: The following technical defaults were selected by the architect to enable immediate implementation without ambiguity.)*

1. **Language & Stack (`Builder's Call`):**
   - **Python 3.11+** package named `code2dmn`.
   - Uses Python's standard library `ast` module for Python source parsing and `xml.etree.ElementTree` for DMN 1.3 XML generation and parsing.
   - Provides a lightweight JS/TS conditional normalizer (handling standard `if (...) { return ... } else if (...)` blocks) or focuses primary deep AST support on Python `.py` files first with extensible parser interfaces. *(Specifically: v1 ships with a full Python `ast` parser and a clean `BaseCodeParser` interface, plus a basic JS/TS function-body branch extractor for common `if/else` return blocks).*
2. **Supported `if-else` Subset & Impurity Detection (`Builder's Call`):**
   - **Inputs:** Function arguments referenced in `if / elif` condition expressions.
   - **Conditions:** Comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`), boolean combinations (`and`, `or`, `not`), and boolean truthiness checks. Compound `or` conditions inside a single `if` either split into multiple rows (one per disjunct) or use FEEL-compatible disjunction expressions where valid; nested `if` statements combine parent and child conditions via logical `AND`.
   - **Outputs:** Branches terminating in `return <constant/expression>` or single output variable assignment (e.g., `discount = 0.15`).
   - **Hit Policy:** `FIRST` (`hitPolicy="FIRST"`), which preserves exact top-to-bottom `if / elif / else` evaluation semantics without requiring combinatorial negation of earlier branches. The final `else` branch maps to a catch-all row where all input entries are `-` (any).
   - **Impurity Handling:** If a candidate `if-else` block contains loops (`For`, `While`), `Try`, `With`, `Yield`, `Await`, or arbitrary statement sequences beyond pure assignments/returns, the parser marks that function as `skipped_impure` with exact reason and line range rather than emitting invalid DMN.
3. **Compact Agent Format & Token Estimator (`Builder's Call`):**
   - The compact format uses a tight Markdown table with FEEL conditions:
     ```markdown
     # Decision: calculate_shipping (HitPolicy: FIRST)
     # Savings: 312 -> 58 tokens (81.4% saved)
     | # | country | order_total | is_vip | => shipping_fee |
     |---|---------|-------------|--------|-----------------|
     | 1 | "TR"    | >= 500      | -      | 0               |
     | 2 | "TR"    | < 500       | true   | 19.99           |
     | 3 | "TR"    | < 500       | false  | 49.99           |
     | 4 | -       | -           | -      | 149.99          |
     ```
   - Token counting uses a deterministic heuristic (`ceil(len(text) / 4)` tokens, plus word/symbol boundary weighting) with zero external network calls, reporting `original_tokens`, `compact_tokens`, `dmn_xml_tokens`, and `savings_pct`.
4. **DMN Runtime Evaluator & Refactoring Generator (`Builder's Call`):**
   - Includes a pure-Python DMN 1.3 table evaluator (`evaluate_dmn_file(dmn_path, decision_id, context_dict)`) that evaluates FEEL unary tests (`>= 500`, `"TR"`, `true`, `-`, `[10..20]`, `"A", "B"`) with `FIRST` and `UNIQUE` hit policies.
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
- Exact XML element ID naming conventions (e.g., deterministic IDs like `Decision_calculate_shipping`, `Rule_1`, `Input_1`).
- Handling of complex `or` expressions in conditions: when an `if` condition has top-level `a or b` across different input variables, splitting into two sequential `FIRST`-hit rules with identical output is preferred because DMN table cells test one input variable per column.
