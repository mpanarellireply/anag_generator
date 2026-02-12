# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent pipeline that generates PL/SQL validation functions from Excel analysis documents. Uses LangChain + OpenAI to parse COBOL-to-SQL migration specs, render SQL via Jinja2 templates, complete validation logic, and review/refine output.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Process all functions from Excel
python main.py NEBULA_ANG_ControlliB2B.xlsx

# Process a single function
python main.py NEBULA_ANG_ControlliB2B.xlsx --function NFD_CTRL_FBROK

# Force re-parse (ignore cache)
python main.py NEBULA_ANG_ControlliB2B.xlsx --force-parse

# Skip pipeline steps
python main.py NEBULA_ANG_ControlliB2B.xlsx --skip-logic --skip-review --skip-refine

# Use reference SQL for style guidance
python main.py NEBULA_ANG_ControlliB2B.xlsx --example-sql NFD_CTRL_FBROK.sql

# Control refine iterations (default 3)
python main.py NEBULA_ANG_ControlliB2B.xlsx --max-refine 5

# Set starting control code number
python main.py NEBULA_ANG_ControlliB2B.xlsx --start-code 1224
```

## Architecture

Seven-phase pipeline coordinated by `src/orchestrator.py`:

```
Excel → Parse → Generate → Logic → Review → Refine → Translate → SQL
```

1. **Excel Parser** (`src/excel_parser.py`) — Reads Excel, groups rows by function name (`FUNZIONE.1`) into `RawFunctionData`
2. **Parser Agent** (`src/agents/parser_agent.py`) — LLM structures raw data into `FunctionSpec`, infers PL/SQL types from naming conventions, assigns sequential NCD/NED codes. Results cached in `parser_cache/`
3. **Generator Agent** (`src/agents/generator_agent.py`) — Renders `FunctionSpec` into SQL via `template.j2`, producing TODO placeholders for logic
4. **Logic Agent** (`src/agents/logic_agent.py`) — LLM replaces TODO placeholders with actual PL/SQL validation logic
5. **Reviewer Agent** (`src/agents/reviewer_agent.py`) — LLM reviews SQL structure, parameters, control codes, INSERT/DELETE correctness
6. **Refiner Agent** (`src/agents/refiner_agent.py`) — Iteratively fixes issues from reviewer feedback (re-reviews after each pass)
7. **Translator Agent** (`src/agents/translator_agent.py`) — Translates comments from English to Italian, preserving all code

## Key Data Models (`src/models.py`)

- **ControlParam** — Function parameter (name, direction, type, description)
- **Variable** — Local variable declaration
- **Control** — Validation control (code, description, logic, error messages)
- **FunctionSpec** — Complete spec for one PL/SQL function
- **ReviewResult** — Review feedback (status, issues, suggestions)

## Important Conventions

- Default LLM model: `gpt-4o-mini` with temperature 0 (configurable via `--model`)
- API key loaded from `.env` via python-dotenv
- Parsed specs are cached incrementally in `parser_cache/parsed_specs.json` — new functions merge with existing cache
- Debug output saved per-function per-phase in `debug/FUNCTION_NAME_TIMESTAMP/` (numbered files: `01_parser_*`, `02_logic_*`, etc.)
- Generated SQL follows a strict pattern: CREATE FUNCTION → IS → BEGIN → control loop → EXCEPTION → END, with NTC_DM_CTRL table operations
- PL/SQL type inference uses prefix naming conventions (V_F*/V_FLG* → CHAR, V_I*/*IMPORT* → NUMBER, V_D* → NUMBER for dates, etc.)
