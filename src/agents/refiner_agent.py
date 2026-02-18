import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.models import FunctionSpec, ReviewResult

logger = logging.getLogger(__name__)

STANDARD_RULES = """Follow also these standard rules:
- Input function parameters follow the pattern V_{{variable_name}}
- Do not use any DEFAULT for input function parameters
"""

REFINER_SYSTEM_PROMPT = """You are an expert PL/SQL developer. Your job is to fix structural and parsing errors
in generated SQL validation functions based on reviewer feedback.

You will receive:
1. The current SQL file content
2. A list of issues identified by the reviewer
3. The original function specification (parameters, controls, etc.)

Fix ONLY the issues listed by the reviewer. Do not change anything else.
Common fixes include:
- Missing or malformed CREATE OR REPLACE / IS / BEGIN / END blocks
- Incorrect parameter declarations (wrong type, missing comma, wrong direction)
- Missing control blocks
- Mismatched function names in CREATE, END, DELETE, INSERT statements
- Missing or incorrect DELETE/INSERT statements at the bottom
- Incorrect V_CERR / V_XERR / V_FLG_ATTIVO pattern in control blocks

{standard_rules}

Respond ONLY with the corrected SQL content. Do NOT wrap it in markdown code blocks.
Do NOT add any explanation before or after the SQL.
"""

REFINER_STANDALONE_SYSTEM_PROMPT = """You are an expert PL/SQL developer. Your job is to review and fix structural
and parsing errors in generated SQL validation functions.

You will receive:
1. The current SQL file content
2. The original function specification (parameters, controls, etc.)

Carefully check the SQL for common issues and fix any you find:
- Missing or malformed CREATE OR REPLACE / IS / BEGIN / END blocks
- Incorrect parameter declarations (wrong type, missing comma, wrong direction)
- Missing control blocks
- Mismatched function names in CREATE, END, DELETE, INSERT statements
- Missing or incorrect DELETE/INSERT statements at the bottom
- Incorrect V_CERR / V_XERR / V_FLG_ATTIVO pattern in control blocks
- Syntax errors or malformed PL/SQL constructs

If the SQL is already correct, return it unchanged.

{standard_rules}

Respond ONLY with the corrected SQL content. Do NOT wrap it in markdown code blocks.
Do NOT add any explanation before or after the SQL.
"""

REFINER_USER_PROMPT = """Fix the following SQL file based on the reviewer's feedback.

## Original Specification
Function: {function_name}
Parameters: {parameters_json}
Controls: {controls_json}

## Reviewer Issues
{issues_text}

## Current SQL
{sql_content}
"""

REFINER_STANDALONE_USER_PROMPT = """Review and fix any issues in the following SQL file.

## Original Specification
Function: {function_name}
Parameters: {parameters_json}
Controls: {controls_json}

## Current SQL
{sql_content}
"""


class RefinerAgent:
    """Agent that fixes structural/parsing errors in generated SQL, with or without review feedback."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", REFINER_SYSTEM_PROMPT.format(standard_rules=STANDARD_RULES)),
            ("human", REFINER_USER_PROMPT),
        ])
        self.standalone_prompt = ChatPromptTemplate.from_messages([
            ("system", REFINER_STANDALONE_SYSTEM_PROMPT.format(standard_rules=STANDARD_RULES)),
            ("human", REFINER_STANDALONE_USER_PROMPT),
        ])
        self.chain = self.prompt | self.llm
        self.standalone_chain = self.standalone_prompt | self.llm
        self.last_responses: dict[str, str] = {}

    def _build_spec_context(self, spec: FunctionSpec) -> dict:
        """Build common spec context for prompts."""
        params_json = json.dumps(
            [p.model_dump() for p in spec.parameters], indent=2, ensure_ascii=False
        )
        controls_json = json.dumps(
            [{"code": c.code, "description": c.description, "error_code": c.error_code,
              "short_desc": c.short_desc, "long_desc": c.long_desc}
             for c in spec.controls],
            indent=2, ensure_ascii=False,
        )
        return {"parameters_json": params_json, "controls_json": controls_json}

    def _clean_response(self, response_content: str, function_name: str) -> str:
        """Process LLM response and return cleaned SQL string."""
        self.last_responses[function_name] = response_content

        corrected = response_content.strip()
        # Strip markdown code blocks if the LLM wraps them anyway
        if corrected.startswith("```"):
            corrected = corrected.split("\n", 1)[1]
            corrected = corrected.rsplit("```", 1)[0]

        return corrected

    def refine(self, spec: FunctionSpec, sql_content: str, review: ReviewResult) -> str:
        """Refine SQL content based on reviewer feedback. Returns the corrected SQL string."""
        context = self._build_spec_context(spec)
        issues_text = "\n".join(f"- {issue}" for issue in review.issues)

        response = self.chain.invoke({
            "function_name": spec.function_name,
            **context,
            "issues_text": issues_text,
            "sql_content": sql_content,
        })

        return self._clean_response(response.content, spec.function_name)

    def refine_standalone(self, spec: FunctionSpec, sql_content: str) -> str:
        """Refine SQL content without reviewer feedback. Returns the corrected SQL string."""
        context = self._build_spec_context(spec)

        response = self.standalone_chain.invoke({
            "function_name": spec.function_name,
            **context,
            "sql_content": sql_content,
        })

        return self._clean_response(response.content, spec.function_name)

    def refine_all(
        self,
        specs: list[FunctionSpec],
        sql_map: dict[str, str],
        reviews: list[ReviewResult],
    ) -> dict[str, str]:
        """Refine all SQL contents that failed review. Returns dict of refined {name: sql}."""
        spec_map = {s.function_name: s for s in specs}

        refined = {}
        for review in reviews:
            if review.status != "FAIL":
                continue
            fname = review.function_name
            if fname not in spec_map or fname not in sql_map:
                continue

            try:
                refined[fname] = self.refine(spec_map[fname], sql_map[fname], review)
                logger.debug("[Refiner] Refined: %s", fname)
            except Exception as e:
                logger.error("[Refiner] ERROR refining %s: %s", fname, e)

        return refined

    def refine_all_standalone(
        self,
        specs: list[FunctionSpec],
        sql_map: dict[str, str],
    ) -> dict[str, str]:
        """Refine all SQL contents without reviewer feedback. Returns dict of refined {name: sql}."""
        spec_map = {s.function_name: s for s in specs}

        refined = {}
        for fname, sql_content in sql_map.items():
            if fname not in spec_map:
                continue
            try:
                refined[fname] = self.refine_standalone(spec_map[fname], sql_content)
                logger.debug("[Refiner] Refined (standalone): %s", fname)
            except Exception as e:
                logger.error("[Refiner] ERROR refining %s: %s", fname, e)

        return refined
