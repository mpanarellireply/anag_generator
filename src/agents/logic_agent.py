import json
import logging
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.models import FunctionSpec

logger = logging.getLogger(__name__)

LOGIC_SYSTEM_PROMPT = """You are an expert PL/SQL developer specializing in Oracle validation functions.
Your job is to replace TODO placeholders in generated SQL validation functions with high-level skeleton logic.

IMPORTANT CONTEXT: We do NOT have access to the real database data model. You must NOT write concrete
table names or column names in SELECT/FROM clauses. Instead, produce skeleton queries where:
- SELECT and FROM use descriptive pseudo-language placeholders that hint at the real entity being queried
- The WHERE clause is inferred from the control description, function parameters, and any example SQL provided

Each TODO block sits inside an IF V_FLG_ATTIVO = 'Y' THEN ... END IF; block for a specific control.
You must infer the validation logic from:
- The control description (what the check validates)
- The function parameters (what input data is available)
- The error pattern: when a check fails, append to the error list with:
  V_ERR_LIST := V_ERR_LIST || V_CERR || ' - ' || V_XERR || ';';

Skeleton query format for validations that require a database lookup:
    SELECT <description_of_what_to_retrieve>
      INTO <target_variable>
      FROM <description_of_source_table_or_entity>
     WHERE <inferred_condition_based_on_parameters>;

Example skeleton queries:
- SELECT <flag_broker_validity> INTO V_COUNT FROM <broker_reference_table> WHERE BROKER_CODE = V_FBROK AND STATUS = 'ACTIVE';
- SELECT <count_matching_records> INTO V_COUNT FROM <product_catalog_table> WHERE PRODUCT_TYPE = V_TIPOPROD AND TRIM(CODE) IS NOT NULL;

For simple validations that do NOT require a database lookup (null checks, flag checks, cross-field checks),
write the logic directly without a skeleton query:
- Flag checks: IF TRIM(V_FLAG) IS NOT NULL AND TRIM(V_FLAG) NOT IN ('S','N') THEN append error
- Required field: IF TRIM(V_FIELD) IS NULL THEN append error
- Numeric range: IF V_NUM < 0 OR V_NUM > 100 THEN append error
- Date checks: IF V_DATE IS NOT NULL AND V_DATE > TO_NUMBER(TO_CHAR(SYSDATE,'YYYYMMDD')) THEN append error
- Cross-field: IF V_A = 'S' AND TRIM(V_B) IS NULL THEN append error

IMPORTANT:
- Replace ONLY the TODO placeholder lines with actual logic
- Keep everything else in the SQL file unchanged (structure, comments, variable declarations, etc.)
- Each validation should be a simple IF ... THEN ... END IF; block
- The error append pattern is ALWAYS: V_ERR_LIST := V_ERR_LIST || V_CERR || ' - ' || V_XERR || ';';
- SELECT/FROM placeholders MUST use angle brackets with descriptive pseudo-names (e.g. <broker_ref_table>)
- WHERE clauses should use real parameter variable names (e.g. V_FBROK, V_TIPOPROD) since those are known
- Do NOT wrap the output in markdown code blocks
- Respond ONLY with the complete corrected SQL file
{example_section}"""

LOGIC_USER_PROMPT = """Replace all TODO placeholders in this SQL file with high-level skeleton validation logic.
For database lookups, use descriptive pseudo-placeholders in SELECT/FROM and infer the WHERE clause from the available information.

## Function Specification
Function: {function_name}
Description: {function_description}
Parameters: {parameters_json}
Controls: {controls_json}

## Current SQL (with TODOs to replace)
{sql_content}
"""


class LogicAgent:
    """Agent that replaces TODO placeholders in generated SQL with high-level skeleton validation logic.
    SELECT/FROM clauses use pseudo-language placeholders; WHERE clauses are inferred from available context."""

    def __init__(self, llm: ChatOpenAI, example_sql_path: str | None = None):
        self.llm = llm
        self.last_responses: dict[str, str] = {}
        self.example_sql = None
        if example_sql_path and os.path.exists(example_sql_path):
            with open(example_sql_path, "r", encoding="utf-8") as f:
                self.example_sql = f.read()

    def _build_chain(self):
        example_section = ""
        if self.example_sql:
            example_section = (
                "\n\nHere is a reference example of a completed validation function. "
                "Use it as a style guide for the logic you generate:\n\n"
                f"```sql\n{self.example_sql}\n```"
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", LOGIC_SYSTEM_PROMPT.format(example_section=example_section)),
            ("human", LOGIC_USER_PROMPT),
        ])
        return prompt | self.llm

    def complete(self, spec: FunctionSpec, sql_content: str) -> str:
        """Replace TODO placeholders in SQL content with actual logic. Returns the completed SQL string."""
        # Skip if no TODOs remain
        if "TODO" not in sql_content:
            return sql_content

        params_json = json.dumps(
            [p.model_dump() for p in spec.parameters], indent=2, ensure_ascii=False
        )
        controls_json = json.dumps(
            [{"code": c.code, "description": c.description, "error_code": c.error_code}
             for c in spec.controls],
            indent=2, ensure_ascii=False,
        )

        chain = self._build_chain()
        response = chain.invoke({
            "function_name": spec.function_name,
            "function_description": spec.function_description,
            "parameters_json": params_json,
            "controls_json": controls_json,
            "sql_content": sql_content,
        })

        self.last_responses[spec.function_name] = response.content

        completed = response.content.strip()
        # Strip markdown code blocks if the LLM wraps them
        if completed.startswith("```"):
            completed = completed.split("\n", 1)[1]
            completed = completed.rsplit("```", 1)[0]

        return completed

    def complete_all(self, specs: list[FunctionSpec], sql_map: dict[str, str]) -> dict[str, str]:
        """Complete TODO logic in all SQL contents. Returns updated sql_map."""
        result = dict(sql_map)

        for fname, sql_content in sql_map.items():
            spec = next((s for s in specs if s.function_name == fname), None)
            if spec is None:
                continue
            try:
                result[fname] = self.complete(spec, sql_content)
                logger.debug("[Logic] Completed: %s", fname)
            except Exception as e:
                logger.error("[Logic] ERROR completing %s: %s", fname, e, exc_info=True)

        return result
