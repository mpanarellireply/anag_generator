import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.models import FunctionSpec, ReviewResult
from src.agents.convention import CONVENTION

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = """You are an expert PL/SQL code reviewer specializing in Oracle validation functions.
You review generated SQL files for correctness, completeness, and consistency with the source specification.

Your review should check:
1. STRUCTURE: The function has proper CREATE OR REPLACE, IS, BEGIN, EXCEPTION, END blocks
2. PARAMETERS: All expected parameters are declared with correct types and directions
3. CONTROLS: All control codes from the specification are present in the function body
4. INSERTS: The DELETE and INSERT statements at the bottom match the controls in the function
5. ERROR HANDLING: Each control properly uses V_CERR, V_XERR, V_FLG_ATTIVO pattern
6. CONSISTENCY: Function name matches in CREATE, END, DELETE, and INSERT statements

{convention}

Respond ONLY with a valid JSON object:
{{
    "function_name": "string",
    "status": "PASS or FAIL",
    "issues": ["list of issues found, empty if none"],
    "suggestions": ["list of improvement suggestions"]
}}
"""

REVIEWER_USER_PROMPT = """Review this generated SQL function against its specification.

## Specification
Function: {function_name}
Expected Parameters: {parameters_json}
Expected Controls: {controls_json}

## Generated SQL
```sql
{sql_content}
```

Provide your review as a JSON object.
"""


class ReviewerAgent:
    """Agent that reviews generated SQL using LLM-based analysis."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", REVIEWER_SYSTEM_PROMPT.format(convention=CONVENTION)),
            ("human", REVIEWER_USER_PROMPT),
        ])
        self.chain = self.prompt | self.llm
        self.last_responses: dict[str, str] = {}

    def review(self, spec: FunctionSpec, sql_content: str) -> ReviewResult:
        """Review a generated SQL against its specification."""
        params_json = json.dumps(
            [p.model_dump() for p in spec.parameters], indent=2, ensure_ascii=False
        )
        controls_json = json.dumps(
            [{"code": c.code, "description": c.description} for c in spec.controls],
            indent=2, ensure_ascii=False,
        )

        response = self.chain.invoke({
            "function_name": spec.function_name,
            "parameters_json": params_json,
            "controls_json": controls_json,
            "sql_content": sql_content,
        })

        self.last_responses[spec.function_name] = response.content

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        data = json.loads(content)
        return ReviewResult(**data)

    def review_all(
        self, specs: list[FunctionSpec], sql_map: dict[str, str]
    ) -> list[ReviewResult]:
        """Review all generated SQL contents."""
        spec_map = {s.function_name: s for s in specs}

        results = []
        for fname, sql_content in sql_map.items():
            if fname not in spec_map:
                results.append(ReviewResult(
                    function_name=fname,
                    status="FAIL",
                    issues=[f"No specification found for {fname}"],
                ))
                continue

            try:
                result = self.review(spec_map[fname], sql_content)
                results.append(result)
                status_icon = "OK" if result.status == "PASS" else "!!"
                logger.info("[Reviewer] %s %s: %s", status_icon, fname, result.status)
                if result.issues:
                    for issue in result.issues:
                        logger.info("[Reviewer]   - %s", issue)
            except Exception as e:
                logger.error("[Reviewer] ERROR reviewing %s: %s", fname, e)
                results.append(ReviewResult(
                    function_name=fname,
                    status="FAIL",
                    issues=[f"Review error: {str(e)}"],
                ))

        return results
