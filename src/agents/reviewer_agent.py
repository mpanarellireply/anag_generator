import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.models import FunctionSpec, ReviewResult

REVIEWER_SYSTEM_PROMPT = """You are an expert PL/SQL code reviewer specializing in Oracle validation functions.
You review generated SQL files for correctness, completeness, and consistency with the source specification.

Your review should check:
1. STRUCTURE: The function has proper CREATE OR REPLACE, IS, BEGIN, EXCEPTION, END blocks
2. PARAMETERS: All expected parameters are declared with correct types and directions
3. CONTROLS: All control codes from the specification are present in the function body
4. INSERTS: The DELETE and INSERT statements at the bottom match the controls in the function
5. ERROR HANDLING: Each control properly uses V_CERR, V_XERR, V_FLG_ATTIVO pattern
6. CONSISTENCY: Function name matches in CREATE, END, DELETE, and INSERT statements

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
    """Agent that reviews generated SQL files using LLM-based analysis."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("human", REVIEWER_USER_PROMPT),
        ])
        self.chain = self.prompt | self.llm

    def review(self, spec: FunctionSpec, sql_path: str) -> ReviewResult:
        """Review a generated SQL file against its specification."""
        with open(sql_path, "r", encoding="utf-8") as f:
            sql_content = f.read()

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

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        data = json.loads(content)
        return ReviewResult(**data)

    def review_all(
        self, specs: list[FunctionSpec], sql_paths: list[str]
    ) -> list[ReviewResult]:
        """Review all generated SQL files."""
        # Build a map of function_name -> spec
        spec_map = {s.function_name: s for s in specs}

        results = []
        for path in sql_paths:
            # Extract function name from filename
            import os
            fname = os.path.splitext(os.path.basename(path))[0]

            if fname not in spec_map:
                results.append(ReviewResult(
                    function_name=fname,
                    status="FAIL",
                    issues=[f"No specification found for {fname}"],
                ))
                continue

            try:
                result = self.review(spec_map[fname], path)
                results.append(result)
                status_icon = "OK" if result.status == "PASS" else "!!"
                print(f"  [Reviewer] {status_icon} {fname}: {result.status}")
                if result.issues:
                    for issue in result.issues:
                        print(f"             - {issue}")
            except Exception as e:
                print(f"  [Reviewer] ERROR reviewing {fname}: {e}")
                results.append(ReviewResult(
                    function_name=fname,
                    status="FAIL",
                    issues=[f"Review error: {str(e)}"],
                ))

        return results
