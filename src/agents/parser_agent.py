import json
import logging
import traceback
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.models import FunctionSpec, ControlParam, Control, Variable
from src.excel_parser import RawFunctionData

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """You are an expert PL/SQL developer analyzing COBOL-to-SQL migration documents.
Your job is to take raw data extracted from an Excel analysis spreadsheet and produce a structured
specification for a PL/SQL validation function.

The function follows this pattern:
- It receives input parameters and validates them
- Each validation control has a code (NCD...) and error code (NED...) that will be assigned separately
- The function returns a list of error codes separated by semicolons, or 'OK' if all checks pass

Given the raw data, you must:
1. Parse the PARAMETRI INPUT string to extract parameter names
2. Assign appropriate PL/SQL types based on naming conventions:
   - Names starting with V_F or V_FLG -> CHAR (flags)
   - Names starting with V_D followed by date-like name -> NUMBER (dates stored as numbers)
   - Names starting with V_I or containing IMPORT/LORDO -> NUMBER (amounts)
   - Names starting with V_C -> CHAR (codes)
   - Names starting with V_X -> CHAR (text/descriptions)
   - Names starting with V_V -> CHAR (values)
   - Names containing ARRAY -> CHAR_ARRAY (array types)
   - Names starting with V_N or V_COUNT -> NUMBER
   - Default -> CHAR
3. All parameters are IN direction unless otherwise specified
4. For each control row, generate:
   - "description": a concise description of what the control validates (based on the CONTROLLO column context)
   - "short_desc": a short label (max ~50 chars) for the NTC_DM_CTRL INSERT
   - "long_desc": a longer description for the NTC_DM_CTRL INSERT error message
5. Set logic to a TODO placeholder with the control description
6. Identify any additional variables that might be needed (leave empty if unclear)
7. The function_description should summarize what the function validates

IMPORTANT:
- Do NOT generate control codes (code) or error codes (error_code) — those will be assigned automatically
- Each row in the control rows represents one distinct control — do NOT deduplicate or skip any
- Clean up descriptions by removing trailing quotes and whitespace

Respond ONLY with a valid JSON object matching this schema:
{{
    "function_name": "string",
    "function_description": "string",
    "parameters": [
        {{"name": "string", "direction": "IN", "type": "CHAR", "description": "string"}}
    ],
    "controls": [
        {{
            "description": "string",
            "short_desc": "string",
            "long_desc": "string",
            "logic": "string"
        }}
    ],
    "variables": [
        {{"declaration": "string"}}
    ]
}}
"""

PARSER_USER_PROMPT = """Analyze this raw function data and produce a structured FunctionSpec:

Function Name: {function_name}
Operatività: {operativita}
Funzione (category): {funzione}
Parametri Input: {parametri_input}

Control Rows (each row is one control):
{control_rows_json}

Remember:
- Do NOT include "code" or "error_code" fields — they are assigned automatically
- Each row is a distinct control — include all of them, do not deduplicate or skip
- Generate "description", "short_desc", and "long_desc" based on the CONTROLLO column context
- Set logic to: "-- TODO: <description>\\n    NULL; -- IMPLEMENT VALIDATION LOGIC"
- Clean up description strings (remove trailing quotes, commas, whitespace)
- Prefix all parameter names with V_ if not already prefixed
"""


class ParserAgent:
    """Agent that structures raw Excel data into FunctionSpec objects using LLM."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PARSER_SYSTEM_PROMPT),
            ("human", PARSER_USER_PROMPT),
        ])
        self.chain = self.prompt | self.llm
        self.last_responses: dict[str, str] = {}

    @staticmethod
    def _assign_codes(spec: FunctionSpec, vertical_code: str, start_code: int) -> FunctionSpec:
        """Assign sequential NCD/NED codes to controls starting from start_code."""
        for i, control in enumerate(spec.controls):
            code_num = start_code + i
            control.code = f"NC{vertical_code}{code_num:05d}"
            control.error_code = f"NE{vertical_code}{code_num:05d}"
        return spec

    def parse(self, raw_data: RawFunctionData, vertical_code: str, start_code: int) -> FunctionSpec:
        """Parse raw function data into a structured FunctionSpec."""
        control_rows = []
        for row in raw_data.rows:
            control_rows.append({
                "controllo": row.controllo,
                "bloccante_warning": row.bloccante_warning,
                "messaggio_errore": row.messaggio_errore,
                "campo_impattato": row.campo_impattato,
            })

        response = self.chain.invoke({
            "function_name": raw_data.function_name,
            "operativita": raw_data.operativita,
            "funzione": raw_data.funzione,
            "parametri_input": raw_data.parametri_input,
            "control_rows_json": json.dumps(control_rows, indent=2, ensure_ascii=False),
        })

        self.last_responses[raw_data.function_name] = response.content

        # Parse the LLM response as JSON
        content = response.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        data = json.loads(content)
        spec = FunctionSpec(**data)

        # Assign sequential NCD/NED codes
        spec = self._assign_codes(spec, vertical_code, start_code)

        return spec

    def parse_all(
        self,
        raw_data_list: list[RawFunctionData],
        vertical_code: str,
        start_code: int = 1,
    ) -> list[FunctionSpec]:
        """Parse all raw function data into FunctionSpec objects.

        Each function gets its own sequential numbering starting from start_code.
        """
        specs = []
        progress_code = start_code
        for raw_data in raw_data_list:
            try:
                spec = self.parse(raw_data, vertical_code, progress_code)
                specs.append(spec)
                logger.debug("[Parser] Parsed: %s (%d params, %d controls, codes NC%s%05d-NC%s%05d)",
                             spec.function_name, len(spec.parameters), len(spec.controls),
                             vertical_code, progress_code, vertical_code, progress_code + len(spec.controls) - 1)
                progress_code += len(spec.controls)
            except Exception as e:
                logger.error("[Parser] ERROR parsing %s: %s", raw_data.function_name, e, exc_info=True)
                logger.debug(traceback.format_exc())
        return specs
