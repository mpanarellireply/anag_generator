import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.models import FunctionSpec, ControlParam, Control, Variable
from src.excel_parser import RawFunctionData

PARSER_SYSTEM_PROMPT = """You are an expert PL/SQL developer analyzing COBOL-to-SQL migration documents.
Your job is to take raw data extracted from an Excel analysis spreadsheet and produce a structured
specification for a PL/SQL validation function.

The function follows this pattern:
- It receives input parameters and validates them
- Each validation is identified by a control code (e.g., NCD01249)
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
4. Identify unique controls from the rows (deduplicate by CODICE CONTROLLO)
5. For each control, extract: code, description, short_desc, error_code, long_desc
6. Set logic to a TODO placeholder with the control description
7. Identify any additional variables that might be needed (leave empty if unclear)

IMPORTANT:
- Skip rows where CODICE CONTROLLO is empty
- Clean up descriptions by removing trailing quotes and whitespace
- The function_description should summarize what the function validates

Respond ONLY with a valid JSON object matching this schema:
{{
    "function_name": "string",
    "function_description": "string",
    "parameters": [
        {{"name": "string", "direction": "IN", "type": "CHAR", "description": "string"}}
    ],
    "controls": [
        {{
            "code": "string",
            "description": "string",
            "short_desc": "string",
            "error_code": "string",
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

Control Rows:
{control_rows_json}

Remember:
- Deduplicate controls by CODICE CONTROLLO
- Skip rows with empty CODICE CONTROLLO
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

    def parse(self, raw_data: RawFunctionData) -> FunctionSpec:
        """Parse raw function data into a structured FunctionSpec."""
        control_rows = []
        for row in raw_data.rows:
            control_rows.append({
                "controllo": row.controllo,
                "codice_controllo": row.codice_controllo,
                "descrizione_controllo": row.descrizione_controllo,
                "codice_errore": row.codice_errore,
                "descrizione_errore": row.descrizione_errore,
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

        # Parse the LLM response as JSON
        content = response.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        data = json.loads(content)
        return FunctionSpec(**data)

    def parse_all(self, raw_data_list: list[RawFunctionData]) -> list[FunctionSpec]:
        """Parse all raw function data into FunctionSpec objects."""
        specs = []
        for raw_data in raw_data_list:
            try:
                spec = self.parse(raw_data)
                specs.append(spec)
                print(f"  [Parser] Parsed: {spec.function_name} "
                      f"({len(spec.parameters)} params, {len(spec.controls)} controls)")
            except Exception as e:
                print(f"  [Parser] ERROR parsing {raw_data.function_name}: {e}")
                print(e.__traceback__)
        return specs
