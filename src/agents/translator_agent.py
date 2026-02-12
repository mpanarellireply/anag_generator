import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

TRANSLATOR_SYSTEM_PROMPT = """You are a translator specializing in PL/SQL code comments.
Your job is to translate all comments in a SQL file from English to Italian.

Rules:
- Translate ONLY lines starting with -- (SQL comments)
- Do NOT modify any SQL code, variable names, string literals, or structure
- Do NOT translate identifiers, function names, or column names
- Do NOT translate content inside single quotes (string literals in SQL)
- Keep the -- prefix and the same indentation
- Respond ONLY with the complete SQL file with translated comments
- Do NOT wrap the output in markdown code blocks
"""

TRANSLATOR_USER_PROMPT = """Translate all SQL comments (lines starting with --) to Italian in this file:

{sql_content}
"""


class TranslatorAgent:
    """Agent that translates SQL comments from English to Italian."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", TRANSLATOR_SYSTEM_PROMPT),
            ("human", TRANSLATOR_USER_PROMPT),
        ])
        self.chain = self.prompt | self.llm
        self.last_responses: dict[str, str] = {}

    def translate(self, sql_path: str) -> str:
        """Translate comments in a SQL file to Italian. Returns the path."""
        with open(sql_path, "r", encoding="utf-8") as f:
            sql_content = f.read()

        response = self.chain.invoke({"sql_content": sql_content})

        fname = os.path.splitext(os.path.basename(sql_path))[0]
        self.last_responses[fname] = response.content

        translated = response.content.strip()
        if translated.startswith("```"):
            translated = translated.split("\n", 1)[1]
            translated = translated.rsplit("```", 1)[0]

        with open(sql_path, "w", encoding="utf-8") as f:
            f.write(translated)

        return sql_path

    def translate_all(self, sql_paths: list[str]) -> list[str]:
        """Translate comments in all SQL files."""
        translated = []
        for path in sql_paths:
            fname = os.path.splitext(os.path.basename(path))[0]
            try:
                self.translate(path)
                translated.append(path)
                print(f"  [Translator] Translated: {fname}")
            except Exception as e:
                print(f"  [Translator] ERROR translating {fname}: {e}")
        return translated
