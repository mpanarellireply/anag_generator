import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

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
- Make all the comments uppercase
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

    def translate(self, sql_content: str, function_name: str = "") -> str:
        """Translate comments in SQL content to Italian. Returns the translated SQL string."""
        response = self.chain.invoke({"sql_content": sql_content})

        self.last_responses[function_name] = response.content

        translated = response.content.strip()
        if translated.startswith("```"):
            translated = translated.split("\n", 1)[1]
            translated = translated.rsplit("```", 1)[0]

        return translated

    def translate_all(self, sql_map: dict[str, str]) -> dict[str, str]:
        """Translate comments in all SQL contents. Returns updated {name: sql} dict."""
        result = {}
        for fname, sql_content in sql_map.items():
            try:
                result[fname] = self.translate(sql_content, function_name=fname)
                logger.debug("[Translator] Translated: %s", fname)
            except Exception as e:
                logger.error("[Translator] ERROR translating %s: %s", fname, e, exc_info=True)
                result[fname] = sql_content  # keep original on error
        return result
