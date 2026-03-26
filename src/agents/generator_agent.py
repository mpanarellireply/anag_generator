import logging
from jinja2 import Environment, FileSystemLoader

from src.models import FunctionSpec

logger = logging.getLogger(__name__)


class GeneratorAgent:
    """Agent that renders FunctionSpec objects into SQL strings using the Jinja2 template."""

    def __init__(self, template_dir: str):
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            keep_trailing_newline=True,
        )
        self.template = self.env.get_template("template.j2")

    def generate(self, spec: FunctionSpec) -> str:
        """Render a FunctionSpec into a SQL string."""
        return self.template.render(
            function_name=spec.function_name,
            function_description=spec.function_description,
            parameters=[p.model_dump() for p in spec.parameters],
            controls=[c.model_dump() for c in spec.controls],
            variables=[v.model_dump() for v in spec.variables],
        )

    def generate_all(self, specs: list[FunctionSpec]) -> dict[str, str]:
        """Generate SQL for all FunctionSpec objects. Returns {function_name: sql_content}."""
        sql_map = {}
        for spec in specs:
            try:
                sql_content = self.generate(spec)
                sql_map[spec.function_name] = sql_content
                logger.debug("[Generator] Generated: %s", spec.function_name)
            except Exception as e:
                logger.error("[Generator] ERROR generating %s: %s", spec.function_name, e, exc_info=True)
        return sql_map
