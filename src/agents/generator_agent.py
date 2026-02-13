import logging
import os
from jinja2 import Environment, FileSystemLoader

from src.models import FunctionSpec

logger = logging.getLogger(__name__)


class GeneratorAgent:
    """Agent that renders FunctionSpec objects into SQL files using the Jinja2 template."""

    def __init__(self, template_dir: str, output_dir: str):
        self.output_dir = output_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            keep_trailing_newline=True,
        )
        self.template = self.env.get_template("template.j2")
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, spec: FunctionSpec) -> str:
        """Render a FunctionSpec into a SQL file and return the file path."""
        rendered = self.template.render(
            function_name=spec.function_name,
            function_description=spec.function_description,
            parameters=[p.model_dump() for p in spec.parameters],
            controls=[c.model_dump() for c in spec.controls],
            variables=[v.model_dump() for v in spec.variables],
        )

        output_path = os.path.join(self.output_dir, f"{spec.function_name}.sql")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered)

        return output_path

    def generate_all(self, specs: list[FunctionSpec]) -> list[str]:
        """Generate SQL files for all FunctionSpec objects."""
        paths = []
        for spec in specs:
            try:
                path = self.generate(spec)
                paths.append(path)
                logger.debug("[Generator] Generated: %s", os.path.basename(path))
            except Exception as e:
                logger.error("[Generator] ERROR generating %s: %s", spec.function_name, e)
        return paths
