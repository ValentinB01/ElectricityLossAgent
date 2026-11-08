"""Jinja2 template environment for deterministic Romanian agent responses."""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).resolve().parent

jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(template_name: str, **context) -> str:
    """Render a Jinja2 template with context."""
    if not template_name.endswith(".j2"):
        template_name = f"{template_name}.j2"
    template = jinja_env.get_template(template_name)
    return template.render(**context).strip()

