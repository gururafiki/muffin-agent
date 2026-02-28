"""Jinja2 template loading for LLM prompts."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = Path(__file__).parent


def render_template(template_name: str, **kwargs: object) -> str:
    """Load and render a Jinja2 template from the prompts directory.

    Args:
        template_name: Template filename relative to prompts/
            (e.g. "technical_analyst.jinja").
        **kwargs: Variables to pass to the template.

    Returns:
        The rendered prompt string.
    """
    env = Environment(loader=FileSystemLoader(PROMPTS_DIR), keep_trailing_newline=True)
    template = env.get_template(template_name)
    return template.render(**kwargs)
