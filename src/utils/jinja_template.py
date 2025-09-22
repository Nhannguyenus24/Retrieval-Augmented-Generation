import os
from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "llm", "prompt")

jinja_env = Environment(
    loader=FileSystemLoader(PROMPT_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)

def render_template(template_name: str, **kwargs) -> str:
    """Render Jinja2 template"""
    try:
        template = jinja_env.get_template(template_name)
        return template.render(**kwargs)
    except Exception as e:
        print(f"Error rendering template {template_name}: {e}")
        return ""
