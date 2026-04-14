from pathlib import Path

import yaml
from jinja2 import Environment


class PromptManager:
    """Manages and renders Jinja2 prompt templates used to format LLM generation inputs.

    This class handles rendering variable replacements mapping the retrieval chunks,
    chat history, and the system persona text into cohesive string blobs ready to be
    passed onto the chosen LLM provider.
    """

    def __init__(self, prompts_path: Path | None = None, prompts_dict: dict | None = None):
        if prompts_dict:
            self.templates = prompts_dict
        elif prompts_path and prompts_path.exists():
            self.templates = yaml.safe_load(prompts_path.read_text(encoding="utf-8")) or {}
        else:
            self.templates = {}

        self.jinja_env = Environment()

    def render(self, template_name: str, **kwargs) -> str:
        """Render a specifically named template using any provided keyword arguments.

        Fetches the matching template blob from the loaded prompts YAML configuration
        and parses it dynamically with Jinja2 string interpolation. Raises KeyError
        if the template does not exist.
        """
        raw = self.templates.get(template_name)
        if raw is None:
            raise KeyError(f"Prompt template '{template_name}' not found")
        template = self.jinja_env.from_string(raw)
        return template.render(**kwargs)

    def get_system_prompt(self) -> str:
        """Fetch the basic system persona instruction blob, normally 'system'."""
        return self.templates.get("system", "")

    def render_qa(self, chunks: list[dict], question: str) -> str:
        """Render the 'qa' template by providing context chunks and a question string."""
        return self.render("qa", chunks=chunks, question=question)

    def render_condense(self, history: list[dict], question: str) -> str:
        """Render the 'condense' template for summarizing session history + question."""
        return self.render("condense", history=history, question=question)

    def render_no_context(self, question: str) -> str:
        """Render the 'no_context' template when similarity search finds no answers."""
        return self.render("no_context", question=question)
