"""Renders the Jinja2 prompt templates (system, qa, condense, no_context) from config/prompts.yaml.

The prompt routes replace the whole instance on hot reload. Variables per template: docs/prompts.md.
"""
from pathlib import Path

import yaml
from jinja2 import Environment


class PromptManager:
    """Renders the Jinja2 prompt templates (system, qa, condense, no_context) loaded from YAML."""

    def __init__(self, prompts_path: Path | None = None, prompts_dict: dict | None = None):
        """Load templates either from a dict (hot-reload) or a YAML file path."""
        if prompts_dict:
            self.templates = prompts_dict
        elif prompts_path and prompts_path.exists():
            self.templates = yaml.safe_load(prompts_path.read_text(encoding="utf-8")) or {}
        else:
            self.templates = {}

        self.jinja_env = Environment()

    def render(self, template_name: str, **kwargs) -> str:
        """Render the named template with the given Jinja2 variables. Raises ``KeyError`` if unknown."""
        raw = self.templates.get(template_name)
        if raw is None:
            raise KeyError(f"Prompt template '{template_name}' not found")
        template = self.jinja_env.from_string(raw)
        return template.render(**kwargs)

    def get_system_prompt(self) -> str:
        """Return the raw ``system`` template, or an empty string if none is configured."""
        return self.templates.get("system", "")

    def render_qa(self, chunks: list[dict], question: str, history: list[dict] | None = None) -> str:
        """Render the ``qa`` template with retrieved chunks, the user question and the recent chat
        (``history``; templates without a history block simply ignore it)."""
        return self.render("qa", chunks=chunks, question=question, history=history or [])

    def render_condense(self, history: list[dict], question: str) -> str:
        """Render the ``condense`` template that rewrites a follow-up into a standalone query."""
        return self.render("condense", history=history, question=question)

    def render_no_context(self, question: str) -> str:
        """Render the ``no_context`` template used when retrieval finds nothing relevant."""
        return self.render("no_context", question=question)
