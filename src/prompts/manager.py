from pathlib import Path

import yaml
from jinja2 import Environment


class PromptManager:
    def __init__(self, prompts_path: Path | None = None, prompts_dict: dict | None = None):
        if prompts_dict:
            self.templates = prompts_dict
        elif prompts_path and prompts_path.exists():
            self.templates = yaml.safe_load(prompts_path.read_text(encoding="utf-8")) or {}
        else:
            self.templates = {}

        self.jinja_env = Environment()

    def render(self, template_name: str, **kwargs) -> str:
        raw = self.templates.get(template_name)
        if raw is None:
            raise KeyError(f"Prompt template '{template_name}' not found")
        template = self.jinja_env.from_string(raw)
        return template.render(**kwargs)

    def get_system_prompt(self) -> str:
        return self.templates.get("system", "")

    def render_qa(self, chunks: list[dict], question: str) -> str:
        return self.render("qa", chunks=chunks, question=question)

    def render_condense(self, history: list[dict], question: str) -> str:
        return self.render("condense", history=history, question=question)

    def render_no_context(self, question: str) -> str:
        return self.render("no_context", question=question)
