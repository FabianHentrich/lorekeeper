"""Models of the /prompts API: the four templates (system, qa, condense, no_context) and their variants.
"""
from pydantic import BaseModel


class PromptSet(BaseModel):
    """The four Jinja2 templates — variables per template: docs/prompts.md."""
    system: str
    qa: str
    condense: str
    no_context: str


class PromptVariant(BaseModel):
    """A named prompt set (a variant file, or the active prompts as ``active``)."""
    name: str
    description: str = ""
    prompts: PromptSet


class PromptVariantSummary(BaseModel):
    """List entry; ``is_active`` = its templates equal the active ones."""
    name: str
    description: str = ""
    is_active: bool = False


class RenderPreviewRequest(BaseModel):
    """Render a template text with sample variables, without saving anything."""
    template_text: str
    sample_data: dict = {}


class RenderPreviewResponse(BaseModel):
    """The rendered template."""
    rendered: str
