from pydantic import BaseModel


class PromptSet(BaseModel):
    system: str
    qa: str
    condense: str
    no_context: str


class PromptVariant(BaseModel):
    name: str
    description: str = ""
    prompts: PromptSet


class PromptVariantSummary(BaseModel):
    name: str
    description: str = ""
    is_active: bool = False


class RenderPreviewRequest(BaseModel):
    template_text: str
    sample_data: dict = {}


class RenderPreviewResponse(BaseModel):
    rendered: str
