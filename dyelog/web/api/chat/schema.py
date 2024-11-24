from typing import List

from pydantic import BaseModel


class ChatInput(BaseModel):
    """Simple message model."""

    letter_ranges: str
    context: str


class PromptOption(BaseModel):
    id: int
    prompt: str


class ChatResponse(BaseModel):
    """The response model."""

    prompt_options: List[PromptOption]
