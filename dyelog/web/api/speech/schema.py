# Pydantic model for request body
from pydantic import BaseModel


class TextToSpeechInput(BaseModel):
    """Input model for text to speech."""

    text: str


class SpeechToTextResponse(BaseModel):
    """Response model for speech to text."""

    text: str
    confidence: float
