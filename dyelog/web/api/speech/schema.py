# Pydantic model for request body
from pydantic import BaseModel


class TextToSpeechInput(BaseModel):
    """Input model for text to speech."""

    text: str
