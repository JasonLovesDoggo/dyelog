from fastapi import APIRouter, HTTPException
from google.cloud import texttospeech
from starlette.responses import Response

from dyelog.web.api.speech.schema import TextToSpeechInput

router = APIRouter()
# Initialize the Text-to-Speech client
client = texttospeech.TextToSpeechClient()
VOICE: texttospeech.SsmlVoiceGender = (
    texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED  # type: ignore
)


@router.post("/synthesize-speech")
async def synthesize_speech(input_data: TextToSpeechInput) -> Response:
    """Synthesizes speech from the input text and returns an audio file."""
    try:
        # Create a synthesis request
        synthesis_input = texttospeech.SynthesisInput(text=input_data.text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=VOICE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        return Response(
            content=response.audio_content,
            media_type="audio/mp3",
            status_code=200,
            headers={"Content-Disposition": "attachment; filename=speech.mp3"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error synthesizing speech: {e}",
        ) from None
