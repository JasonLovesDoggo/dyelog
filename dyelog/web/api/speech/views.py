from fastapi import APIRouter, File, HTTPException, UploadFile
from google.cloud import speech, texttospeech
from starlette.responses import Response

from dyelog.web.api.speech.schema import SpeechToTextResponse, TextToSpeechInput

router = APIRouter()
# Initialize the Text-to-Speech tts_client
tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()
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

        response = tts_client.synthesize_speech(
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


@router.post("/transcribe-speech", response_model=SpeechToTextResponse)
async def transcribe_speech(
    audio_file: UploadFile = File(...),
    language_code: str = "en-US",
) -> SpeechToTextResponse:
    """Transcribes speech from an audio file and returns the text."""
    try:
        # Read the audio file
        content = await audio_file.read()

        # Configure the recognition request
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=16000,  # You might want to make this configurable
            language_code=language_code,
            enable_automatic_punctuation=True,
        )

        # Perform the transcription
        response = stt_client.recognize(config=config, audio=audio)

        # Get the most likely transcription
        if not response.results:
            raise HTTPException(
                status_code=400,
                detail="No speech could be recognized in the audio file",
            )

        transcript = response.results[0].alternatives[0]

        return SpeechToTextResponse(
            text=transcript.transcript,
            confidence=transcript.confidence,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error transcribing speech: {e}",
        ) from None
