import enum
from pathlib import Path
from tempfile import gettempdir

from google.cloud import texttospeech
from pydantic_settings import BaseSettings, SettingsConfigDict

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b-instruct-fp16"
    voice: int = texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED
    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.DEBUG
    words_file: Path = Path("data/words_alpha.txt")
    # This variable is used to define
    # multiproc_dir. It's required for [uvi|guni]corn projects.
    prometheus_dir: Path = TEMP_DIR / "prom"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DYELOG_",
        env_file_encoding="utf-8",
    )


settings = Settings()
