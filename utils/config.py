"""
Configuration centralisée via .env
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SPRING_BACKEND_URL: str = "http://localhost:8080"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral:7b"
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
    REDIS_URL: str = "redis://localhost:6379"
    WHISPER_MODEL: str = "base"  # tiny | base | small | medium | large
    TTS_VOICE: str = "fr-FR-DeniseNeural"  # Azure TTS Voice (fr-FR-HenriNeural, fr-FR-DeniseNeural)
    GROQ_API_KEY: str = "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"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
