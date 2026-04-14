"""
Router Health (Liveness/Readiness probes)
"""
from fastapi import APIRouter
from utils.config import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    """Vérifie l'état du microservice AI Inference."""
    return {
        "status": "up",
        "service": "ai-inference",
        "stt": settings.WHISPER_MODEL,
        "tts": settings.TTS_VOICE,
        "rabbitmq_url": settings.RABBITMQ_URL.split("@")[-1]  # Hide credentials
    }
