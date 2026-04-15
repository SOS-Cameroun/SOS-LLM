"""
Router Health — Probes de santé
==================================
GET /health : vérifie l'état du microservice et de ses composants.
"""

from fastapi import APIRouter
from utils.config import settings
from services.tts_service import tts_service

router = APIRouter()


@router.get(
    "/health",
    summary="🩺 Vérifier l'état du microservice",
    description="Retourne l'état de santé du microservice et de tous ses composants.",
    response_description="Status de chaque composant du système",
)
async def health_check():
    """Vérifie l'état du microservice AI Inference."""
    return {
        "status": "up",
        "service": "ai-inference",
        "version": "2.0.0",
        "components": {
            "stt": {"model": settings.WHISPER_MODEL, "engine": "faster-whisper"},
            "tts": {
                "primary": "coqui-xtts-v2" if tts_service.coqui_available else "indisponible",
                "fallback": f"edge-tts ({settings.TTS_VOICE})",
                "active_engine": "coqui" if tts_service.coqui_available else "edge",
            },
            "llm": {"model": settings.GROQ_MODEL, "engine": "groq-cloud"},
            "vision": {"model": settings.GROQ_VISION_MODEL, "engine": "groq-cloud"},
            "fraud_db": {"path": settings.FRAUD_DB_PATH, "engine": "sqlite"},
        },
        "geo_coverage": "Yaoundé (Cameroun)",
    }
