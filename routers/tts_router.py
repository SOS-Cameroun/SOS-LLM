"""
Router TTS — Synthèse vocale
================================
POST /tts/synthesize : convertit un texte en audio via Edge-TTS.
GET  /tts/download/{filename} : télécharge le fichier audio généré.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import logging
import os

from models.schemas import TTSRequest, TTSResponse
from services.tts_service import tts_service

router = APIRouter()
logger = logging.getLogger("ai-inference.tts-router")


@router.post(
    "/synthesize",
    response_model=TTSResponse,
    summary="Synthétiser du texte en audio",
    description="""
Convertit un texte en fichier audio via Edge-TTS (Microsoft Azure).

**Intelligence émotionnelle** : Si le ``stress_level`` est fourni,
la voix s'adapte automatiquement :
- **CRITICAL** : Voix lente, pauses entre les phrases, ton très calme
- **HIGH** : Voix calme et ferme, pauses légèrement allongées
- **MEDIUM/LOW** : Débit normal
    """,
    response_description="Chemin du fichier audio + moteur utilisé",
)
async def synthesize_text(request: TTSRequest):
    """Synthétise un texte en audio avec adaptation émotionnelle."""
    try:
        result = await tts_service.synthesize(
            text=request.text,
            voice=request.voice,
            stress_level=request.stress_level.value if request.stress_level else None,
        )
        return TTSResponse(
            audio_url=result["audio_path"],
            engine_used=result["engine_used"],
        )
    except Exception as e:
        logger.error(f"❌ Erreur TTS : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/download/{filename}",
    summary="Télécharger un audio TTS",
    description="Télécharge un fichier audio précédemment généré par le service TTS.",
    response_description="Fichier audio MP3 ou WAV",
)
async def download_audio(filename: str):
    """Télécharge un fichier audio TTS généré."""
    file_path = os.path.join("/tmp/sos_audio_tts", filename)
    if os.path.exists(file_path):
        media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
        return FileResponse(file_path, media_type=media_type, filename=filename)
    raise HTTPException(status_code=404, detail="Fichier audio non trouvé")
