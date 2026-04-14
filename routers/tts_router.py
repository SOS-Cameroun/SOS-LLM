from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import logging
from models.schemas import TTSRequest, TTSResponse
from services.tts_service import tts_service
import os

router = APIRouter()
logger = logging.getLogger("ai-inference.tts-router")

@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_text(request: TTSRequest):
    """
    Reçoit un texte, lance l'inférence TTS et retourne l'URL (ou on peut renvoyer le fichier direct).
    Pour l'instant, on renvoie le chemin du fichier (qui pourra servir à un composant de téléchargement).
    """
    try:
        audio_path = await tts_service.synthesize(request.text, request.voice)
        return TTSResponse(audio_url=audio_path)
        
    except Exception as e:
        logger.error(f"Erreur de routeur TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{filename}")
async def download_audio(filename: str):
    """
    Permet de télécharger l'audio généré.
    """
    # En prod, valider l'existence et sécuriser le filepath
    file_path = os.path.join("/tmp/sos_audio_tts", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")
