"""
Router STT — Transcription vocale
====================================
POST /stt/transcribe : transcrit un fichier audio en texte via faster-whisper.
POST /stt/transcribe-and-extract : pipeline complet Audio → Texte → Entités.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
import shutil
import os
import uuid
import asyncio
import logging

from services.stt_service import stt_service
from services.llm_service import llm_service
from services.nlp_service import nlp_service
from models.schemas import STTResponse

router = APIRouter()
logger = logging.getLogger("ai-inference.stt-router")

UPLOAD_DIR = "/tmp/sos_audio_stt"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post(
    "/transcribe",
    response_model=STTResponse,
    summary="🎤 Transcrire un fichier audio",
    description="Transcrit un fichier audio (wav/mp3) en texte via faster-whisper. "
                "Retourne le texte, la langue détectée et la durée.",
    response_description="Transcription textuelle avec métadonnées",
)
async def transcribe_audio(file: UploadFile = File(..., description="Fichier audio wav/mp3")):
    """Transcrit un fichier audio en texte."""
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        stt_result = await asyncio.to_thread(stt_service.transcribe, temp_path)
        return STTResponse(
            text=stt_result["text"],
            language=stt_result["language"],
            duration=stt_result.get("duration"),
        )
    except Exception as e:
        logger.error(f"❌ Erreur STT : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# transcribe_and_extract supprimé car redondant avec /pipeline/alert