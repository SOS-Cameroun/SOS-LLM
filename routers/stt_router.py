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
    summary="Transcrire un fichier audio",
    description="Transcrit un fichier audio (wav/mp3) en texte via faster-whisper. "
                "Retourne le texte, la langue détectée et la durée.",
    response_description="Transcription textuelle avec métadonnées",
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Fichier audio wav/mp3"),
    refine: bool = True,
):
    """Transcrit un fichier audio en texte avec raffinement optionnel."""
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        stt_result = await asyncio.to_thread(stt_service.transcribe, temp_path)
        raw_text = stt_result["text"]
        final_text = raw_text

        if refine and raw_text:
            logger.info("🪄 Raffinement intelligent de la transcription via LLM (Geo-Awar)...")
            from services.geo_service import geo_service
            landmarks = ", ".join(geo_service.get_all_landmarks())
            final_text = llm_service.repair_transcription(raw_text, known_places=landmarks)

        return STTResponse(
            text=final_text,
            raw_text=raw_text if refine else None,
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