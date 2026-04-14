from fastapi import APIRouter, File, UploadFile, HTTPException
import shutil
import os
import uuid
import logging
from services.stt_service import stt_service
from models.schemas import STTResponse

router = APIRouter()
logger = logging.getLogger("ai-inference.stt-router")

UPLOAD_DIR = "/tmp/sos_audio_stt"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/transcribe", response_model=STTResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Reçoit un fichier audio, lance l'inférence STT (Whisper)
    et supprime le fichier temporaire ensuite. (UC11, UC12, UC50)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Fichier non fourni")

    # Génération d'un nom de fichier unique sécurisé
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".wav"
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    try:
        # Écriture du fichier
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Appel au service
        result = stt_service.transcribe(temp_path)
        return STTResponse(**result)

    except Exception as e:
        logger.error(f"Erreur lors de la transcription : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Nettoyage du fichier pour ne pas saturer le disque
        if os.path.exists(temp_path):
            os.remove(temp_path)
