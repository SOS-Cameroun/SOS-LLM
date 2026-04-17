"""
Router Vision — Analyse de fiabilité des images
=================================================
Endpoint pour vérifier si une image de preuve est cohérente avec
la description textuelle d'une alerte (Anti-Fraude).
"""

import os
import uuid
import shutil
import logging
from typing import Optional, Annotated
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from services.vision_service import vision_service
from services.fraud_db import fraud_db
from models.schemas import AnalyseVision, LabelFiabilite

router = APIRouter()
logger = logging.getLogger("ai-inference.vision-router")

UPLOAD_DIR = "/tmp/sos_vision_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post(
    "/analyze",
    response_model=AnalyseVision,
    summary="🔍 Analyser la fiabilité d'une image",
    description="Compare une image de preuve avec le texte de l'alerte pour détecter les fraudes.",
)
async def analyze_image_reliability(
    text: Annotated[str, Form(description="Description textuelle de l'alerte")],
    image: Annotated[UploadFile, File(description="Image de preuve (JPG/PNG)")],
):
    """Calcule le score de fiabilité entre une image et un texte."""
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(image.filename or "image.jpg")[1] or ".jpg"
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    try:
        # Sauvegarde temporaire
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # 1. Analyse de la scène
        vision_res = vision_service.analyze_image(temp_path)
        
        # 2. Comparaison texte-image
        comparison = vision_service.compare_text_image(text, vision_res)
        
        score = comparison.get("score_coherence", 50)
        is_coherent = comparison.get("coherent", True)
        
        # 3. Logging automatique en cas de suspicion (score < 70 ou incoherent)
        if not is_coherent or score < 70:
            label = LabelFiabilite.FRAUDE if score < 40 else LabelFiabilite.SUSPECTE
            await fraud_db.log_fraud(
                score_fiabilite=int(score),
                label=label.value,
                raison=comparison.get("explication", "Incohérence détectée par Vision"),
                alert_text=text,
                description_image=vision_res.get("description_scene"),
                lieu_declare=None, # On pourrait extraire le lieu si besoin
            )

        return AnalyseVision(
            description_scene=vision_res.get("description_scene", ""),
            dangers_detectes=vision_res.get("dangers_detectes", []),
            coherence_avec_texte=is_coherent,
            score_coherence=score,
            details_incoherence=comparison.get("details_incoherence"),
        )

    except Exception as e:
        logger.error(f"❌ Erreur analyse vision : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
