from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
import logging
import os
import uuid
import shutil
import asyncio
from typing import Optional

from services.llm_service import llm_service
from services.stt_service import stt_service
from services.notification_service import notification_service
from services.tts_service import tts_service
from services.fraud_db import fraud_db
from services.geo_service import geo_service
from services.supabase_service import supabase_service
from utils.config import settings
from models.schemas import NiveauStress, LabelFiabilite

router = APIRouter()
logger = logging.getLogger("ai-inference.alerts-router")

@router.post(
    "/report",
    summary="Signaler une urgence (Voix ou Texte)",
    description="""
    Endpoint unifié pour le signalement d'urgences. 
    Prend en charge l'audio (fichier) et/ou le texte brut.
    
    Pipeline : 
    1. Transcription (si audio)
    2. Nettoyage et Réparation (IA)
    3. Analyse de Stress
    4. Extraction d'incident (Type, Gravité, Lieu)
    5. Notification Email (Brevo) aux autorités
    6. Génération de conseils de réassurance (IA)
    7. Synthèse Vocale (TTS) de la réponse
    """
)
async def report_alert(
    background_tasks: BackgroundTasks,
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    citizen_id: Optional[str] = Form(None),
):
    if not text and not file:
        raise HTTPException(status_code=400, detail="Vous devez fournir soit un texte, soit un fichier audio.")

    alert_id = str(uuid.uuid4())
    raw_text = text or ""
    transcript = ""
    audio_path = None

    # 1. Traitement Audio (STT)
    if file:
        UPLOAD_DIR = "/tmp/sos_audio_reports"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        audio_path = os.path.join(UPLOAD_DIR, f"{alert_id}{ext}")
        
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        try:
            stt_result = await asyncio.to_thread(stt_service.transcribe, audio_path)
            transcript = stt_result["text"]
            raw_text = (raw_text + " " + transcript).strip()
        except Exception as e:
            logger.error(f"❌ Erreur STT : {e}")

    # 2. Nettoyage et Réparation du texte (IA)
    # On utilise repair_transcription pour gérer les fautes et accents
    clean_text = llm_service.repair_transcription(raw_text)
    
    # 3. Résolution Géographique (GPS)
    resolved_lieu = None
    if latitude and longitude:
        nearest = geo_service.get_nearest_landmark(latitude, longitude)
        if nearest:
            resolved_lieu = nearest["display"]
            logger.info(f"Localisation GPS résolue : {resolved_lieu} ({nearest['distance_km']} km)")

    # 4. Analyse de Stress et d'Incident
    # On fait ça en parallèle pour gagner du temps
    stress_task = asyncio.to_thread(llm_service.analyze_stress_level, clean_text)
    entities_task = asyncio.to_thread(llm_service.extract_entities, clean_text)
    
    stress_result, entities = await asyncio.gather(stress_task, entities_task)
    
    type_incident = entities.get("type_incident", "AUTRE")
    gravite = entities.get("gravite", "Moyenne")
    
    # On privilégie le lieu résolu par GPS, sinon celui extrait du texte
    lieu = resolved_lieu or entities.get("lieu", "Inconnu")
    stress_level = stress_result.get("niveau", "MEDIUM")

    # 5. Notification Email (Brevo) - En arrière-plan
    incident_data = {
        "type_incident": type_incident,
        "gravite": gravite,
        "lieu": lieu,
        "description": clean_text,
        "gps": f"{latitude}, {longitude}" if latitude else "Non fourni"
    }
    
    background_tasks.add_task(
        notification_service.send_emergency_email,
        to_email=settings.EMERGENCY_AUTHORITY_EMAIL,
        subject=f"URGENCE {type_incident} - {lieu} ({gravite})",
        incident_data=incident_data
    )

    # 5. Notification aux contacts d'urgence (si citizen_id fourni)
    if citizen_id:
        contacts = supabase_service.get_citizen_contacts(citizen_id)
        for contact in contacts:
            c_email = contact.get("email")
            c_name = contact.get("nom") or "Proche"
            if c_email:
                background_tasks.add_task(
                    notification_service.send_emergency_email,
                    to_email=c_email,
                    subject=f"SOS : Alerte concernant un proche - {type_incident}",
                    incident_data=incident_data,
                    recipient_name=c_name
                )

    # 6. Réassurance et TTS
    reassurance_text = llm_service.generate_reassurance_advice(type_incident, gravite, stress_level)
    
    # On génère l'audio TTS
    tts_result = await tts_service.synthesize_emergency(reassurance_text, stress_level)

    # 6. Archivage Fraude (Si score faible)
    score = entities.get("score_fiabilite_initial", 100)
    if score < 80:
        label = LabelFiabilite.FRAUDE if score < 40 else LabelFiabilite.SUSPECTE
        await fraud_db.log_fraud(
            score_fiabilite=int(score),
            label=label.value,
            raison=f"Signalement mobile : {entities.get('resume', 'Lieu suspect')}",
            alert_text=clean_text,
            lieu_declare=lieu,
            lieu_detecte=lieu
        )

    return {
        "alert_id": alert_id,
        "status": "received",
        "analysis": {
            "type_incident": type_incident,
            "gravite": gravite,
            "lieu": lieu,
            "stress": stress_level,
            "clean_text": clean_text
        },
        "reassurance": {
            "text": reassurance_text,
            "audio_url": tts_result["audio_path"]
        }
    }
