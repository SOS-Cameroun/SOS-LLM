"""
Router LLM — Inférence Groq Cloud
=====================================
POST /llm/prompt : prompt libre vers le LLM
POST /llm/extract : extraction d'entités (type incident, gravité, lieu)
POST /llm/action : parsing de commande vocale agent
POST /llm/stress : analyse du niveau de stress d'un texte
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
import logging

from models.schemas import (
    PromptRequest,
    LLMResponse,
    ExtractionResponse,
    ActionCommandResponse,
    StressAnalysis,
    NiveauStress,
)
from services.llm_service import llm_service
from services.stt_service import stt_service
import shutil
import os
import uuid
import asyncio

router = APIRouter()
logger = logging.getLogger("ai-inference.llm-router")


@router.post(
    "/prompt",
    response_model=LLMResponse,
    summary="Prompt libre vers Groq LLM",
    description="""
Envoie un prompt libre au LLM Groq (Llama 3.3 70B Versatile).
Le system prompt par défaut est ancré sur le contexte SOS-Cameroun / Yaoundé.

Peut être utilisé pour :
- Guider une victime (UC13)
- Générer un résumé pour un agent (UC51)
- Toute tâche de compréhension de texte
    """,
    response_description="Réponse textuelle du LLM",
)
async def prompt_llm(request: PromptRequest):
    """Envoie un prompt au LLM et retourne la réponse."""
    try:
        content = llm_service.generate_response(request.prompt)
        return LLMResponse(response=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="Extraire les entités d'un texte d'alerte",
    description="""
Extrait les entités structurées d'un texte d'alerte :
- **type_incident** : INCENDIE, ACCIDENT, MEDICAL, INONDATION, AGRESSION, AUTRE
- **gravite** : Critique, Haute, Moyenne, Basse
- **lieu** : lieu identifié dans le texte (validé contre la topographie de Yaoundé)
    """,
    response_description="Entités extraites (type, gravité, lieu)",
)
async def extract_information(request: PromptRequest):
    """Extrait les entités d'un texte via le LLM."""
    try:
        entities = llm_service.extract_entities(request.prompt)
        
        score = entities.get("score_fiabilite_initial", 100)
        
        # Logging automatique si le score de fiabilité est faible (< 80)
        if score < 80:
            from services.fraud_db import fraud_db
            from models.schemas import LabelFiabilite
            
            label = LabelFiabilite.FRAUDE if score < 40 else LabelFiabilite.SUSPECTE
            await fraud_db.log_fraud(
                score_fiabilite=int(score),
                label=label.value,
                raison=f"Fiabilité initiale faible : {entities.get('resume', 'Lieu suspect')}",
                alert_text=request.prompt,
                lieu_declare=entities.get("lieu"),
                lieu_detecte=entities.get("lieu"),
            )

        return ExtractionResponse(
            type_incident=entities.get("type_incident", "Inconnu"),
            gravite=entities.get("gravite", "Inconnue"),
            lieu=entities.get("lieu"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/action",
    response_model=ActionCommandResponse,
    summary="Parser une commande vocale agent",
    description="""
Convertit une commande vocale brute d'un agent du CINU en action structurée.

Actions reconnues :
- ``affecter_agent`` : affecter un agent à une alerte
- ``rejeter_alerte`` : rejeter une alerte (fausse alerte)
- ``valider_alerte`` : valider et traiter une alerte
    """,
    response_description="Action structurée avec paramètres",
)
async def parse_action(request: PromptRequest):
    """Parse une commande vocale en action structurée."""
    try:
        action_obj = llm_service.process_voice_action(request.prompt)
        return ActionCommandResponse(
            action=action_obj.get("action", "unknown"),
            parametres=action_obj.get("parametres", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/summarize_for_tts",
    response_model=LLMResponse,
    summary="Convertir des données structurées en texte pour TTS",
    description="""
    Prend un objet JSON (ex: conseils, contacts) et le convertit en un paragraphe
    fluide et bienveillant, prêt à être lu par le moteur TTS (Edge-TTS).
    """,
    response_description="Texte fluide pour synthèse vocale",
)
async def summarize_for_tts(data: dict):
    """Génère un résumé textuel pour le TTS à partir de JSON."""
    try:
        text = llm_service.generate_tts_response(data)
        return LLMResponse(response=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/stress",
    response_model=StressAnalysis,
    summary="Analyser le niveau de stress d'un texte",
    description="""
Analyse le niveau de stress émotionnel dans un texte d'alerte.

Détection basée sur :
- Mots de panique (au secours, SOS, on va mourir...)
- Ponctuation excessive (!!!, ???, MAJUSCULES)
- Répétitions (vite vite, à l'aide à l'aide)
- Phrases fragmentées et mots isolés
- Vocabulaire émotionnel (peur, sang, mort)

Niveaux : LOW (0-0.3), MEDIUM (0.3-0.6), HIGH (0.6-0.8), CRITICAL (0.8-1.0)
    """,
    response_description="Analyse de stress : niveau, score, indicateurs",
)
async def analyze_stress(request: PromptRequest):
    """Analyse le stress émotionnel d'un texte."""
    try:
        result = llm_service.analyze_stress_level(request.prompt)
        return StressAnalysis(
            niveau=NiveauStress(result.get("niveau", "MEDIUM")),
            score=result.get("score", 0.5),
            indicateurs=result.get("indicateurs", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/stress-audio",
    response_model=StressAnalysis,
    summary="Analyser le niveau de stress d'un fichier audio",
    description="""
Analyse le stress émotionnel en combinant l'acoustique (ton) et le texte (transcription).
Utile pour détecter l'urgence réelle via le ton de la voix.
    """,
    response_description="Analyse de stress complète (Audio + Texte)",
)
async def analyze_stress_audio(
    file: UploadFile = File(..., description="Fichier audio wav/mp3")
):
    """Analyse le stress émotionnel d'un audio."""
    UPLOAD_DIR = "/tmp/sos_audio_stress"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 1. Transcription pour analyse textuelle
        stt_result = await asyncio.to_thread(stt_service.transcribe, temp_path)
        transcript = stt_result["text"]
        
        # 2. Analyse acoustique du ton
        tone_result = await asyncio.to_thread(stt_service.analyze_tone, temp_path)
        
        # 3. Analyse hybride via LLM
        final_result = llm_service.analyze_stress_level(
            text=transcript,
            tone_score=tone_result["tone_score"],
            acoustic_indicators=tone_result["indicators"]
        )
        
        return StressAnalysis(
            niveau=NiveauStress(final_result.get("niveau", "MEDIUM")),
            score=final_result.get("score", 0.5),
            indicateurs=final_result.get("indicateurs", []),
        )
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse de stress audio : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
