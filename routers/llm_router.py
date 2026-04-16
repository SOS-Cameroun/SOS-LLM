"""
Router LLM — Inférence Groq Cloud
=====================================
POST /llm/prompt : prompt libre vers le LLM
POST /llm/extract : extraction d'entités (type incident, gravité, lieu)
POST /llm/action : parsing de commande vocale agent
POST /llm/stress : analyse du niveau de stress d'un texte
"""

from fastapi import APIRouter, HTTPException
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

router = APIRouter()
logger = logging.getLogger("ai-inference.llm-router")


@router.post(
    "/prompt",
    response_model=LLMResponse,
    summary="🧠 Prompt libre vers Groq LLM",
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
        content = llm_service.generate_response(request.prompt, request.context or "")
        return LLMResponse(response=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="📋 Extraire les entités d'un texte d'alerte",
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
    summary="🎤 Parser une commande vocale agent",
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
    summary="🎙️ Convertir des données structurées en texte pour TTS",
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
    summary="😰 Analyser le niveau de stress d'un texte",
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
