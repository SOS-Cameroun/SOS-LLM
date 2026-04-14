from fastapi import APIRouter, HTTPException
import logging
from models.schemas import PromptRequest, LLMResponse, ExtractionResponse, ActionCommandResponse
from services.llm_service import llm_service

router = APIRouter()
logger = logging.getLogger("ai-inference.llm-router")

@router.post("/prompt", response_model=LLMResponse)
async def prompt_llm(request: PromptRequest):
    """
    Endpoint général pour poser des questions ou guider (UC13, UC51)
    """
    try:
        content = llm_service.generate_response(request.prompt, request.context or "")
        return LLMResponse(response=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract", response_model=ExtractionResponse)
async def extract_information(request: PromptRequest):
    """
    Extrait les entités (type incident, gravité) pour pré-remplissage (UC14, UC63)
    """
    try:
        entities = llm_service.extract_entities(request.prompt)
        # Assure le retour conforme au Pydantic Schema
        return ExtractionResponse(
            type_incident=entities.get("type_incident", "Inconnu"),
            gravite=entities.get("gravite", "Inconnue"),
            lieu=entities.get("lieu")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/action", response_model=ActionCommandResponse)
async def parse_action(request: PromptRequest):
    """
    Parse la commande vocale de l'agent en action structurée (UC52)
    """
    try:
        action_obj = llm_service.process_voice_action(request.prompt)
        return ActionCommandResponse(
            action=action_obj.get("action", "unknown"),
            parametres=action_obj.get("parametres", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
