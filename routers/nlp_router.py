"""
Router NLP — UC14
POST /nlp/extract-entities : extrait les informations (Lieux, Personnes, Urgence) d'un texte simple.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from services.nlp_service import nlp_service

router = APIRouter()

class NLPRequest(BaseModel):
    text: str

@router.post("/extract-entities")
async def extract_entities(req: NLPRequest):
    """
    UC14 : Extraction d'entités depuis un texte pour pré-remplissage.
    (généralement utilisé seul ou en combinaison via le router STT)
    """
    return {
        "status": "ok",
        "entities": nlp_service.extract_entities(req.text)
    }
