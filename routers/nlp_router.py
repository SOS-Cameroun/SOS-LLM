"""
Router NLP — Extraction d'entités
====================================
POST /nlp/extract-entities : extrait les lieux, personnes, et type d'urgence d'un texte.
POST /nlp/clean : nettoie un texte brut (transcription vocale, saisie).
POST /nlp/check-fragment : vérifie si un texte est fragmenté (danger imminent).
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from services.nlp_service import nlp_service

router = APIRouter()


class NLPRequest(BaseModel):
    """Requête NLP : texte brut à analyser."""
    text: str = Field(..., description="Texte à analyser", example="Incendie au marché Mokolo, il y a des blessés")


@router.post(
    "/extract-entities",
    summary="📝 Extraire les entités d'un texte",
    description="""
Extraction d'entités nommées via spaCy + validation topographique de Yaoundé.

Entités extraites :
- **localisation** : lieu identifié (validé contre la base de Yaoundé)
- **personnes** : noms de personnes mentionnées
- **organisations** : organisations mentionnées
- **type_urgence_detecte** : INCENDIE, ACCIDENT, MEDICAL, INONDATION, AGRESSION, AUTRE
- **lieu_valide_yaounde** : True si le lieu est reconnu dans la topographie de Yaoundé
    """,
    response_description="Entités extraites avec validation topographique",
)
async def extract_entities(req: NLPRequest):
    """Extrait les entités d'un texte via spaCy + GeoService."""
    return {
        "status": "ok",
        "entities": nlp_service.extract_entities(req.text),
    }


@router.post(
    "/clean",
    summary="🧹 Nettoyer un texte brut",
    description="Normalise un texte brut (unicode, espaces, casse). "
                "Utile pour les transcriptions vocales ou les saisies utilisateur.",
    response_description="Texte nettoyé",
)
async def clean_text(req: NLPRequest):
    """Nettoie un texte brut via le pipeline NLP."""
    return {
        "original": req.text,
        "cleaned": nlp_service.clean_text(req.text),
    }


@router.post(
    "/check-fragment",
    summary="⚠️ Vérifier si un message est fragmenté",
    description="""
Détermine si un message semble fragmenté (danger imminent probable).

Critères de détection :
- Moins de 5 mots
- Points de suspension (...)
- Mots isolés séparés par des virgules

Si True, le pipeline multimodal déclenchera l'hallucination intelligente
pour compléter le message sans perdre de temps.
    """,
    response_description="Résultat de la vérification de fragmentation",
)
async def check_fragment(req: NLPRequest):
    """Vérifie si un texte est fragmenté (danger imminent)."""
    return {
        "text": req.text,
        "is_fragmented": nlp_service.is_fragmented(req.text),
    }
