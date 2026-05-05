from fastapi import APIRouter, HTTPException, Body
import logging
from typing import Optional, List
from services.supabase_service import supabase_service
from pydantic import BaseModel, EmailStr

router = APIRouter()
logger = logging.getLogger("ai-inference.registration")

class RegistrationRequest(BaseModel):
    nom: str
    email: EmailStr                  # Email du citoyen lui-même
    telephone: str                   # Numéro de téléphone du citoyen
    contact_email: EmailStr          # Email du contact d'urgence
    contact_phone: str               # Téléphone du contact d'urgence
    nom_contact: str

class ContactRequest(BaseModel):
    citizen_id: str
    email: EmailStr
    phone: str
    nom: Optional[str] = "Contact d'Urgence"

@router.post(
    "/register",
    summary="Enregistrer un citoyen et son contact d'urgence",
    description="Insère les informations dans les tables Supabase 'citoyen' et 'contact_urgence'."
)
async def register(request: RegistrationRequest):
    try:
        result = supabase_service.register_citizen(
            nom=request.nom,
            email=request.email,
            telephone=request.telephone,
            contact_email=request.contact_email,
            contact_phone=request.contact_phone,
            nom_contact=request.nom_contact
        )
        return {
            "status": "success",
            "message": "Citoyen enregistré avec succès",
            "data": result
        }
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/add-contact",
    summary="Ajouter un contact d'urgence supplémentaire",
    description="Lie un nouveau contact d'urgence à un citoyen existant."
)
async def add_contact(request: ContactRequest):
    try:
        result = supabase_service.add_emergency_contact(
            citizen_id=request.citizen_id,
            email=request.email,
            phone=request.phone,
            nom=request.nom
        )
        return {
            "status": "success",
            "message": "Contact d'urgence ajouté avec succès",
            "data": result
        }
    except Exception as e:
        logger.error(f"Add contact error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/citizen/{citizen_id}",
    summary="Récupérer les informations d'un citoyen",
    description="Renvoie les détails d'un citoyen depuis Supabase."
)
async def get_citizen(citizen_id: str):
    try:
        citizen = supabase_service.get_citizen(citizen_id)
        if not citizen:
            raise HTTPException(status_code=404, detail="Citoyen non trouvé")
        return {
            "status": "success",
            "data": citizen
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get citizen error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/citizen/{citizen_id}/contacts",
    summary="Récupérer les contacts d'urgence d'un citoyen",
    description="Renvoie la liste des contacts d'urgence liés à un citoyen."
)
async def get_citizen_contacts(citizen_id: str):
    try:
        contacts = supabase_service.get_citizen_contacts(citizen_id)
        return {
            "status": "success",
            "data": contacts
        }
    except Exception as e:
        logger.error(f"Get citizen contacts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
