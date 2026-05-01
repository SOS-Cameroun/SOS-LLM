from fastapi import APIRouter, HTTPException, Body
import logging
from typing import Optional, List
from services.supabase_service import supabase_service
from pydantic import BaseModel, EmailStr

router = APIRouter()
logger = logging.getLogger("ai-inference.registration")

class RegistrationRequest(BaseModel):
    nom: str
    contact_email: EmailStr
    contact_phone: str
    nom_contact:str

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
