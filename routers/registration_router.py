from fastapi import APIRouter, HTTPException, Body
import logging
from services.supabase_service import supabase_service
from pydantic import BaseModel, EmailStr

router = APIRouter()
logger = logging.getLogger("ai-inference.registration")

class RegistrationRequest(BaseModel):
    nom: str
    contact_email: EmailStr
    contact_phone: str

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
            contact_phone=request.contact_phone
        )
        return {
            "status": "success",
            "message": "Citoyen enregistré avec succès",
            "data": result
        }
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
