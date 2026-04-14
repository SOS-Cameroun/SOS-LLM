from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class STTResponse(BaseModel):
    text: str = Field(..., description="Texte transcrit de l'audio")
    language: str = Field(..., description="Langue détectée")
    duration: Optional[float] = Field(None, description="Durée de l'audio en secondes")

class TTSRequest(BaseModel):
    text: str = Field(..., description="Texte à synthétiser")
    voice: Optional[str] = Field("fr-FR-DeniseNeural", description="Voix d'Edge TTS")

class TTSResponse(BaseModel):
    audio_url: str = Field(..., description="Chemin ou URL du fichier audio généré")

class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Le prompt à envoyer au modèle LLM")
    context: Optional[str] = Field(None, description="Contexte supplémentaire (RAG)")

class LLMResponse(BaseModel):
    response: str = Field(..., description="Réponse brute du LLM")
    
class ExtractionResponse(BaseModel):
    type_incident: str
    gravite: str    
    lieu: Optional[str] = None
    
class ActionCommandResponse(BaseModel):
    action: str
    parametres: Dict[str, Any]
