"""
SOS-Cameroun — Schémas Pydantic
================================
Modèles de données pour le microservice AI Inference.
Couvre : STT, TTS, NLP, LLM, Pipeline multimodal, et Détection de fraude.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class NiveauStress(str, Enum):
    """Niveau de stress détecté dans le message de l'utilisateur."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class LabelFiabilite(str, Enum):
    """Label de fiabilité d'une alerte après analyse anti-fraude."""
    FIABLE = "FIABLE"
    SUSPECTE = "SUSPECTE"
    FRAUDE = "FRAUDE"


class TypeUrgence(str, Enum):
    """Types d'urgence reconnus par le système."""
    INCENDIE = "INCENDIE"
    ACCIDENT = "ACCIDENT"
    MEDICAL = "MEDICAL"
    INONDATION = "INONDATION"
    AGRESSION = "AGRESSION"
    AUTRE = "AUTRE"


# ═══════════════════════════════════════════════════════════════════════════════
# STT (Speech-to-Text)
# ═══════════════════════════════════════════════════════════════════════════════

class STTResponse(BaseModel):
    """Résultat de la transcription audio → texte via faster-whisper."""
    text: str = Field(..., description="Texte transcrit de l'audio")
    language: str = Field(..., description="Langue détectée (code ISO)")
    duration: Optional[float] = Field(None, description="Durée de l'audio en secondes")


# ═══════════════════════════════════════════════════════════════════════════════
# TTS (Text-to-Speech)
# ═══════════════════════════════════════════════════════════════════════════════

class TTSRequest(BaseModel):
    """Requête de synthèse vocale."""
    text: str = Field(..., description="Texte à synthétiser en audio")
    voice: Optional[str] = Field(
        "fr-FR-DeniseNeural",
        description="Voix Edge-TTS (fallback). Ignoré si Coqui-TTS est actif."
    )
    stress_level: Optional[NiveauStress] = Field(
        None,
        description="Niveau de stress de l'appelant. Si HIGH/CRITICAL, la voix sera calme et ferme."
    )


class TTSResponse(BaseModel):
    """Résultat de la synthèse vocale."""
    audio_url: str = Field(..., description="Chemin ou URL du fichier audio généré")
    engine_used: str = Field("edge", description="Moteur utilisé : 'coqui' ou 'edge'")


# ═══════════════════════════════════════════════════════════════════════════════
# NLP
# ═══════════════════════════════════════════════════════════════════════════════

class NLPRequest(BaseModel):
    """Requête d'extraction d'entités NLP."""
    text: str = Field(..., description="Texte brut à analyser")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM (Groq)
# ═══════════════════════════════════════════════════════════════════════════════

class PromptRequest(BaseModel):
    """Requête générique vers le LLM Groq. Le contexte est géré en interne."""
    prompt: str = Field(..., description="Le prompt à envoyer au modèle LLM")


class LLMResponse(BaseModel):
    """Réponse brute du LLM."""
    response: str = Field(..., description="Réponse textuelle du LLM")


class ExtractionResponse(BaseModel):
    """Entités extraites par le LLM pour pré-remplissage du formulaire d'alerte."""
    type_incident: str = Field(..., description="Type d'incident détecté")
    gravite: str = Field(..., description="Niveau de gravité estimé")
    lieu: Optional[str] = Field(None, description="Lieu identifié dans le texte")


class ActionCommandResponse(BaseModel):
    """Commande vocale d'un agent, convertie en action structurée."""
    action: str = Field(..., description="Action reconnue (affecter_agent, rejeter_alerte, etc.)")
    parametres: Dict[str, Any] = Field(
        default_factory=dict, description="Paramètres de l'action (id alerte, etc.)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Analyse de Stress
# ═══════════════════════════════════════════════════════════════════════════════

class StressAnalysis(BaseModel):
    """Résultat de l'analyse émotionnelle du message."""
    niveau: NiveauStress = Field(..., description="Niveau de stress détecté")
    score: float = Field(
        ..., ge=0, le=1,
        description="Score de stress normalisé entre 0.0 (calme) et 1.0 (panique totale)"
    )
    indicateurs: List[str] = Field(
        default_factory=list,
        description="Indicateurs détectés (cris, mots de panique, répétitions, ponctuation excessive)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Vision & Anti-Fraude
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyseVision(BaseModel):
    """Résultat de l'analyse d'image par Groq Vision (Llama 3.2 Vision)."""
    description_scene: str = Field(..., description="Description textuelle de la scène visible")
    dangers_detectes: List[str] = Field(
        default_factory=list, description="Dangers visibles (feu, fumée, accident, etc.)"
    )
    coherence_avec_texte: bool = Field(
        ..., description="True si la scène correspond à la description textuelle"
    )
    score_coherence: float = Field(
        ..., ge=0, le=100,
        description="Score de cohérence texte-image (0 = totalement incohérent, 100 = parfaitement cohérent)"
    )
    details_incoherence: Optional[str] = Field(
        None,
        description="Explication si incohérence détectée (ex: 'La photo montre une fête, pas un incendie')."
    )


class ScoreFiabilite(BaseModel):
    """Score de fiabilité global de l'alerte."""
    score: int = Field(
        ..., ge=0, le=100,
        description="Score de fiabilité global (0 = fraude certaine, 100 = alerte vérifiée)"
    )
    label: LabelFiabilite = Field(..., description="Verdict : FIABLE, SUSPECTE ou FRAUDE")
    raison: str = Field(..., description="Explication résumée de la note attribuée")


class FraudLogEntry(BaseModel):
    """Entrée dans le journal de fraude (table SQLite)."""
    id: str = Field(..., description="Identifiant unique UUID du log")
    timestamp: str = Field(..., description="Date/heure ISO 8601 de l'alerte")
    alert_text: Optional[str] = Field(None, description="Texte brut de l'alerte")
    image_hash: Optional[str] = Field(None, description="Hash perceptuel de l'image (pHash)")
    image_path: Optional[str] = Field(None, description="Chemin du fichier image archivé")
    description_image: Optional[str] = Field(None, description="Description IA de l'image")
    score_fiabilite: int = Field(..., description="Score de fiabilité 0-100")
    label: LabelFiabilite = Field(..., description="Verdict de fiabilité")
    raison: str = Field(..., description="Explication de la décision")
    lieu_declare: Optional[str] = Field(None, description="Lieu mentionné par l'utilisateur")
    lieu_detecte: Optional[str] = Field(None, description="Lieu identifié par le système (NLP/GPS)")
    ip_source: Optional[str] = Field(None, description="IP de l'appelant")
    resolved: bool = Field(False, description="True si le cas a été traité/résolu")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Multimodal — Requête / Réponse principale
# ═══════════════════════════════════════════════════════════════════════════════

class AlerteMultimodaleResponse(BaseModel):
    """
    Réponse complète du pipeline multimodal SOS-Cameroun.

    Contient l'ensemble des résultats des étapes :
    STT → NLP → Groq Vision → Fraude → LLM → TTS.
    """
    # Transcription (si audio fourni)
    transcript: Optional[str] = Field(None, description="Texte transcrit de l'audio (si fourni)")
    langue_detectee: Optional[str] = Field(None, description="Langue détectée dans l'audio")

    # Texte final utilisé (après nettoyage / hallucination)
    texte_final: str = Field(
        ..., description="Texte final après nettoyage NLP et hallucination intelligente si fragmenté"
    )
    texte_hallucine: bool = Field(
        False,
        description="True si le LLM a complété un message fragmenté (danger imminent)"
    )

    # Entités extraites
    type_incident: str = Field(..., description="Type d'incident détecté")
    gravite: str = Field(..., description="Niveau de gravité")
    lieu: Optional[str] = Field(None, description="Lieu identifié et validé")
    lieu_valide: bool = Field(
        False,
        description="True si le lieu a été confirmé dans la topographie de Yaoundé"
    )

    # Analyse de stress
    stress: StressAnalysis = Field(..., description="Analyse émotionnelle du message")

    # Vision & Fraude (si image fournie)
    vision: Optional[AnalyseVision] = Field(
        None, description="Résultat de l'analyse d'image (si image fournie)"
    )
    fiabilite: ScoreFiabilite = Field(..., description="Score de fiabilité anti-fraude")

    # Réponse IA
    reponse_ia: str = Field(
        ...,
        description="Réponse du LLM pour guider la victime ou informer le centre d'urgence"
    )

    # TTS (si stress élevé)
    audio_reponse: Optional[str] = Field(
        None,
        description="Chemin vers l'audio TTS de la réponse (généré si stress HIGH/CRITICAL)"
    )
    tts_engine: Optional[str] = Field(
        None, description="Moteur TTS utilisé : 'coqui' ou 'edge'"
    )
