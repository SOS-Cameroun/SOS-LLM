"""
Configuration centralisée via .env
====================================
Charge toutes les variables d'environnement nécessaires au microservice.
Utilise pydantic-settings pour la validation et les valeurs par défaut.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Paramètres du microservice AI Inference SOS-Cameroun.

    Toutes les valeurs peuvent être surchargées via le fichier ``.env``
    ou des variables d'environnement système.
    """

    # ── Backend Spring Boot ──────────────────────────────────────────────────
    SPRING_BACKEND_URL: str = "http://localhost:8080"

    # ── Groq LLM (Texte) ────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ── Groq Vision (Analyse d'image anti-fraude) ────────────────────────────
    GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # ── STT : faster-whisper ─────────────────────────────────────────────────
    WHISPER_MODEL: str = "base"  # tiny | base | small | medium | large

    # ── TTS : Coqui (Primaire) + Edge-TTS (Fallback) ────────────────────────
    TTS_ENGINE: str = "coqui"  # "coqui" = XTTS v2 (primaire), "edge" = Edge-TTS (secours)
    TTS_VOICE: str = "fr-FR-DeniseNeural"  # Voix Edge-TTS (fallback)
    COQUI_MODEL_NAME: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    STRESS_TTS_THRESHOLD: str = "HIGH"  # Seuil de stress pour déclencher le TTS : HIGH ou CRITICAL

    # ── Base de données Fraude (SQLite) ──────────────────────────────────────
    FRAUD_DB_PATH: str = "./data/fraud.db"

    # ── Message Queue ────────────────────────────────────────────────────────
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"

    # ── Cache ────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"

    # ── Notifications (Brevo) ────────────────────────────────────────────────
    BREVO_API_KEY: str = ""
    BREVO_SENDER_EMAIL: str = "sos-cameroun@aciai.com"
    EMERGENCY_AUTHORITY_EMAIL: str = "secours@aciai.com"

    # ── Supabase ─────────────────────────────────────────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # ── Serveur ──────────────────────────────────────────────────────────────
    PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
