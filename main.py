"""
SOS-Cameroun — AI Inference Microservice
Point d'entrée FastAPI : gère STT (Whisper), TTS (CoquiTTS), NLP (spaCy)
"""
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from routers import stt_router, tts_router, nlp_router, health_router, llm_router
from utils.config import settings
from utils.rabbitmq_client import start_rabbitmq_consumer
from services.stt_service import stt_service
from services.tts_service import tts_service
from services.nlp_service import nlp_service
from services.llm_service import llm_service

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=settings.LOG_LEVEL.upper())
logger = logging.getLogger("ai-inference")


# ---------------------------------------------------------------------------
# Lifecycle (chargement des modèles au démarrage)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Démarrage du microservice AI Inference...")
    
    # Démarrage du consommateur RabbitMQ en tâche de fond
    asyncio.create_task(start_rabbitmq_consumer())
    
    logger.info("✅ Tous les services et modèles sont prêts.")
    yield
    logger.info("⏹️ Arrêt du microservice AI Inference.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SOS-Cameroun AI Inference",
    description="Microservice Python : Whisper STT · CoquiTTS · spaCy NER · Groq LLM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restreindre en production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routeurs
app.include_router(health_router.router, tags=["Health"])
app.include_router(stt_router.router, prefix="/stt", tags=["STT — Whisper"])
app.include_router(tts_router.router, prefix="/tts", tags=["TTS — EdgeTTS"])
app.include_router(nlp_router.router, prefix="/nlp", tags=["NLP — spaCy"])
app.include_router(llm_router.router, prefix="/llm", tags=["LLM — Groq"])

@app.get("/", include_in_schema=False)
async def root_redirect():
    """
    Redirige la racine vers la documentation API.
    """
    return RedirectResponse(url="/docs")
