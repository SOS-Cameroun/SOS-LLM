"""
SOS-Cameroun — AI Inference Microservice
================================================================
Point d'entrée FastAPI optimisé pour le traitement multimodal.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from routers import (
    health_router,
    alerts_router
)
from utils.config import settings
from utils.rabbitmq_client import start_rabbitmq_consumer
from services.fraud_db import fraud_db

# Configuration du Logging
logging.basicConfig(level=settings.LOG_LEVEL.upper())
logger = logging.getLogger("ai-inference")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie : RabbitMQ + SQLite Init."""
    logger.info("🚀 SOS-Cameroun AI Inference - Démarrage")
    await fraud_db.init()
    asyncio.create_task(start_rabbitmq_consumer())
    logger.info("✅ Services prêts (RabbitMQ + SQLite)")
    yield
    logger.info("⏹️ Arrêt du microservice.")

app = FastAPI(
    title="SOS-Cameroun LLM",
    description="Microservice multimodal SOS-Cameroun : Whisper + LLM + Vision + Edge-TTS",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routers
app.include_router(alerts_router.router, prefix="/alerts", tags=["Alertes Intelligentes"])
app.include_router(health_router.router, tags=["Health"])

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
