import asyncio
import logging
import aio_pika
import json
import os
from services.stt_service import stt_service
from services.llm_service import llm_service

logger = logging.getLogger("ai-inference.rabbitmq")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

async def process_message(message: aio_pika.IncomingMessage):
    """
    Callback exécuté quand Spring Boot envoie un message dans la queue.
    """
    async with message.process():
        try:
            body = json.loads(message.body.decode())
            action_type = body.get("type")
            
            logger.info(f"📨 Message reçu depuis Spring Boot : Action={action_type}")
            
            if action_type == "STT_INFERENCE":
                audio_path = body.get("audio_path")
                # Appel STT Service (bloquant potentiellement, il faudrait faire un await en asyncio.to_thread si trop long)
                result = await asyncio.to_thread(stt_service.transcribe, audio_path)
                logger.info(f"✅ Résultat STT renvoyé via callback ou webhook : {result['text'][:50]}...")
                # Ici on pourrait publier le résultat dans une autre RabbitMQ Queue destinée à Java

            elif action_type == "EXTRACT_ENTITIES":
                text = body.get("text")
                result = await asyncio.to_thread(llm_service.extract_entities, text)
                logger.info(f"✅ Entités extraites : {result}")

            else:
                logger.warning(f"Action non supportée : {action_type}")

        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du message RabbitMQ : {e}")


async def start_rabbitmq_consumer():
    """
    Démarre la connexion à RabbitMQ et écoute la file 'SOS_AI_INFERENCE_QUEUE'.
    Doit être appelé dans le lifespan de FastAPI (main.py).
    """
    try:
        logger.info(f"🔗 Tentative de connexion à RabbitMQ : {RABBITMQ_URL}")
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        channel = await connection.channel()

        # Déclaration de la file
        queue = await channel.declare_queue("SOS_AI_INFERENCE_QUEUE", durable=True)

        logger.info("🐇 RabbitMQ Consumer démarré. En écoute sur 'SOS_AI_INFERENCE_QUEUE'...")
        
        # Consomme les messages
        await queue.consume(process_message)
    except Exception as e:
        logger.error(f"❌ Impossible de se connecter à RabbitMQ sur {RABBITMQ_URL}: {e}")
