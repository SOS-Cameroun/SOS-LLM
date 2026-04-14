import logging
import asyncio
import edge_tts
import os
import uuid

logger = logging.getLogger("ai-inference.tts")

AUDIO_OUTPUT_DIR = "/tmp/sos_audio_tts"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

class TTSService:
    def __init__(self):
        logger.info("✅ Service TTS (Edge-TTS) initialisé.")

    async def synthesize(self, text: str, voice: str = "fr-FR-DeniseNeural") -> str:
        """
        Convertit un texte en fichier audio MP3 et retourne le chemin
        (UC25 Guider victime, UC51 Résumé Agent, UC53 Alerte)
        """
        logger.info(f"🔊 Synthèse vocale requise avec la voix {voice}")
        
        file_id = str(uuid.uuid4())
        output_path = os.path.join(AUDIO_OUTPUT_DIR, f"{file_id}.mp3")
        
        try:
            # edge-tts est asynchrone, il télécharge directement le flux vers le fichier
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            
            logger.info(f"✅ Audio généré : {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération vocale : {e}")
            raise e

# Instance Singleton
tts_service = TTSService()
