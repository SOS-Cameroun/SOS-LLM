import logging
from faster_whisper import WhisperModel
import os

logger = logging.getLogger("ai-inference.stt")

class STTService:
    def __init__(self):
        # Initialisation du modèle Whisper_small (assez léger pour 8Go RAM)
        # device="cpu" permet de forcer sur CPU pour les machines sans GPU. 
        # compute_type="int8" diminue l'empreinte mémoire à < 1 Go.
        model_size = os.getenv("WHISPER_MODEL", "small")
        logger.info(f"⏳ Chargement du modèle Whisper ({model_size}) en RAM...")
        try:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("✅ Modèle Whisper chargé avec succès.")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de Whisper : {e}")
            self.model = None

    def transcribe(self, audio_path: str):
        """
        Transcrit le fichier audio situé dans audio_path en texte.
        Retourne le texte et la langue détectée.
        """
        if self.model is None:
            raise RuntimeError("Le modèle Whisper n'est pas instancié.")

        logger.info(f"🎤 Transcription en cours : {audio_path}")
        segments, info = self.model.transcribe(audio_path, beam_size=5)

        # Les segments sont un iterateur, on extrait tout le texte
        transcript = " ".join([segment.text.strip() for segment in segments])
        
        return {
            "text": transcript,
            "language": info.language,
            "duration": info.duration
        }

# Instance Singleton
stt_service = STTService()
