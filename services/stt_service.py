import logging
from faster_whisper import WhisperModel
import os
import numpy as np
from pydub import AudioSegment

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

    def analyze_tone(self, audio_path: str) -> dict:
        """
        Analyse le ton de l'audio (acoustique) pour détecter le stress.
        Retourne un score de stress basé sur le volume (RMS) et l'agitation.
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            
            # 1. Volume (RMS)
            # Un volume élevé (cris) augmente le score de stress.
            # On normalise par rapport à un maximum théorique.
            rms = audio.rms
            max_rms = 32767 # Max pour 16-bit
            rms_score = min(rms / (max_rms * 0.5), 1.0) # On considère que 50% du max est déjà très fort
            
            # 2. Agitation (Variabilité du volume)
            # On découpe l'audio en morceaux de 100ms et on calcule la variation du RMS.
            chunk_length_ms = 100
            chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            if len(chunks) > 1:
                rms_values = [c.rms for c in chunks]
                agitation = np.std(rms_values) / (np.mean(rms_values) + 1e-6)
                # Une agitation élevée (> 1.0) indique une voix instable/paniquée.
                agitation_score = min(agitation / 1.5, 1.0)
            else:
                agitation_score = 0.0
            
            # Score final pondéré
            tone_score = (rms_score * 0.6) + (agitation_score * 0.4)
            
            indicators = []
            if rms_score > 0.6: indicators.append("volume_eleve")
            if agitation_score > 0.6: indicators.append("voix_agitee")
            
            return {
                "tone_score": float(tone_score),
                "indicators": indicators,
                "rms": float(rms_score),
                "agitation": float(agitation_score)
            }
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse acoustique : {e}")
            return {"tone_score": 0.0, "indicators": ["erreur_analyse_acoustique"]}

# Instance Singleton
stt_service = STTService()
