"""
Service TTS — Synthèse vocale (Edge-TTS uniquement)
=====================================================
Génère des réponses audio pour les victimes en détresse via Edge-TTS (Microsoft Azure Edge).

Intelligence émotionnelle :
    - Si stress CRITICAL → voix lente, posée, instructions courtes
    - Si stress HIGH → voix calme mais ferme, rassurante
    - Si stress MEDIUM/LOW → voix normale, informative
"""

import logging
import asyncio
import os
import uuid
from typing import Optional

import edge_tts
from utils.config import settings

logger = logging.getLogger("ai-inference.tts")

# Répertoire de sortie des fichiers audio
AUDIO_OUTPUT_DIR = "/tmp/sos_audio_tts"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


class TTSService:
    """
    Service de synthèse vocale avec intelligence émotionnelle.

    Moteur exclusif : **Edge-TTS** (Microsoft Azure Edge).
    Voix par défaut : fr-FR-DeniseNeural (claire et rassurante).
    """

    def __init__(self):
        self.edge_voice = settings.TTS_VOICE
        self.coqui_available = False  # Actuellement désactivé pour simplifier le déploiement
        logger.info(
            f"✅ TTSService initialisé — Edge-TTS uniquement ({self.edge_voice})"
        )

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        stress_level: Optional[str] = None,
    ) -> dict:
        """
        Synthétise un texte en audio avec adaptation émotionnelle.

        Le texte est adapté selon le niveau de stress avant synthèse :
        - CRITICAL : pauses ajoutées, phrases raccourcies
        - HIGH : ton calme injecté via la ponctuation
        - MEDIUM/LOW : texte normal

        Args:
            text: Texte à synthétiser.
            voice: Voix Edge-TTS (optionnel, défaut = config).
            stress_level: Niveau de stress (LOW, MEDIUM, HIGH, CRITICAL).

        Returns:
            dict avec ``audio_path`` et ``engine_used`` ('edge').
        """
        adapted_text = self._adapt_text_for_stress(text, stress_level)
        return await self._synthesize_edge(adapted_text, voice)

    async def synthesize_emergency(self, text: str, stress_level: str) -> dict:
        """
        Point d'entrée spécialisé pour les réponses d'urgence.

        Combine l'adaptation émotionnelle et la synthèse vocale.
        Le stress_level détermine directement le comportement vocal.

        Args:
            text: Réponse textuelle du LLM à vocaliser.
            stress_level: Niveau de stress de la victime (LOW/MEDIUM/HIGH/CRITICAL).

        Returns:
            dict avec ``audio_path`` et ``engine_used``.
        """
        return await self.synthesize(text, stress_level=stress_level)

    # ═══════════════════════════════════════════════════════════════════════════
    # Moteur de synthèse
    # ═══════════════════════════════════════════════════════════════════════════

    async def _synthesize_edge(self, text: str, voice: Optional[str] = None) -> dict:
        """
        Synthèse vocale via Edge-TTS (Microsoft Azure Edge).

        Args:
            text: Texte à synthétiser.
            voice: Voix Edge-TTS (défaut: fr-FR-DeniseNeural).

        Returns:
            dict avec audio_path et engine_used='edge'.
        """
        voice = voice or self.edge_voice
        file_id = str(uuid.uuid4())
        output_path = os.path.join(AUDIO_OUTPUT_DIR, f"{file_id}.mp3")

        logger.info(f"🔊 Synthèse Edge-TTS avec la voix {voice}...")

        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            logger.info(f"✅ Audio Edge-TTS généré : {output_path}")
            return {"audio_path": output_path, "engine_used": "edge"}
        except Exception as e:
            logger.error(f"❌ Edge-TTS a échoué : {e}")
            raise RuntimeError(f"Edge-TTS synthesis failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Adaptation émotionnelle
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _adapt_text_for_stress(text: str, stress_level: Optional[str]) -> str:
        """
        Adapte le texte pour la synthèse vocale selon le niveau de stress.

        - CRITICAL : Ajoute des pauses (points) entre les phrases courtes
        - HIGH : Allonge légèrement les pauses naturelles
        - MEDIUM/LOW : Texte inchangé

        Args:
            text: Texte original.
            stress_level: Niveau de stress.

        Returns:
            Texte adapté pour une meilleure synthèse émotionnelle.
        """
        if not stress_level or stress_level in ("LOW", "MEDIUM"):
            return text

        if stress_level == "CRITICAL":
            # Ajouter des pauses entre les phrases pour un débit plus lent
            text = text.replace(". ", "... ")
            # Ajouter une pause au début pour laisser la victime écouter
            text = f"... {text}"

        elif stress_level == "HIGH":
            # Pauses légèrement plus longues
            text = text.replace(". ", ".. ")

        return text


# ── Singleton ────────────────────────────────────────────────────────────────
tts_service = TTSService()
