"""
Service Vision — Analyse d'Image Anti-Fraude
==============================================
Utilise Groq Vision (Llama 3.2 Vision) pour analyser les images
jointes à une alerte et détecter les incohérences avec le texte.

Pipeline :
    1. Réception de l'image (JPEG/PNG)
    2. Encodage Base64 → envoi à Groq Vision
    3. Description de la scène (dangers, contexte, lieu)
    4. Comparaison avec la description textuelle de l'alerte
    5. Score de cohérence → verdict FIABLE / SUSPECTE / FRAUDE
"""

import base64
import logging
import json
from typing import Optional, Dict
from groq import Groq

from utils.config import settings

logger = logging.getLogger("ai-inference.vision")

# ── Prompt Système pour l'analyse visuelle ───────────────────────────────────
VISION_SYSTEM_PROMPT = """Tu es SOS-Vision, le module de vision par ordinateur du Centre National d'Urgence du Cameroun (CINU) à Yaoundé.

TON RÔLE : Analyser les images envoyées avec les alertes d'urgence pour :
1. DÉCRIRE la scène visible en détail
2. IDENTIFIER les dangers visibles (feu, fumée, accident, blessés, inondation, etc.)
3. DÉTECTER toute incohérence (fête au lieu d'incendie, lieu calme au lieu d'urgence)
4. Identifier les LIEUX reconnaissables de Yaoundé si possible

RÉPONDS EN JSON STRICT avec cette structure :
{
    "description_scene": "Description factuelle et détaillée de la scène",
    "dangers_detectes": ["danger1", "danger2"],
    "contexte_lieu": "Indications sur le lieu visible (urbain/rural, type de bâtiment, etc.)",
    "indicateurs_urgence": true/false,
    "indicateurs_calme": ["fête", "personnes souriantes", "lieu paisible"]
}

IMPORTANT : Sois FACTUEL. Ne déduis PAS de danger absent de l'image. Si l'image montre une scène calme, DIS-LE clairement."""

# ── Prompt pour la comparaison texte-image ───────────────────────────────────
COMPARISON_PROMPT_TEMPLATE = """Comparaison Anti-Fraude pour une alerte SOS-Cameroun.

ALERTE TEXTUELLE de l'utilisateur :
"{alert_text}"

ANALYSE VISUELLE de l'image jointe :
{image_analysis}

TÂCHE : Compare la description textuelle et l'analyse de l'image.
Détermine si l'image CONFIRME ou CONTREDIT la description de l'alerte.

Exemples de FRAUDE :
- L'utilisateur signale un "Incendie à Etoudi" mais la photo montre une fête
- L'utilisateur dit "Accident grave" mais la photo montre une rue calme
- L'utilisateur signale une "Inondation" mais la photo montre un temps sec

Exemples de COHÉRENCE :
- L'utilisateur signale un feu et l'image montre de la fumée/des flammes
- L'utilisateur signale un accident et l'image montre des véhicules endommagés

RÉPONDS EN JSON STRICT :
{{
    "coherent": true/false,
    "score_coherence": <0-100>,
    "explication": "Explication détaillée",
    "details_incoherence": "null ou description précise de l'incohérence"
}}"""


class VisionService:
    """
    Service d'analyse visuelle anti-fraude.

    Utilise Groq Vision (Llama 3.2 Vision) pour analyser les images
    et les comparer avec la description textuelle de l'alerte.
    """

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.vision_model = settings.GROQ_VISION_MODEL

        if not self.api_key:
            logger.error("❌ GROQ_API_KEY non configurée — Vision désactivée")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"✅ VisionService initialisé (Modèle : {self.vision_model})")

    def _encode_image(self, image_path: str) -> str:
        """
        Encode une image en Base64 pour l'API Groq Vision.

        Args:
            image_path: Chemin absolu vers le fichier image.

        Returns:
            Chaîne Base64 de l'image.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _detect_mime_type(self, image_path: str) -> str:
        """Détecte le type MIME basique de l'image à partir de l'extension."""
        ext = image_path.lower().rsplit(".", 1)[-1]
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        return mime_map.get(ext, "image/jpeg")

    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyse une image via Groq Vision et décrit la scène.

        Args:
            image_path: Chemin vers l'image à analyser.

        Returns:
            dict avec les clés : description_scene, dangers_detectes,
            contexte_lieu, indicateurs_urgence, indicateurs_calme.
        """
        if not self.client:
            raise RuntimeError("VisionService non initialisé : GROQ_API_KEY manquante.")

        logger.info(f"👁️ Analyse d'image : {image_path}")

        img_b64 = self._encode_image(image_path)
        mime = self._detect_mime_type(image_path)

        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {"role": "system", "content": VISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{img_b64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": "Analyse cette image pour une alerte d'urgence SOS-Cameroun. "
                                        "Décris la scène et identifie les dangers. Réponds en JSON.",
                            },
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=1024,
            )

            raw = response.choices[0].message.content
            logger.debug(f"Réponse Vision brute : {raw[:200]}...")

            # Parsing JSON (gestion des blocs markdown)
            if "```json" in raw:
                raw = raw.split("```json")[-1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            return json.loads(raw)

        except json.JSONDecodeError as e:
            logger.error(f"❌ Vision : réponse JSON invalide — {e}")
            return {
                "description_scene": raw if 'raw' in dir() else "Analyse échouée",
                "dangers_detectes": [],
                "contexte_lieu": "Indéterminé",
                "indicateurs_urgence": False,
                "indicateurs_calme": [],
            }
        except Exception as e:
            logger.error(f"❌ Erreur Groq Vision : {e}")
            raise RuntimeError(f"Groq Vision error: {str(e)}")

    def compare_text_image(self, alert_text: str, image_analysis: Dict) -> Dict:
        """
        Compare la description textuelle d'une alerte avec l'analyse visuelle
        de l'image jointe pour détecter les fraudes.

        Args:
            alert_text: Texte de l'alerte tel que déclaré par l'utilisateur.
            image_analysis: Résultat de ``analyze_image()`` (dict JSON).

        Returns:
            dict avec score_coherence (0-100), coherent (bool),
            explication, details_incoherence.
        """
        if not self.client:
            raise RuntimeError("VisionService non initialisé : GROQ_API_KEY manquante.")

        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            alert_text=alert_text,
            image_analysis=json.dumps(image_analysis, ensure_ascii=False, indent=2),
        )

        try:
            response = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,  # LLM texte pour la comparaison
                messages=[
                    {"role": "system", "content": "Tu es un expert anti-fraude du CINU Cameroun."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )

            raw = response.choices[0].message.content

            if "```json" in raw:
                raw = raw.split("```json")[-1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            result = json.loads(raw)
            logger.info(
                f"🔍 Comparaison texte-image : score={result.get('score_coherence')}, "
                f"cohérent={result.get('coherent')}"
            )
            return result

        except json.JSONDecodeError:
            logger.error("❌ Comparaison anti-fraude : réponse JSON invalide")
            return {
                "coherent": True,
                "score_coherence": 50,
                "explication": "Analyse de cohérence non concluante",
                "details_incoherence": None,
            }
        except Exception as e:
            logger.error(f"❌ Erreur comparaison texte-image : {e}")
            return {
                "coherent": True,
                "score_coherence": 50,
                "explication": f"Erreur lors de l'analyse : {str(e)}",
                "details_incoherence": None,
            }


# ── Singleton ────────────────────────────────────────────────────────────────
vision_service = VisionService()
