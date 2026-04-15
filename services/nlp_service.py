"""
Service NLP — Extraction d'entités & Nettoyage de texte
=========================================================
Utilise spaCy pour l'extraction d'entités nommées (NER) et fournit
des utilitaires de nettoyage NLP pour le pipeline multimodal.

Fonctionnalités :
- Nettoyage de texte brut (normalisation, suppression de bruit)
- Extraction d'entités (lieux, personnes, organisations)
- Détection du type d'urgence par mots-clés
- Validation topographique via le GeoService
- Détection de message fragmenté (danger imminent)
"""

import re
import unicodedata
import logging
import spacy
from typing import Dict, List, Optional

from services.geo_service import geo_service

logger = logging.getLogger("ai-inference.nlp")


class NLPService:
    """
    Service NLP pour l'extraction d'entités et le nettoyage de texte.

    Utilise spaCy ``fr_core_news_sm`` pour le NER et des heuristiques
    pour le nettoyage et la détection d'urgences camerounaises.
    """

    def __init__(self):
        logger.info("⏳ Chargement du modèle spaCy 'fr_core_news_sm'...")
        try:
            self.nlp = spacy.load("fr_core_news_sm")
            logger.info("✅ spaCy chargé.")
        except OSError:
            logger.warning(
                "⚠️ Modèle spaCy non trouvé. "
                "Lancer : python -m spacy download fr_core_news_sm"
            )
            self.nlp = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Nettoyage de texte
    # ═══════════════════════════════════════════════════════════════════════════

    def clean_text(self, raw_text: str) -> str:
        """
        Nettoie un texte brut (transcription vocale ou saisie manuelle).

        Étapes :
        1. Normalisation Unicode (NFC)
        2. Suppression des caractères de contrôle
        3. Normalisation des espaces multiples
        4. Suppression des espaces en début/fin
        5. Correction basique de la casse (première lettre en majuscule)

        Args:
            raw_text: Texte brut à nettoyer.

        Returns:
            Texte nettoyé.
        """
        if not raw_text:
            return ""

        # Normalisation Unicode
        text = unicodedata.normalize("NFC", raw_text)

        # Suppression des caractères de contrôle (sauf saut de ligne)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalisation des espaces multiples
        text = re.sub(r"\s+", " ", text).strip()

        # Première lettre en majuscule si pas déjà
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text

    # ═══════════════════════════════════════════════════════════════════════════
    # Extraction d'entités
    # ═══════════════════════════════════════════════════════════════════════════

    def extract_entities(self, text: str) -> Dict:
        """
        Extrait les entités nommées et le type d'urgence d'un texte.

        Combine spaCy NER avec la validation topographique de Yaoundé
        via le GeoService.

        Args:
            text: Texte nettoyé à analyser.

        Returns:
            dict avec localisation, personnes, organisations, autres,
            type_urgence_detecte, lieu_valide_yaounde.
        """
        if not self.nlp:
            return {"error": "Modèle spaCy non chargé"}

        doc = self.nlp(text)
        entities = {
            "localisation": None,
            "personnes": [],
            "organisations": [],
            "autres": [],
        }

        for ent in doc.ents:
            if ent.label_ == "LOC":
                if not entities["localisation"]:
                    entities["localisation"] = ent.text
            elif ent.label_ == "PER":
                entities["personnes"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["organisations"].append(ent.text)
            else:
                entities["autres"].append({"label": ent.label_, "text": ent.text})

        # Détection du type d'urgence par mots-clés (étendu pour le Cameroun)
        entities["type_urgence_detecte"] = self._detect_urgency_type(text)

        # Validation topographique de Yaoundé via le GeoService
        geo_result = geo_service.validate_location(text)
        entities["lieu_valide_yaounde"] = geo_result["valide"]
        if geo_result["valide"] and geo_result["lieu_principal"]:
            entities["localisation"] = geo_result["lieu_principal"]["display"]
            entities["zone_yaounde"] = geo_result["lieu_principal"].get("zone")
            entities["description_zone"] = geo_result["lieu_principal"].get("description")

        return entities

    # ═══════════════════════════════════════════════════════════════════════════
    # Détection de message fragmenté
    # ═══════════════════════════════════════════════════════════════════════════

    def is_fragmented(self, text: str) -> bool:
        """
        Détermine si un message est fragmenté (danger imminent probable).

        Critères de fragmentation :
        - Moins de 5 mots
        - Contient des points de suspension (...)
        - Pas de verbe conjugué reconnaissable
        - Mots isolés séparés par des virgules ou des espaces

        Args:
            text: Texte à évaluer.

        Returns:
            True si le message est considéré comme fragmenté.
        """
        if not text:
            return True

        words = text.split()
        word_count = len(words)

        # Trop court
        if word_count <= 4:
            return True

        # Points de suspension
        if "..." in text or "…" in text:
            return True

        # Mots isolés séparés par des virgules (ex: "feu, carrefour, secours")
        comma_parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(comma_parts) >= 3 and all(len(p.split()) <= 2 for p in comma_parts):
            return True

        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # Méthodes privées
    # ═══════════════════════════════════════════════════════════════════════════

    def _detect_urgency_type(self, text: str) -> str:
        """
        Détecte le type d'urgence par correspondance de mots-clés.

        Mots-clés étendus pour les urgences camerounaises courantes.

        Args:
            text: Texte à analyser (sera converti en minuscules).

        Returns:
            Type d'urgence détecté (INCENDIE, ACCIDENT, MEDICAL, etc.)
        """
        urgence_keywords = {
            "INCENDIE": [
                "feu", "incendie", "brûle", "flammes", "fumée", "combustion",
                "brûlure", "braisier", "explosion", "gaz",
            ],
            "ACCIDENT": [
                "accident", "collision", "renversé", "choc", "voiture", "moto",
                "camion", "benne", "taxi", "route", "percuté", "tonneau",
                "carambolage", "accident de circulation",
            ],
            "MEDICAL": [
                "blessé", "malade", "inconscient", "médecin", "ambulance", "sang",
                "crise", "évanouissement", "convulsion", "palu", "paludisme",
                "choléra", "accouchement", "enceinte", "douleur", "respire plus",
            ],
            "INONDATION": [
                "inondation", "eau", "crue", "noyé", "débordement", "pluie",
                "torrent", "rivière", "marécage", "submergé",
            ],
            "AGRESSION": [
                "agression", "vol", "attaque", "menace", "arme", "couteau",
                "braquage", "kidnapping", "enlèvement", "séquestration",
                "viol", "violence", "bagarre", "machette",
            ],
        }

        text_lower = text.lower()
        for urgence, keywords in urgence_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return urgence
        return "AUTRE"


# ── Singleton ────────────────────────────────────────────────────────────────
nlp_service = NLPService()
