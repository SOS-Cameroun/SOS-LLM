"""
Service NLP (spaCy) — extraction d'entités
UC14 : pré-remplissage automatique du formulaire d'alerte
"""
import spacy
from loguru import logger


class NLPService:
    def __init__(self):
        logger.info("Chargement du modèle spaCy 'fr_core_news_sm'...")
        try:
            self.nlp = spacy.load("fr_core_news_sm")
            logger.info("✅ spaCy chargé.")
        except OSError:
            logger.warning("⚠️  Modèle spaCy non trouvé. Lancer : python -m spacy download fr_core_news_sm")
            self.nlp = None

    def extract_entities(self, text: str) -> dict:
        """
        Extrait les entités nommées pour pré-remplir le formulaire Alerte.
        Labels spaCy FR : LOC (lieu), PER (personne), ORG (organisation), MISC
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
                # Le premier lieu trouvé = localisation principale
                if not entities["localisation"]:
                    entities["localisation"] = ent.text
            elif ent.label_ == "PER":
                entities["personnes"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["organisations"].append(ent.text)
            else:
                entities["autres"].append({"label": ent.label_, "text": ent.text})

        # Détection basique du type d'urgence par mots-clés
        urgence_keywords = {
            "INCENDIE": ["feu", "incendie", "brûle", "flammes", "fumée"],
            "ACCIDENT": ["accident", "collision", "renversé", "choc", "voiture"],
            "MEDICAL": ["blessé", "malade", "inconscient", "médecin", "ambulance", "sang"],
            "INONDATION": ["inondation", "eau", "crue", "noyé", "débordement"],
            "AGRESSION": ["agression", "vol", "attaque", "menace", "arme"],
        }
        text_lower = text.lower()
        type_urgence = "AUTRE"
        for urgence, keywords in urgence_keywords.items():
            if any(kw in text_lower for kw in keywords):
                type_urgence = urgence
                break

        entities["type_urgence_detecte"] = type_urgence
        return entities


# Singleton
nlp_service = NLPService()
