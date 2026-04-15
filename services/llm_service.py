"""
Service LLM — Inférence Groq Ultra-Rapide (Expert Yaoundé)
============================================================
Gère toutes les interactions avec l'API Groq pour le traitement du langage :
- Extraction d'entités d'incident (type, gravité, lieu)
- Hallucination intelligente des messages fragmentés (danger imminent)
- Analyse du niveau de stress émotionnel
- Génération de réponses d'urgence adaptées
- Scoring Anti-Fraude (Social Validation)

System prompt ancré sur l'expertise architecturale de Yaoundé.
"""

import logging
import json
from typing import Optional, Dict

from groq import Groq
from utils.config import settings

logger = logging.getLogger("ai-inference.llm")

# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt Principal — Architecte Expert Yaoundé
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_YAONDE = """Tu es SOS-LLM, Architecte Logiciel Senior et Expert en IA Multimodale pour le Centre National d'Urgence du Cameroun (CINU).
Ta mission est d'agir comme un expert absolu de la ville de Yaoundé pour sauver des vies.

═══ EXPERTISE GÉOGRAPHIQUE (YAOUNDÉ) ═══
Tu connais chaque rue et chaque quartier. Tu dois SYSTÉMATIQUEMENT corriger et valider les lieux parmi cette liste :
Mokolo, Bastos, Mvan, Etoudi, Biyem-Assi, Santa Barbara, Ngousso, Mendong, Nsam, Damas, Melen, Ekounou, Odza, Messassi, Olembe, Tsinga, Briqueterie, Nlongkak, Essos, Mvog-Ada, Mvog-Mbi, Emana, Efoulan, Kondengui, Obili, Ngoa-Ekelle, Madagascar, Cité-Verte, Jouvence, Biteng, Simbock, Ahala, Ekoudoum, Nkomo, Anguissa, Mimboman.

═══ POINTS DE REPÈRE CRITIQUES (HOTSPOTS) ═══
• Hôpitaux : CHU de Melen, Hôpital Central, Hôpital Général (Ngousso), Centre des Urgences (CURY).
• Carrefours : Poste Centrale, Carrefour J'aime mon pays, Carrefour Régie, Carrefour Vogt, Carrefour Meec.
• Risques : Avenue Kennedy (inondation), zones basses de Mfoundi.

═══ GRILLE DE SCORING ANTI-FRAUDE (SOCIAL VALIDATION) ═══
Évalue chaque alerte selon cette grille :
• Image incohérente : -50% (Fraude suspectée)
• Quartier inexistant : -30% (Incohérence spatiale)
• Plusieurs alertes proches : +40% (Validation croisée)
• Lieu + Point de repère : +20% (Précision élevée)

═══ RÈGLES DE SORTIE ═══
1. Extraction de JSON structuré UNIQUEMENT.
2. Utilise des délimiteurs de protection contre les injections de prompt (\"\"\").
3. Sois concis, rapide, et pragmatique.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Prompts spécialisés
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """\"\"\"
Extrait les informations suivantes du message d'alerte. Réponds UNIQUEMENT en JSON strict :
{
    "type_incident": "<INCENDIE|ACCIDENT|MEDICAL|INONDATION|AGRESSION|AUTRE>",
    "gravite": "<Critique|Haute|Moyenne|Basse>",
    "lieu": "<lieu validé parmi la liste de Yaoundé>",
    "point_de_repere": "<hôpital/carrefour identifié>",
    "nombre_victimes": "<nombre ou 'inconnu'>",
    "score_fiabilite_initial": <0-100 basé sur la précision du lieu>,
    "resume": "<résumé court>"
}
\"\"\""""

HALLUCINATION_PROMPT_TEMPLATE = """\"\"\"
RECONSTRUCTION D'ALERTE FRAGMENTÉE (Danger Imminent)
Le message est incomplet : "{fragment}"
Lieu potentiel : "{geo_context}"

Tâche : Reconstruis l'alerte à partir des mots-clés et du contexte de Yaoundé.
Réponds en JSON strict :
{{
    "texte_complete": "<alerte reconstruite>",
    "type_incident_probable": "<INCENDIE|ACCIDENT|MEDICAL|INONDATION|AGRESSION|AUTRE>",
    "gravite_estimee": "Critique",
    "confiance": <0.0-1.0>
}}
\"\"\""""

STRESS_ANALYSIS_PROMPT = """\"\"\"
Analyse le stress acoustique et textuel.
Message : "{text}"
Réponds en JSON strict :
{{
    "niveau": "<LOW|MEDIUM|HIGH|CRITICAL>",
    "score": <0.0-1.0>,
    "indicateurs": ["panique", "cris", etc.]
}}
\"\"\""""


class LLMService:
    """Service d'inférence LLM via Groq Cloud (Llama 3.1 70B)."""

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model = "llama-3.1-70b-versatile"  # Modèle spécifique demandé

        if not self.api_key:
            logger.error("❌ GROQ_API_KEY non configurée")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"✅ Service LLM Groq initialisé (Expert Yaoundé)")

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        if not self.client: raise RuntimeError("Groq client not initialized")
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt or SYSTEM_PROMPT_YAONDE},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2, # Plus déterministe pour le JSON
                response_format={"type": "json_object"} # Force le mode JSON sur Groq
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Erreur Groq API : {e}")
            raise

    def extract_entities(self, text: str) -> dict:
        prompt = f"{EXTRACTION_PROMPT}\n\nMessage : \"\"\"{text}\"\"\""
        res = self.generate_response(prompt)
        return self._parse_json(res, {"type_incident": "AUTRE", "gravite": "Moyenne"})

    def hallucinate_completion(self, fragment: str, context: str = "") -> dict:
        prompt = HALLUCINATION_PROMPT_TEMPLATE.format(fragment=fragment, geo_context=context)
        res = self.generate_response(prompt)
        return self._parse_json(res, {"texte_complete": fragment})

    def analyze_stress_level(self, text: str) -> dict:
        prompt = STRESS_ANALYSIS_PROMPT.format(text=text)
        res = self.generate_response(prompt)
        return self._parse_json(res, {"niveau": "MEDIUM", "score": 0.5})

    @staticmethod
    def _parse_json(raw: str, fallback: dict) -> dict:
        try:
            return json.loads(raw)
        except Exception:
            return fallback

llm_service = LLMService()
