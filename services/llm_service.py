"""
Service LLM — Inférence Groq Ultra-Rapide
============================================================
Gère toutes les interactions avec l'API Groq pour le traitement du langage :
- Extraction d'entités d'incident (type, gravité, lieu)
- Hallucination intelligente des messages fragmentés (danger imminent)
- Analyse du niveau de stress émotionnel
- Génération de réponses d'urgence adaptées
- Scoring Anti-Fraude (Social Validation)

System prompt ancré sur la ville de Yaoundé.
"""

import logging
import json
from typing import Optional, Dict, Any

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
Indicateurs acoustiques : {acoustic_data}
Réponds en JSON strict :
{{
    "niveau": "<LOW|MEDIUM|HIGH|CRITICAL>",
    "score": <0.0-1.0>,
    "indicateurs": ["panique", "cris", etc.]
}}
\"\"\""""

REPAIR_TRANSCRIPTION_PROMPT = """\"\"\"
Tu es SOS-Rédacteur, un expert en correction de transcriptions vocales pour le CINU Cameroun.
Ta tâche est de corriger les erreurs phonétiques et grammaticales issues d'une transcription Whisper.

VOICI LES LIEUX VALIDES À YAOUNDÉ (Utilise cette orthographe en priorité) :
{known_places}

CONSIGNES :
1. Corrige les mots phonétiquement proches du langage courant (ex: "allaitre" -> "alerte").
2. RECONNAISSANCE GÉOGRAPHIQUE : Si un mot dans la transcription ressemble phonétiquement à un lieu de la liste (ex: "Aubilly", "Hobili" -> "Obili"), remplace-le IMPÉRATIVEMENT par l'orthographe exacte de la liste.
3. Préserve les informations critiques (nombres, types de blessures).
4. Rends le texte fluide, clair et professionnel pour les autorités.
5. RÉPONDS UNIQUEMENT PAR LE TEXTE CORRIGÉ.
\"\"\""""


class LLMService:
    """Service d'inférence LLM via Groq Cloud (Llama 3.1 70B)."""

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model = "llama-3.3-70b-versatile"  # Modèle spécifique demandé

        if not self.api_key:
            logger.error("❌ GROQ_API_KEY non configurée")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"✅ Service LLM Groq initialisé (Expert Yaoundé)")

    def generate_response(self, prompt: str, system_prompt: str = "", json_mode: bool = False) -> str:
        if not self.client: raise RuntimeError("Groq client not initialized")
        
        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt or SYSTEM_PROMPT_YAONDE},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
            }
            
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
                
            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Erreur Groq API : {e}")
            raise

    def extract_entities(self, text: str) -> dict:
        prompt = f"{EXTRACTION_PROMPT}\n\nMessage : \"\"\"{text}\"\"\""
        res = self.generate_response(prompt, json_mode=True)
        return self._parse_json(res, {"type_incident": "AUTRE", "gravite": "Moyenne"})

    def hallucinate_completion(self, fragment: str, context: str = "") -> dict:
        prompt = HALLUCINATION_PROMPT_TEMPLATE.format(fragment=fragment, geo_context=context)
        res = self.generate_response(prompt, json_mode=True)
        return self._parse_json(res, {"texte_complete": fragment})

    def analyze_stress_level(self, text: str, tone_score: Optional[float] = None, acoustic_indicators: Optional[list] = None) -> dict:
        acoustic_data = "Aucun"
        if tone_score is not None:
            acoustic_data = f"Tone Score: {tone_score}, Indicateurs: {', '.join(acoustic_indicators or [])}"

        prompt = STRESS_ANALYSIS_PROMPT.format(text=text, acoustic_data=acoustic_data)
        res = self.generate_response(prompt, json_mode=True)
        
        result = self._parse_json(res, {"niveau": "MEDIUM", "score": 0.5})
        
        # Si un tone_score est présent, on l'utilise pour pondérer le score final du LLM
        if tone_score is not None:
            # On donne 40% de poids à l'acoustique et 60% au texte analysé par le LLM
            llm_score = result.get("score", 0.5)
            final_score = (llm_score * 0.6) + (tone_score * 0.4)
            result["score"] = round(final_score, 2)
            
            # Ajustement du niveau si le score final change de catégorie
            if final_score < 0.3: result["niveau"] = "LOW"
            elif final_score < 0.6: result["niveau"] = "MEDIUM"
            elif final_score < 0.8: result["niveau"] = "HIGH"
            else: result["niveau"] = "CRITICAL"
            
            # Ajout des indicateurs acoustiques
            if acoustic_indicators:
                result["indicateurs"] = list(set(result.get("indicateurs", []) + acoustic_indicators))

        return result

    def repair_transcription(self, text: str, known_places: str = "") -> str:
        """
        Répare les erreurs de transcription STT (phonétique, grammaire).
        """
        prompt = f"{REPAIR_TRANSCRIPTION_PROMPT.format(known_places=known_places)}\n\nTexte à corriger : \"{text}\""
        # On utilise le mode texte libre (json_mode=False par défaut)
        return self.generate_response(prompt)

    def process_voice_action(self, text: str) -> dict:
        """
        Parse une commande vocale d'un agent en action structurée.
        """
        prompt = f"""
        Analyse la commande suivante d'un agent de secours et convertis-la en JSON.
        Actions possibles : affecter_agent, rejeter_alerte, valider_alerte.
        Commande : "{text}"
        
        Réponds UNIQUEMENT en JSON :
        {{
            "action": "<nom_action>",
            "parametres": {{ "id_alerte": "<id>", "agent": "<nom_si_dispo>" }}
        }}
        """
        res = self.generate_response(prompt, json_mode=True)
        return self._parse_json(res, {"action": "unknown", "parametres": {}})

    def generate_tts_response(self, structured_data: dict) -> str:
        """
        Convertit des données JSON en texte fluide "bonne et due forme" pour le TTS.
        """
        prompt = f"""
        Convertis ces informations d'urgence en un paragraphe fluide et rassurant pour une victime.
        Le texte doit être prêt à être lu par une synthèse vocale (TTS).
        Informations : {json.dumps(structured_data, ensure_ascii=False)}
        
        Règle : Pas de listes, pas de JSON, juste un discours continu et calme.
        """
        # On désactive le mode JSON ici pour avoir du texte brut
        return self.generate_response(prompt)

    def generate_reassurance_advice(self, type_incident: str, gravite: str, stress_level: str) -> str:
        """
        Génère un message de réassurance et des conseils de premiers secours 
        spécifiques au type d'incident.
        """
        prompt = f"""
        Tu es une assistante sociale experte en gestion de crise au Cameroun. 
        Une victime vient de signaler un incident de type {type_incident} avec une gravité {gravite}. 
        Son niveau de stress est {stress_level}.
        
        Tâche : 
        1. RASSURE la victime avec des mots calmes et bienveillants.
        2. Donne 2 ou 3 CONSEILS DE SÉCURITÉ immédiats et simples (ex: 'éloignez-vous du feu', 'restez allongé').
        3. Confirme que les secours ont été informés.
        
        Contrainte : Sois concis (max 3 phrases), utilise un ton très humain et rassurant.
        Réponds directement par le texte à lire à la victime.
        """
        return self.generate_response(prompt)

    @staticmethod
    def _parse_json(raw: str, fallback: dict) -> dict:
        try:
            return json.loads(raw)
        except Exception:
            return fallback

llm_service = LLMService()
