import logging
import os
import json
from groq import Groq
from utils.config import settings

logger = logging.getLogger("ai-inference.llm")

class LLMService:
    def __init__(self):
        # On utilise Groq (Online) pour l'inférence
        
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL 
        
        if not self.api_key:
            logger.error("❌ GROQ_API_KEY non configurée dans le fichier .env")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"✅ Service LLM Groq initialisé (Modèle : {self.model})")

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """
        Envoie un prompt à l'API Groq Cloud
        """
        if not self.client:
            raise RuntimeError("Groq client not initialized. Check GROQ_API_KEY.")

        logger.info(f"🧠 Appel LLM Groq ({self.model}) en cours...")
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Erreur Groq API : {e}")
            raise RuntimeError(f"Groq API error: {str(e)}")

    def extract_entities(self, text: str) -> dict:
        """
        Utilise le LLM pour extraire le type d'incident et la gravité (UC14, UC63).
        Force une sortie JSON.
        """
        system_p = (
            "Tu es un assistant IA d'urgence médical et sécuritaire. "
            "Extrait le type_incident (ex: 'incendie', 'malaise', 'accident', 'agression'), "
            "la gravite (cririque, haute, moyenne, faible) et le lieu si mentionné du texte fourni."
            "Réponds UNIQUEMENT avec un objet JSON valide contenant ces 3 clés."
        )
        try:
            res = self.generate_response(text, system_p)
            # Nettoyage grossier pour parser le JSON au cas où le modèle ajoute du texte
            if "```json" in res:
                res = res.split("```json")[-1].split("```")[0].strip()
            
            return json.loads(res)
        except Exception as e:
            logger.error(f"Failed to parse entities JSON: {e}")
            return {"type_incident": "Inconnu", "gravite": "Moyenne", "lieu": None}

    def process_voice_action(self, text: str) -> dict:
        """
        Transforme une commande vocale brute en une action structurée (UC52).
        """
        system_p = (
            "Tu écoutes un agent de centre d'urgence. "
            "Convertis son ordre en JSON avec la clé 'action' (parmi: affecter_agent, rejeter_alerte, valider_alerte) "
            "et 'parametres' (ex: id de l'alerte). Renvoie UNIQUEMENT le JSON."
        )
        try:
            res = self.generate_response(text, system_p)
            if "```json" in res:
                res = res.split("```json")[-1].split("```")[0].strip()
            return json.loads(res)
        except Exception as e:
            logger.error(f"Failed to parse action JSON: {e}")
            return {"action": "inconnu", "parametres": {}}

# Instance Singleton
llm_service = LLMService()
