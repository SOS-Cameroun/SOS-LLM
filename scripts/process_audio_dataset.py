import os
import sys
import json
import logging
import random
import glob
from dotenv import load_dotenv

# Ajouter le répertoire racine au PYTHONPATH pour éviter le ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Charger les variables d'environnement avant d'importer les services
load_dotenv()

from services.stt_service import stt_service
from services.llm_service import llm_service

# Configuration du Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dataset-processor")

# Dossiers sources et destinations
AUDIO_DIR = "data/training/audio"
OUTPUT_FILE = "data/training/text/dataset.jsonl"

# Liste des instructions types trouvées dans le dataset original
INSTRUCTIONS = [
    "Traite ce signalement d'urgence provenant de Yaoundé.",
    "Donne les consignes de sécurité pour cette alerte.",
    "Analyse ce signalement et indique la marche à suivre.",
    "Analyse cette alerte et donne la marche à suivre."
]

def process_audios():
    # Créer le dossier parent de la sortie si inexistant
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Récupérer tous les fichiers audio
    audio_files = []
    for ext in ['*.ogg', '*.wav', '*.mp3', '*.m4a']:
        audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, ext)))
    
    if not audio_files:
        logger.warning(f"⚠️ Aucun fichier audio trouvé dans {AUDIO_DIR}")
        return

    logger.info(f"🚀 Début du traitement de {len(audio_files)} fichiers...")

    results = []
    
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        logger.info(f"🎤 Traitement de : {filename}")
        
        try:
            # 1. Transcription brute (sans correction)
            stt_result = stt_service.transcribe(audio_path)
            raw_text = stt_result["text"].strip()
            
            if not raw_text:
                logger.warning(f"⏩ Texte vide pour {filename}, passage au suivant.")
                continue

            # 2. Extraction des entités pour générer la réponse
            # On utilise le LLM pour comprendre ce qui est dit
            entities = llm_service.extract_entities(raw_text)
            type_incident = entities.get("type_incident", "AUTRE")
            gravite = entities.get("gravite", "Moyenne")
            
            # 3. Analyse de stress (optionnel mais utile pour la réponse)
            stress_result = llm_service.analyze_stress_level(raw_text)
            stress_level = stress_result.get("niveau", "MEDIUM")
            
            # 4. Génération de l'output (réponse rassurante)
            output_text = llm_service.generate_reassurance_advice(type_incident, gravite, stress_level)
            
            # 5. Construction de l'objet JSONL
            entry = {
                "instruction": random.choice(INSTRUCTIONS),
                "input": raw_text,
                "output": output_text
            }
            
            results.append(entry)
            logger.info(f"✅ Succès : {filename}")

        except Exception as e:
            logger.error(f"❌ Erreur sur {filename} : {e}")

    # 6. Écriture dans le fichier JSONL (en mode append pour ne pas tout écraser si l'utilisateur veut fusionner)
    try:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"🏁 Traitement terminé. {len(results)} lignes ajoutées à {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'écriture du fichier : {e}")

if __name__ == "__main__":
    process_audios()
