import asyncio
import os
import sys
import logging

# Ajouter le chemin du projet pour l'import des services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.tts_service import tts_service
from services.supabase_service import supabase_service

logging.basicConfig(level=logging.INFO)

async def test_tts_and_upload():
    print("🚀 Test de génération TTS et upload Supabase...")
    
    text = "Ceci est un test de génération audio pour le système SOS Cameroun. Tout va bien se passer."
    stress_level = "LOW"
    
    try:
        # 1. Génération Audio
        print(f"🔊 Génération audio pour : '{text}'")
        result = await tts_service.synthesize_emergency(text, stress_level)
        
        print(f"✅ Résultat TTS : {result}")
        
        audio_url = result.get("audio_url")
        if audio_url:
            print(f"✨ URL Publique Supabase : {audio_url}")
        else:
            print("❌ L'URL audio est manquante dans le résultat.")
            
    except Exception as e:
        print(f"💥 Erreur lors du test : {e}")

if __name__ == "__main__":
    asyncio.run(test_tts_and_upload())
