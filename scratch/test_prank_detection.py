import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.llm_service import llm_service
from services.nlp_service import nlp_service

def test_extraction(text):
    print(f"\n--- Testing: {text} ---")
    
    # 1. Test NLP keyword detection
    nlp_type = nlp_service._detect_urgency_type(text)
    print(f"NLP Detected Type: {nlp_type}")
    
    # 2. Test LLM extraction
    try:
        entities = llm_service.extract_entities(text)
        print(f"LLM Extracted Entities: {entities}")
        
        type_incident = entities.get("type_incident")
        gravite = entities.get("gravite")
        
        # 3. Test Reassurance Logic
        reassurance = llm_service.generate_reassurance_advice(type_incident, gravite, "LOW")
        print(f"Reassurance: {reassurance}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_cases = [
        "Il y a un gros incendie au Carrefour Vogt, venez vite !",
        "Je voudrais commander une pizza quatre fromages s'il vous plaît.",
        "Vous êtes tous des imbéciles, je rigole haha.",
        "Simple test de connexion, rien à signaler.",
        "Au secours, un accident de taxi à Mokolo !"
    ]
    
    for case in test_cases:
        test_extraction(case)
