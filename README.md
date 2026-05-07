---
title: SOS-Cameroun AI Inference
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# 🚑 SOS-Cameroun : Intelligence Artificielle d'Urgence (SOS-LLM)

Ce microservice est le cœur d'intelligence artificielle du projet **SOS-Cameroun**. Il s'agit d'un système multimodal capable de traiter des alertes d'urgence (Audio, Texte) en temps réel, spécifiquement optimisé pour la topographie et le contexte socio-linguistique de **Yaoundé**.

---

## 🏗️ Conception du Système (Architecture & Vision)

Le système repose sur une philosophie de **"Robustesse en milieu dégradé"**. En situation d'urgence, les messages sont souvent fragmentés, bruyants ou imprécis. Notre architecture traite cette incertitude via trois couches :

1.  **Couche de Perception (STT & Vision)** : Capte le signal brut.
2.  **Couche Cognitive (Fine-Tuned LLM)** : Corrige, interprète et restructure l'information.
3.  **Couche de Validation (Geo-Inference & Anti-Fraude)** : Vérifie la cohérence spatiale et visuelle.

```mermaid
graph TD
    subgraph "Entrées Multimodales"
        A[Note Vocale] --> B[Whisper STT]
        T[Message Texte] --> C[NLP Cleaning]
        I[Photo Terrain] --> D[Groq Vision]
    end

    subgraph "Cœur d'Intelligence (SOS-LLM)"
        B --> LLM{Llama 3 Fine-Tuned}
        C --> LLM
        LLM --> |Extraction| E[JSON Structuré]
        LLM --> |Correction| F[Repair Transcription]
    end

    subgraph "Validation & Action"
        E --> G[GeoService: Yaoundé Map]
        D --> H[Score Anti-Fraude]
        G --> J[Alerte Validée]
        H --> J
    end

    J --> K[RabbitMQ / Spring Boot]
    J --> L[Edge-TTS: Voix de Réconfort]
```

---

## 🧠 Technologies de l'Entraînement (Fine-Tuning)

L'intelligence du système ne repose pas sur un modèle générique, mais sur un modèle **Llama 3 8B** spécialisé via un entraînement spécifique.

### 🛠️ Stack d'Entraînement
*   **Modèle de base** : `Meta-Llama-3-8B`.
*   **Framework d'optimisation** : **Unsloth** (permet un entraînement 2x plus rapide et 70% moins gourmand en VRAM).
*   **Méthode** : **LoRA (Low-Rank Adaptation)** avec un rang $r=16$ et $\alpha=16$.
*   **Quantification** : **4-bit (BitsAndBytes)** pour permettre l'inférence sur des infrastructures légères.
*   **Environnement** : Kaggle (GPU T4) / Hugging Face.

### 📊 Dataset & "Bruitage Intentionnel"
Le modèle a été entraîné sur un dataset JSONL de ~200 exemples incluant :
*   Des transcriptions phonétiques de l'accent camerounais.
*   Des fautes de frappe typiques des situations de panique.
*   Un ancrage géographique strict sur les 35 quartiers majeurs de Yaoundé (Mokolo, Bastos, Mvan, etc.).

---

## 🔄 Le "Mélange" : Intégration NLP & LLM

L'une des innovations majeures de ce projet est la fusion entre le **NLP classique** et le **LLM (Large Language Model)** :

### 1. Le Pipeline de "Réparation" (Whisper ➡️ Llama)
Les modèles STT (Whisper) font souvent des erreurs sur les noms de lieux locaux (ex: "Aubilly" au lieu de "Obili").
*   **Action** : Le texte brut de Whisper est passé au LLM avec un *System Prompt* agissant comme un correcteur topographique. Le LLM "hallucine intelligemment" la correction correcte en se basant sur sa connaissance de Yaoundé.

### 2. L'Analyse de Stress Hybride
Nous combinons deux métriques pour évaluer l'urgence :
*   **Analyse Textuelle** : Le LLM détecte les mots de panique.
*   **Analyse Acoustique** : Le score de confiance de Whisper et l'intensité sonore sont mixés pour ajuster le ton de la réponse TTS.

### 3. Vision Anti-Fraude (Llama 3.2 Vision)
Pour éviter les fausses alertes, nous utilisons **Groq Vision** pour comparer le texte de l'alerte avec l'image envoyée. Si un utilisateur signale un "Incendie" mais envoie une photo d'une rue calme, le système génère un score de fiabilité faible.

---

## 🛠️ Stack Technique

*   **Backend** : FastAPI (Asynchrone).
*   **IA Inférence** : Groq API (Llama 3.3 70B) pour la rapidité (< 500ms).
*   **NLP** : SpaCy (NER) pour l'extraction rapide d'entités.
*   **STT** : OpenAI Whisper (Faster-Whisper).
*   **TTS** : Microsoft Edge-TTS (Synthèse vocale émotionnelle).
*   **Base de données** : Supabase (Stockage des vecteurs et images).
*   **Messaging** : RabbitMQ (Liaison avec le backend central).

---

## 🚀 Installation & Développement

### Prérequis
- Python 3.12+
- Docker & Docker Compose (optionnel)
- Clé API Groq

### Setup Local
1. Clonez le dépôt et créez un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configurez votre `.env` :
   ```env
   GROQ_API_KEY=votre_cle_groq
   RABBITMQ_URL=amqp://guest:guest@localhost:5672/
   ```
3. Lancez le serveur :
   ```bash
   uvicorn main:app --reload --port 8001
   ```

---

## 🚢 Déploiement

### Docker
```bash
docker-compose up -d --build
```

### Hugging Face Spaces
Ce projet est compatible avec Hugging Face Spaces (SDK Docker).
- **RAM recommandée** : 16 Go (CPU).
- **Port** : 7860 (configuré dans le `Dockerfile`).

---

## 📞 API Overview
- `POST /stt/transcribe` : Audio -> Texte
- `POST /llm/extract` : Extraction d'entités JSON
- `POST /llm/summarize_for_tts` : JSON -> Texte fluide pour TTS
- `POST /vision/analyze` : Analyse d'image anti-fraude
- `POST /tts/synthesize` : Texte -> Audio MP3

---
*Développé par PJDPL4*
