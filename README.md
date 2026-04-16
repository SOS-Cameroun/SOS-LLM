---
title: SOS-Cameroun AI Inference
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# 🚑 SOS-Cameroun AI Inference Microservice

Ce microservice est le cœur d'intelligence artificielle du projet **SOS-Cameroun**. Il est conçu pour traiter les alertes d'urgence de manière multimodale (Audio, Texte, Image) et fournir une assistance intelligente en temps réel centrée sur la ville de **Yaoundé**.

## 🏗️ Architecture Technique

Le service est construit avec **FastAPI** et s'intègre asynchronement via **RabbitMQ** avec le backend Spring Boot principal.

```mermaid
graph TD
    User((Victime)) -->|Audio/Texte/Image| API(FastAPI)
    API --> STT[Whisper: Audio to Text]
    STT --> LLM[Groq: Analyse & Hallucination]
    LLM --> Vision[Groq Vision: Anti-Fraude]
    Vision --> TTS[Edge-TTS: Voix de réconfort]
    TTS --> User
    
    SB(Spring Boot) <-->|RabbitMQ| API
```

## 🚀 Fonctionnalités Clés

### 1. 🎤 Reconnaissance Vocale (STT)
Transformation des alertes vocales en texte via **faster-whisper**. Le système supporte la détection automatique de la langue.

### 2. 🧠 Intelligence Multimodale (LLM)
Utilise **Groq (Llama 3.3 70B)** pour :
- **Extraction d'entités** : Identification automatique du type d'incident, de la gravité et du lieu.
- **Hallucination intelligente** : Reconstruction de messages fragmentés en cas de mauvaise réception.
- **Agent Actions** : Conversion de commandes vocales d'agents en actions structurées (ex: "Affecter agent").

### 3. 👁️ Analyse de Vision & Anti-Fraude
Analyse des photos envoyées par les victimes via **Groq Vision** pour :
- Vérifier la cohérence entre le texte et l'image (détection de fraude).
- Identifier les dangers réels sur le terrain.

### 4. 🗣️ Synthèse Vocale Émotionnelle (TTS)
Génération de réponses vocales rassurantes via **Edge-TTS**. 
- **Adaptation au stress** : Le ton et le débit de parole s'adaptent automatiquement au niveau de stress détecté chez la victime.

## 🛠️ Installation & Développement

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

## 🚢 Déploiement

### Docker
```bash
docker-compose up -d --build
```

### Hugging Face Spaces
Ce projet est compatible avec Hugging Face Spaces (SDK Docker).
- **RAM recommandée** : 16 Go (CPU).
- **Port** : 7860 (configuré dans le `Dockerfile`).

## 📞 API Overview
- `POST /stt/transcribe` : Audio -> Texte
- `POST /llm/extract` : Extraction d'entités JSON
- `POST /llm/summarize_for_tts` : JSON -> Texte fluide pour TTS
- `POST /vision/analyze` : Analyse d'image anti-fraude
- `POST /tts/synthesize` : Texte -> Audio MP3
