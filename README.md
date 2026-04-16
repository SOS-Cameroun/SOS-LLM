---
title: SOS-Cameroun AI Inference
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# SOS-Cameroun - AI Inference Microservice

Ce service gère l'inférence pour le système SOS-Cameroun :
- STT (Whisper)
- LLM (Groq)
- Vision (Groq Vision)
- TTS (Edge-TTS)

## Déploiement local
```bash
docker build -t sos-ai .
docker run -p 7860:7860 --env-file .env sos-ai
```
