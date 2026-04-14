#!/bin/bash
# =============================================================================
# SOS-Cameroun - Script d'installation du microservice Python AI Inference
# =============================================================================
# Prérequis : Python 3.10+, pip, ffmpeg
# Usage: chmod +x setup_ai_inference.sh && ./setup_ai_inference.sh

set -e

echo "=============================================="
echo " SOS-Cameroun - Setup AI Inference Service"
echo "=============================================="

# --- 1. Vérifier Python ---
PYTHON=$(command -v python3.10 || command -v python3.11 || command -v python3)
if [ -z "$PYTHON" ]; then
    echo "[ERREUR] Python 3.10+ est requis."
    exit 1
fi
echo "[OK] Python détecté : $($PYTHON --version)"

# --- 2. Vérifier / installer ffmpeg (requis par Whisper) ---
if ! command -v ffmpeg &> /dev/null; then
    echo "[INFO] Installation de ffmpeg..."
    sudo apt-get update -q && sudo apt-get install -y ffmpeg
else
    echo "[OK] ffmpeg déjà installé"
fi

# --- 3. Créer l'environnement virtuel ---
AI_DIR="$(dirname "$0")"
VENV_DIR="$AI_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Création du venv Python dans $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
fi

echo "[INFO] Activation du venv..."
source "$VENV_DIR/bin/activate"

# --- 4. Mettre à jour pip ---
echo "[INFO] Mise à jour de pip..."
pip install --upgrade pip --quiet

# --- 5. Installer les dépendances ---
echo "[INFO] Installation des dépendances Python..."
pip install -r "$AI_DIR/requirements.txt"

# --- 6. Télécharger le modèle spaCy français ---
echo "[INFO] Téléchargement du modèle spaCy (fr_core_news_sm)..."
python -m spacy download fr_core_news_sm

# --- 7. Créer le fichier .env si absent ---
ENV_FILE="$AI_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "[INFO] Création du fichier .env..."
    cat > "$ENV_FILE" <<EOF
# Configuration du microservice AI Inference
SPRING_BACKEND_URL=http://localhost:8080
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
REDIS_URL=redis://localhost:6379
WHISPER_MODEL=base            # Options: tiny, base, small, medium, large
TTS_VOICE=fr-FR-DeniseNeural  # Options intéressantes: fr-FR-HenriNeural, fr-FR-DeniseNeural
PORT=8001
LOG_LEVEL=INFO
EOF
    echo "[OK] Fichier .env créé. Veuillez le remplir avec vos paramètres."
fi

echo ""
echo "=============================================="
echo " Installation terminée avec succès!"
echo "=============================================="
echo ""
echo "Prochaines étapes :"
echo "  1. Éditer ai-inference/.env"
echo "  2. Installer Ollama : https://ollama.ai"
echo "  3. Tirer le modèle : ollama pull mistral:7b"
echo "  4. Démarrer le service : source venv/bin/activate && uvicorn main:app --reload --port 8001"
