
# ============================================================
# SOS-CAMEROUN — NOTEBOOK COMPLET DE FINE-TUNING (Google Colab)
# Pipeline Multimodal : Audio (Whisper STT) + Texte → Llama 3
# ============================================================
# INSTRUCTIONS :
# 1. Copiez chaque cellule dans une cellule séparée de votre Notebook Colab.
# 2. Exécutez-les dans l'ordre : Cellule 1, 2, 3, ...
# 3. Activez le GPU T4 : Exécution > Modifier le type d'exécution > GPU T4
# ============================================================


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELLULE 0 — Installation des dépendances (À faire en 1er)  ║
# ╚══════════════════════════════════════════════════════════════╝

"""
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
!pip install openai-whisper librosa soundfile
!pip install datasets transformers
"""

# NOTE : Redémarrez votre session Colab après l'installation (Runtime > Restart session)
# Puis REPRENEZ à partir de la Cellule 1 sans ré-exécuter la Cellule 0.


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELLULE 1 — Connexion à Hugging Face                       ║
# ╚══════════════════════════════════════════════════════════════╝

from huggingface_hub import login

# Remplacez par votre token Hugging Face
# Obtenez-le sur : https://huggingface.co/settings/tokens
HF_TOKEN = "hf_VOTRE_TOKEN_ICI"
login(token=HF_TOKEN)
print("✅ Connecté à Hugging Face avec succès !")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELLULE 2 — Chargement du modèle Llama 3 avec Unsloth      ║
# ╚══════════════════════════════════════════════════════════════╝

from unsloth import FastLanguageModel
import torch

# --- Paramètres du modèle ---
max_seq_length = 2048   # Longueur max des séquences (contexte)
dtype = None            # Détection automatique (float16 sur T4, bfloat16 sur A100)
load_in_4bit = True     # Quantification 4-bit → réduit la mémoire GPU de 75%

print("⏳ Chargement de Llama 3 8B (peut prendre 3-5 minutes)...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = "unsloth/llama-3-8b-bnb-4bit",  # Version optimisée Unsloth
    max_seq_length  = max_seq_length,
    dtype           = dtype,
    load_in_4bit    = load_in_4bit,
    token           = HF_TOKEN,
)

print("✅ Modèle Llama 3 8B chargé avec succès !")
print(f"   GPU utilisé : {torch.cuda.get_device_name(0)}")
print(f"   Mémoire GPU disponible : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELLULE 3 — Configuration LoRA (Adaptateurs d'Apprentissage)║
# ╚══════════════════════════════════════════════════════════════╝

# LoRA = Low-Rank Adaptation
# Au lieu de modifier les 8 milliards de paramètres, on en modifie ~1%.
# Cela rend l'entraînement 10x plus rapide et possible sur GPU gratuit.

model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,    # Rang LoRA. 16 = bon équilibre vitesse/qualité
    lora_alpha     = 16,    # Amplitude des adaptateurs (= r pour simplifier)
    lora_dropout   = 0,     # Pas de dropout pour stabilité
    bias           = "none",
    use_gradient_checkpointing = "unsloth",  # Économise encore de la mémoire
    random_state   = 42,
    use_rslora     = False,
    loftq_config   = None,
    target_modules = [      # Couches du Transformer que l'on va fine-tuner
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Affichage du nombre de paramètres entraînables
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA configuré !")
print(f"   Paramètres entraînables : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELLULE 4 — Préparation du Dataset JSONL (Texte + Audio)   ║
# ╚══════════════════════════════════════════════════════════════╝

# AVANT D'EXÉCUTER CETTE CELLULE :
# Importez votre fichier sos_dataset.jsonl dans Colab :
# → Panneau gauche > icône Dossier > glissez votre fichier
# OU montez votre Google Drive (voir cellule optionnelle ci-dessous).

# --- (OPTIONNEL) Si vous utilisez Google Drive ---
# from google.colab import drive
# drive.mount('/content/drive')
# DATASET_PATH = "/content/drive/MyDrive/sos_cameroun/sos_dataset.jsonl"

DATASET_PATH = "sos_dataset.jsonl"  # Chemin si fichier uploadé directement

from datasets import load_dataset

# Template qui structure l'entrée pour que Llama comprenne son rôle
PROMPT_TEMPLATE = """Tu es l'IA de SOS-Cameroun, un système d'urgence basé à Yaoundé, Cameroun. Ton rôle est d'analyser les alertes d'urgence (messages texte ou transcriptions vocales imparfaites) et de fournir une réponse professionnelle, rassurante et structurée.

### Instruction:
{}

### Alerte reçue:
{}

### Réponse SOS-Cameroun:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Token de fin obligatoire

def format_examples(examples):
    """Formate chaque exemple du dataset selon le template."""
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    ):
        text = PROMPT_TEMPLATE.format(instruction, inp, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Chargement et transformation du dataset
print(f"⏳ Chargement du dataset depuis : {DATASET_PATH}")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(format_examples, batched=True)

print(f"✅ Dataset chargé et formaté !")
print(f"   Nombre d'exemples : {len(dataset)}")
print(f"\n📋 Aperçu du premier exemple :")
print(dataset[0]["text"][:500] + "...")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELLULE 5 — Lancement de l'Entraînement (Fine-Tuning)      ║
# ╚══════════════════════════════════════════════════════════════╝

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = dataset,
    dataset_text_field = "text",
    max_seq_length     = max_seq_length,
    dataset_num_proc   = 2,
    packing            = False,
    args = TrainingArguments(
        # --- Paramètres d'entraînement ---
        per_device_train_batch_size  = 2,    # Exemples traités en parallèle
        gradient_accumulation_steps  = 4,    # Accumule pour simuler un batch de 8
        warmup_steps                 = 10,   # Étapes de montée en puissance
        num_train_epochs             = 3,    # 3 passages sur tout le dataset
        # max_steps = 60,                    # ← Décommentez pour un test rapide
        learning_rate                = 2e-4, # Vitesse d'apprentissage
        weight_decay                 = 0.01,
        lr_scheduler_type            = "cosine",  # Décroissance progressive
        fp16                         = not is_bfloat16_supported(),
        bf16                         = is_bfloat16_supported(),
        logging_steps                = 5,    # Affiche la perte toutes les 5 étapes
        optim                        = "adamw_8bit",
        seed                         = 42,
        output_dir                   = "outputs",
        report_to                    = "none",  # Désactive WandB
    ),
)

print("🚀 Démarrage du fine-tuning SOS-Cameroun...")
print("   (Durée estimée : 10-20 min sur GPU T4 avec 150 exemples, 3 epochs)")
print("-" * 60)

trainer_stats = trainer.train()

print("-" * 60)
print(f"✅ Fine-tuning terminé !")
print(f"   Durée totale    : {trainer_stats.metrics['train_runtime']:.0f} secondes")
print(f"   Perte finale    : {trainer_stats.metrics['train_loss']:.4f}")
print(f"   Étapes totales  : {trainer_stats.metrics['train_steps_per_second']:.2f} étapes/sec")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELLULE 6 — Test du Modèle Texte (Post Fine-Tuning)        ║
# ╚══════════════════════════════════════════════════════════════╝

FastLanguageModel.for_inference(model)  # Mode inférence rapide

def generer_reponse_sos(alerte: str, instruction: str = "Analyse cette alerte d'urgence et fournis la marche à suivre.") -> str:
    """Génère une réponse SOS-Cameroun à partir d'une alerte textuelle."""
    prompt = PROMPT_TEMPLATE.format(instruction, alerte, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens = 256,
        temperature    = 0.7,
        top_p          = 0.9,
        use_cache      = True,
    )
    # On ne garde que la partie "Réponse" générée
    full_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    reponse   = full_text.split("### Réponse SOS-Cameroun:")[-1].strip()
    return reponse

# --- Tests avec des alertes TEXTE ---
alertes_test_texte = [
    "au secour ma maison brule a essos pardon venez vite l feu est gran on a peur",
    "ya 1 gro accidan a mvan jsute avan le peage un bus a cogner une moto ya le san partou vené vit svp",
    "braquage a bastos pres de lambassade il ont pris mon sac et mon telephone aidez moi",
]

print("=" * 60)
print("🔍 TESTS DU MODÈLE — ALERTES TEXTE")
print("=" * 60)
for i, alerte in enumerate(alertes_test_texte, 1):
    print(f"\n📨 Alerte #{i} : {alerte}")
    print(f"🤖 Réponse SOS-Cameroun :\n{generer_reponse_sos(alerte)}")
    print("-" * 60)


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELLULE 7 — Pipeline Audio (Whisper STT → Llama)           ║
# ╚══════════════════════════════════════════════════════════════╝

import whisper
import os

# Chargement du modèle Whisper (medium = bon équilibre vitesse/précision)
print("⏳ Chargement de Whisper 'medium'...")
whisper_model = whisper.load_model("medium")
print("✅ Whisper chargé !")

def transcrire_note_vocale(chemin_audio: str, langue: str = "fr") -> str:
    """
    Transcrit une note vocale en texte avec Whisper.
    Retourne la transcription brute (avec hésitations potentielles).
    """
    print(f"🎙️ Transcription de : {chemin_audio}")
    result = whisper_model.transcribe(
        chemin_audio,
        language        = langue,
        task            = "transcribe",
        verbose         = False,
        no_speech_threshold   = 0.5,
        logprob_threshold     = -1.0,
        compression_ratio_threshold = 2.4,
    )
    return result["text"].strip()


def pipeline_audio_sos(chemin_audio: str) -> dict:
    """
    Pipeline complet : Audio → Transcription Whisper → Réponse Llama.
    
    Args:
        chemin_audio: Chemin vers le fichier audio (.mp3, .wav, .ogg, .m4a)
    
    Returns:
        dict avec 'transcription' et 'reponse_sos'
    """
    # Étape 1 : Transcription Audio → Texte
    transcription = transcrire_note_vocale(chemin_audio)
    print(f"📝 Transcription brute :\n   '{transcription}'")

    # Étape 2 : Analyse par le modèle fine-tuné
    instruction = "Analyse cette transcription d'alerte vocale et indique la marche à suivre."
    reponse = generer_reponse_sos(transcription, instruction)

    return {
        "transcription"  : transcription,
        "reponse_sos"    : reponse,
    }


# --- Test Audio (si vous avez un fichier) ---
# INSTRUCTIONS :
# 1. Importez un fichier audio dans Colab (panneau gauche > dossier)
# 2. Remplacez "alerte_test.ogg" par le vrai nom de votre fichier

AUDIO_TEST = "alerte_test.ogg"

if os.path.exists(AUDIO_TEST):
    print("\n" + "=" * 60)
    print("🔍 TEST DU PIPELINE AUDIO")
    print("=" * 60)
    resultat = pipeline_audio_sos(AUDIO_TEST)
    print(f"\n🤖 Réponse SOS-Cameroun :\n{resultat['reponse_sos']}")
else:
    print(f"\n⚠️ Fichier audio '{AUDIO_TEST}' non trouvé.")
    print("   Importez un fichier .ogg/.mp3/.wav dans Colab pour tester.")
    print("   Simulation avec transcription manuelle...")

    # Simulation d'une transcription typique de note vocale paniquée
    transcription_simulee = "euh allô oui pardon euh il y a un gros accident ici à ème van là juste avant le péage euh une voiture a cogné le bain skin pardon venez il y a le sang partout"
    instruction = "Analyse cette transcription d'alerte vocale et indique la marche à suivre."
    print(f"\n📝 Transcription simulée :\n   '{transcription_simulee}'")
    print(f"\n🤖 Réponse SOS-Cameroun :\n{generer_reponse_sos(transcription_simulee, instruction)}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELLULE 8 — Sauvegarde du Modèle Fine-Tuné                 ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Option A : Sauvegarder localement dans Colab (puis télécharger) ---
print("💾 Sauvegarde des adaptateurs LoRA localement...")
model.save_pretrained("sos_cameroun_lora")
tokenizer.save_pretrained("sos_cameroun_lora")
print("✅ Sauvegardé dans le dossier 'sos_cameroun_lora/'")
print("   → Vous pouvez télécharger ce dossier depuis le panneau fichiers de Colab.")

# --- Option B : Pousser directement sur Hugging Face Hub ---
# Décommentez les lignes suivantes pour l'utiliser :
# model.push_to_hub("votre_username_hf/sos-cameroun-llama3", token=HF_TOKEN)
# tokenizer.push_to_hub("votre_username_hf/sos-cameroun-llama3", token=HF_TOKEN)
# print("✅ Modèle publié sur Hugging Face Hub !")

# --- Option C : Sauvegarder sur Google Drive ---
# Décommentez si vous avez monté votre Drive dans la cellule 4 :
# from google.colab import drive
# drive.mount('/content/drive')
# SAVE_PATH = "/content/drive/MyDrive/sos_cameroun/lora_model"
# model.save_pretrained(SAVE_PATH)
# tokenizer.save_pretrained(SAVE_PATH)
# print(f"✅ Modèle sauvegardé sur Google Drive : {SAVE_PATH}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELLULE 9 — Courbes de Perte (Pour le Jury)                ║
# ╚══════════════════════════════════════════════════════════════╝

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Récupération des logs d'entraînement
log_history = trainer.state.log_history
steps  = [x["step"] for x in log_history if "loss" in x]
losses = [x["loss"] for x in log_history if "loss" in x]

# Tracé de la courbe
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, losses, color="#E63946", linewidth=2, marker="o", markersize=4, label="Perte d'entraînement")
ax.fill_between(steps, losses, alpha=0.1, color="#E63946")
ax.set_title("Courbe de Perte — Fine-Tuning SOS-Cameroun (Llama 3 8B + LoRA)", fontsize=14, fontweight="bold")
ax.set_xlabel("Étapes d'entraînement (Steps)")
ax.set_ylabel("Perte (Loss)")
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
ax.legend()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("courbe_perte_sos_cameroun.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Courbe de perte sauvegardée → 'courbe_perte_sos_cameroun.png'")
print("   Téléchargez cette image pour votre présentation jury !")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELLULE 10 — Résumé Final & Checklist                      ║
# ╚══════════════════════════════════════════════════════════════╝

print("""
╔═══════════════════════════════════════════════════════════════╗
║         RÉSUMÉ — Fine-Tuning SOS-Cameroun Terminé !          ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  ✅ Modèle de base    : Llama 3 8B (4-bit quantifié)          ║
║  ✅ Méthode           : LoRA (r=16, alpha=16)                  ║
║  ✅ Données           : Dataset JSONL Yaoundé (texte + audio) ║
║  ✅ Pipeline Audio    : Whisper STT → Llama 3                  ║
║  ✅ Adaptateurs       : Sauvegardés dans 'sos_cameroun_lora/' ║
║  ✅ Courbe de perte   : courbe_perte_sos_cameroun.png          ║
║                                                               ║
║  PROCHAINES ÉTAPES :                                          ║
║  1. Télécharger 'sos_cameroun_lora/' → data/fine-tuned-models ║
║  2. Intégrer dans llm_service.py (FastAPI backend)            ║
║  3. Préparer la démo jury avec une note vocale réelle         ║
╚═══════════════════════════════════════════════════════════════╝
""")
