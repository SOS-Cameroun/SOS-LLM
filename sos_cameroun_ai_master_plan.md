# Master Plan : De la Collecte au Fine-Tuning (A à Z)

Ce guide récapitule l'intégralité du processus pour créer l'intelligence de **SOS-Cameroun**.

---

## Phase 1 : Collecte et Préparation des Données (La "Nourriture")
Sans bonnes données, l'IA ne sera pas performante.

1.  **Collecte brute Multimodale (Texte & Audio)** : Rassemblez 500 à 1000 exemples. Pour gérer les **notes vocales**, nous n'allons pas entraîner Llama directement sur l'audio, mais sur la **transcription textuelle** (Speech-to-Text) de ces audios.
    *   *Exemple Texte* : "Il y a un accident à l'entrée de Biyem-Assi, envoyez les secours."
    *   *Exemple Transcription Audio* : "euh allô pardon il y a un gros accident euh ici à mvan juste avant le péage il y a le sang partout venez vite" (sans ponctuation, avec hésitations).
2.  **Formatage JSONL** : Transformez ces données en un fichier structuré que l'IA peut lire.
    *   Fichier : `data/training/text/sos_dataset.jsonl`
    *   Structure : `{"instruction": "...", "input": "...", "output": "..."}`
3.  **Bruitage Intentionnel** : Simulez les erreurs de frappe (pour les SMS) et les erreurs phonétiques de transcription (pour les notes vocales) dans le `input` pour rendre l'IA robuste.

---

## Phase 2 : Configuration du Laboratoire (Google Colab)
C'est ici que l'IA va "apprendre".

1.  **Hugging Face** : Créez un compte et demandez l'accès à **Llama-3-8B** (Meta). Récupérez votre **Token API**.
2.  **Google Colab** : Ouvrez un notebook, allez dans `Exécution` > `Modifier le type d'exécution` > Sélectionnez **GPU T4**.
3.  **Installation d'Unsloth** : Exécutez les commandes d'installation (voir tutoriel précédent). Unsloth permet d'entraîner Llama sur le GPU gratuit de Colab.

---

## Phase 3 : Le Fine-Tuning (L'Apprentissage)
Le code va exécuter ces étapes automatiquement :

1.  **Chargement** : Le script télécharge Llama-3-8B depuis Hugging Face.
2.  **Configuration LoRA** : On définit les "paramètres d'apprentissage" (Rang, Alpha). C'est comme régler la sensibilité d'un cerveau.
3.  **Entraînement** : Le modèle lit vos 1000 exemples plusieurs fois. Il apprend à prédire la réponse parfaite pour chaque urgence camerounaise.
    *   *Durée* : Entre 10 et 20 minutes sur Colab.

---

## Phase 4 : Évaluation et Sauvegarde (Le Diplôme)
Une fois l'entraînement fini, vous devez vérifier la qualité.

1.  **Test d'Inférence** : Posez une question au modèle sur Colab (ex: "J'ai un blessé grave à Mokolo, que faire ?"). S'il répond avec vos conseils et connaît Mokolo, c'est réussi !
2.  **Exportation** : Sauvegardez le modèle (les "Adapters") sur votre Google Drive ou sur Hugging Face.
3.  **Téléchargement** : Récupérez les fichiers dans votre dossier `data/fine-tuned-models/` sur votre machine locale.

---

## Phase 5 : Intégration Finale (Le Déploiement Pipeline Audio/Texte)
Votre backend Python (FastAPI) va maintenant utiliser ce nouveau cerveau.

1.  **Pipeline Multi-Modale** : 
    *   *Si Alerte Texte* : Envoyer le texte directement au modèle Llama fine-tuné.
    *   *Si Alerte Vocale* : Le backend utilise un modèle **Speech-to-Text (ex: Whisper d'OpenAI)** pour transcrire l'audio en texte. Ce texte (même imparfait) est ensuite envoyé à votre modèle Llama.
2.  **Chargement local** : On configure `llm_service.py` pour charger vos poids personnalisés au lieu d'appeler Groq.
3.  **Validation Jury** : Préparez vos captures d'écran de l'entraînement et une démonstration en direct d'une note vocale envoyée via l'application, transcrite puis analysée.

---

## Tableau de Bord du Processus

| Étape | Action | Outil | Résultat attendu |
| :--- | :--- | :--- | :--- |
| 1 | Collecte | Humain / Script | Fichier JSONL |
| 2 | Setup | Colab / HF | GPU prêt |
| 3 | Training | Unsloth | Modèle spécialisé |
| 4 | Export | GDrive / HF | Fichiers `.bin` ou `.safetensors` |
| 5 | Demo | FastAPI | Réponse IA parfaite |

**Voulez-vous que nous passions à la pratique en générant ensemble les 20 premières lignes de votre dataset JSONL pour être sûr que le format est parfait ?**
