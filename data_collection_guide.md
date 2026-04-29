# Guide de Collecte de Données : SOS-Cameroun

Pour le concours ACIAI, la qualité et l'originalité de vos données ("La précision sur les données") sont des critères majeurs. Voici comment construire votre dataset local.

---

## 1. Données Textuelles (Signalements d'Urgences)
L'objectif est d'avoir des milliers d'exemples de signalements avec des variantes linguistiques (Français, Anglais, Camfranglais).

### A. Génération Synthétique (Rapide)
Utilisez un script pour demander à un LLM (comme GPT-4 ou Llama 3) de générer des variantes.
*   **Prompt** : "Génère 100 phrases de signalement d'accidents de la route à Yaoundé en utilisant un langage familier, des fautes de frappe et des noms de quartiers comme Mokolo, Etoudi, etc."
*   **Résultat** : Vous obtenez rapidement un gros volume de données pour tester la robustesse de votre modèle.

### B. Scraping de Réseaux Sociaux
*   **Sources** : Groupes Facebook locaux, Twitter (X) avec des hashtags comme #Cameroun, #Yaounde, #Urgence.
*   **Outil** : Python (`snscrape` ou `BeautifulSoup`).
*   **Éthique** : Anonymisez toujours les données collectées.

---

## 2. Données Vocales (Accents et Dialectes)
C'est ici que vous marquerez le plus de points sur la "Viabilité Locale".

### A. Crowdsourcing Étudiant
*   **Action** : Demandez à 20-30 camarades de différentes régions (Nord, Ouest, Littoral) d'enregistrer 10 phrases d'urgence sur leur téléphone.
*   **Format** : Fichiers `.wav` ou `.mp3`.
*   **Diversité** : Assurez-vous d'avoir des voix d'hommes, de femmes, et des accents variés.

### B. Enregistrement en Environnement Bruyant
*   Enregistrez quelques clips près d'un marché ou d'une route passante. Cela permettra de tester si Whisper (STT) arrive à extraire la voix malgré le bruit ambiant (critère de Robustesse).

---

## 3. Données Géographiques (Points de Repère)
Le jury appréciera que l'IA connaisse les "Hotspots" locaux.
*   **Action** : Créez un fichier JSON contenant les coordonnées GPS et les noms populaires des carrefours, hôpitaux et commissariats de chaque grande ville.
*   **Source** : OpenStreetMap (OSM) ou Google Maps.

---

## 4. Outils de Labellisation
Une fois les données collectées, vous devez les "étiqueter" (dire à l'IA ce qu'elles signifient).
*   **Label Studio** (Open Source) : Excellent pour annoter du texte (entités nommées) et de l'audio.
*   **Format cible** : JSONL (JSON Lines).
    ```json
    {"text": "Il y a trop de feu à Mokolo", "label": {"incident": "INCENDIE", "lieu": "Mokolo", "gravite": "Haute"}}
    ```

---

## 5. Ce qu'il faut mettre dans votre Rapport ACIAI
Dans votre Note Conceptuelle, dédiez une section à la **"Méthodologie de Données"** :
1.  **Volume** : "Nous avons collecté X millier(s) de signalements."
2.  **Diversité** : "Dataset incluant les 10 régions du Cameroun."
3.  **Nettoyage** : "Processus d'anonymisation et de correction IA."

**Voulez-vous que je crée un petit script Python pour vous aider à générer ces données synthétiques dès maintenant ?**
