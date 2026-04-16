"""
Service Géolocalisation — Topographie de Yaoundé
==================================================
Base de connaissances locale des quartiers, carrefours et repères
de Yaoundé pour la validation des alertes d'urgence.

Le système connaît parfaitement la topographie de la ville pour :
- Valider qu'un lieu mentionné existe réellement
- Compléter (halluciner intelligemment) un message fragmenté en situation de danger
- Fournir du contexte au LLM pour des réponses géolocalisées
"""

import logging
import math
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger("ai-inference.geo")


# ═══════════════════════════════════════════════════════════════════════════════
# Base de données topographique de Yaoundé
# ═══════════════════════════════════════════════════════════════════════════════

# Format : nom_normalisé -> { display: str, lat: float, lon: float, zone: str, description: str }
YAOUNDE_LANDMARKS: Dict[str, Dict] = {
    # ── Centre-ville & Administration ───────────────────────────────────────
    "poste centrale": {
        "display": "Poste Centrale",
        "lat": 3.8667, "lon": 11.5167,
        "zone": "Centre-ville",
        "description": "Centre névralgique, Poste Centrale du Cameroun, zone très fréquentée"
    },
    "carrefour warda": {
        "display": "Carrefour Warda",
        "lat": 3.8680, "lon": 11.5180,
        "zone": "Centre-ville",
        "description": "Carrefour commercial près de la Poste Centrale"
    },
    "avenue kennedy": {
        "display": "Avenue Kennedy",
        "lat": 3.8650, "lon": 11.5200,
        "zone": "Centre-ville",
        "description": "Artère principale du centre-ville administratif"
    },

    # ── Quartiers résidentiels ──────────────────────────────────────────────
    "bastos": {
        "display": "Bastos",
        "lat": 3.8800, "lon": 11.5050,
        "zone": "Nord",
        "description": "Quartier diplomatique et résidentiel haut standing, ambassades"
    },
    "etoudi": {
        "display": "Etoudi",
        "lat": 3.8900, "lon": 11.5100,
        "zone": "Nord",
        "description": "Quartier du Palais présidentiel, zone hautement sécurisée"
    },
    "nsimeyong": {
        "display": "Nsimeyong",
        "lat": 3.8400, "lon": 11.4950,
        "zone": "Sud-Ouest",
        "description": "Quartier résidentiel populaire, zone escarpée"
    },
    "biyem-assi": {
        "display": "Biyem-Assi",
        "lat": 3.8450, "lon": 11.4830,
        "zone": "Ouest",
        "description": "Grand quartier populaire dense, accès difficile en saison des pluies"
    },
    "melen": {
        "display": "Melen",
        "lat": 3.8600, "lon": 11.4950,
        "zone": "Ouest",
        "description": "Quartier universitaire dense, proximité Université Yaoundé I"
    },
    "obili": {
        "display": "Obili",
        "lat": 3.8580, "lon": 11.4920,
        "zone": "Ouest",
        "description": "Quartier étudiant très animé, marchés, restauration"
    },
    "ngousso": {
        "display": "Ngousso",
        "lat": 3.8750, "lon": 11.5250,
        "zone": "Est",
        "description": "Quartier en pleine expansion, nouvelles constructions"
    },
    "emana": {
        "display": "Emana",
        "lat": 3.8950, "lon": 11.5300,
        "zone": "Nord-Est",
        "description": "Quartier périphérique en développement"
    },
    "nkolbisson": {
        "display": "Nkolbisson",
        "lat": 3.8650, "lon": 11.4650,
        "zone": "Ouest",
        "description": "Zone péri-urbaine, centres de recherche (IRAD)"
    },
    "essos": {
        "display": "Essos",
        "lat": 3.8700, "lon": 11.5350,
        "zone": "Est",
        "description": "Quartier commercial et résidentiel, axe vers Douala"
    },
    "mvog-mbi": {
        "display": "Mvog-Mbi",
        "lat": 3.8550, "lon": 11.5200,
        "zone": "Centre-Sud",
        "description": "Grand marché populaire Mvog-Mbi, forte densité"
    },
    "mvog-ada": {
        "display": "Mvog-Ada",
        "lat": 3.8520, "lon": 11.5250,
        "zone": "Centre-Sud",
        "description": "Quartier résidentiel populaire proche du centre"
    },
    "nkoldongo": {
        "display": "Nkoldongo",
        "lat": 3.8600, "lon": 11.5400,
        "zone": "Est",
        "description": "Quartier populaire, marché Nkoldongo"
    },
    "etoa-meki": {
        "display": "Etoa-Meki",
        "lat": 3.8720, "lon": 11.5100,
        "zone": "Centre-Nord",
        "description": "Proche du centre, zone mixte résidentielle et commerciale"
    },
    "nlongkak": {
        "display": "Nlongkak",
        "lat": 3.8780, "lon": 11.5130,
        "zone": "Nord",
        "description": "Quartier résidentiel calme, proximité hôpitaux"
    },
    "omnisport": {
        "display": "Omnisport",
        "lat": 3.8850, "lon": 11.5450,
        "zone": "Est",
        "description": "Stade Omnisport, zone sportive et résidentielle"
    },
    "mvan": {
        "display": "Mvan",
        "lat": 3.8300, "lon": 11.5100,
        "zone": "Sud",
        "description": "Sortie sud de Yaoundé, gare routière inter-urbaine"
    },
    "mendong": {
        "display": "Mendong",
        "lat": 3.8380, "lon": 11.4780,
        "zone": "Sud-Ouest",
        "description": "Quartier résidentiel en expansion, collines"
    },
    "simbock": {
        "display": "Simbock",
        "lat": 3.8330, "lon": 11.4700,
        "zone": "Sud-Ouest",
        "description": "Zone péri-urbaine, université catholique (UCAC)"
    },
    "ahala": {
        "display": "Ahala",
        "lat": 3.8200, "lon": 11.5050,
        "zone": "Sud",
        "description": "Quartier périphérique sud, zone résidentielle"
    },
    "olembe": {
        "display": "Olembé",
        "lat": 3.9050, "lon": 11.5300,
        "zone": "Nord",
        "description": "Stade d'Olembé (Coupe d'Afrique 2022), zone en développement"
    },

    # ── Carrefours majeurs ──────────────────────────────────────────────────
    "carrefour jouvence": {
        "display": "Carrefour Jouvence",
        "lat": 3.8700, "lon": 11.5050,
        "zone": "Centre",
        "description": "Carrefour stratégique très fréquenté, axe Est-Ouest"
    },
    "rond-point nlongkak": {
        "display": "Rond-Point Nlongkak",
        "lat": 3.8760, "lon": 11.5120,
        "zone": "Nord",
        "description": "Rond-point principal vers Bastos et le centre"
    },
    "carrefour bastos": {
        "display": "Carrefour Bastos",
        "lat": 3.8810, "lon": 11.5060,
        "zone": "Nord",
        "description": "Entrée du quartier diplomatique"
    },
    "carrefour biyem-assi": {
        "display": "Carrefour Biyem-Assi",
        "lat": 3.8460, "lon": 11.4850,
        "zone": "Ouest",
        "description": "Grand carrefour du quartier Biyem-Assi"
    },
    "carrefour obili": {
        "display": "Carrefour Obili",
        "lat": 3.8590, "lon": 11.4930,
        "zone": "Ouest",
        "description": "Carrefour étudiant très animé"
    },

    # ── Marchés ─────────────────────────────────────────────────────────────
    "marche mokolo": {
        "display": "Marché Mokolo",
        "lat": 3.8650, "lon": 11.5000,
        "zone": "Centre-Ouest",
        "description": "Plus grand marché de Yaoundé, très dense, risque incendie élevé"
    },
    "mokolo": {
        "display": "Mokolo",
        "lat": 3.8650, "lon": 11.5000,
        "zone": "Centre-Ouest",
        "description": "Quartier du marché Mokolo, très dense et populaire"
    },
    "marche central": {
        "display": "Marché Central",
        "lat": 3.8660, "lon": 11.5175,
        "zone": "Centre-ville",
        "description": "Marché central historique de Yaoundé"
    },
    "marche mvog-mbi": {
        "display": "Marché Mvog-Mbi",
        "lat": 3.8540, "lon": 11.5210,
        "zone": "Centre-Sud",
        "description": "Grand marché populaire, forte densité"
    },
    "marche essos": {
        "display": "Marché Essos",
        "lat": 3.8710, "lon": 11.5360,
        "zone": "Est",
        "description": "Marché du quartier Essos"
    },
    "marche nkoldongo": {
        "display": "Marché Nkoldongo",
        "lat": 3.8610, "lon": 11.5410,
        "zone": "Est",
        "description": "Marché du quartier Nkoldongo, produits vivriers"
    },

    # ── Points d'intérêt clés ───────────────────────────────────────────────
    "kondengui": {
        "display": "Kondengui",
        "lat": 3.8550, "lon": 11.5350,
        "zone": "Est",
        "description": "Prison centrale de Kondengui, zone sensible"
    },
    "hopital central": {
        "display": "Hôpital Central de Yaoundé",
        "lat": 3.8700, "lon": 11.5160,
        "zone": "Centre",
        "description": "Hôpital Central, plus grand hôpital public de la ville"
    },
    "chuy": {
        "display": "CHU Yaoundé",
        "lat": 3.8700, "lon": 11.5160,
        "zone": "Centre",
        "description": "Centre Hospitalier Universitaire de Yaoundé"
    },
    "hopital general": {
        "display": "Hôpital Général de Yaoundé",
        "lat": 3.8600, "lon": 11.5080,
        "zone": "Centre",
        "description": "Hôpital Général, référence nationale"
    },
    "universite yaounde 1": {
        "display": "Université de Yaoundé I",
        "lat": 3.8600, "lon": 11.4970,
        "zone": "Ouest",
        "description": "Campus principal Ngoa-Ekelle, forte population étudiante"
    },
    "universite yaounde 2": {
        "display": "Université de Yaoundé II - Soa",
        "lat": 3.9300, "lon": 11.5900,
        "zone": "Nord-Est périphérique",
        "description": "Campus de Soa, zone péri-urbaine"
    },
    "palais des congres": {
        "display": "Palais des Congrès",
        "lat": 3.8730, "lon": 11.5180,
        "zone": "Centre",
        "description": "Centre de conférences, événements officiels"
    },
    "gare routiere mvan": {
        "display": "Gare Routière de Mvan",
        "lat": 3.8310, "lon": 11.5110,
        "zone": "Sud",
        "description": "Gare routière principale, départs vers toutes les régions"
    },
}


# Aliases courants (variantes d'écriture, abréviations)
ALIASES: Dict[str, str] = {
    "jouvence": "carrefour jouvence",
    "la poste": "poste centrale",
    "biyemassi": "biyem-assi",
    "biyem assi": "biyem-assi",
    "mvog mbi": "mvog-mbi",
    "mvog ada": "mvog-ada",
    "ngoa ekelle": "universite yaounde 1",
    "ngoa-ekelle": "universite yaounde 1",
    "soa": "universite yaounde 2",
    "prison kondengui": "kondengui",
    "prison centrale": "kondengui",
    "stade olembe": "olembe",
    "stade omnisport": "omnisport",
    "hgy": "hopital general",
    "hcy": "hopital central",
    "marche mokolo": "mokolo",
    "gare mvan": "gare routiere mvan",
}


class GeoService:
    """
    Service de validation topographique pour Yaoundé.

    Permet de :
    - Détecter des noms de lieu dans un texte libre
    - Valider qu'un lieu existe dans la base topographique
    - Calculer le repère le plus proche depuis des coordonnées GPS
    - Fournir du contexte géographique au LLM
    """

    def __init__(self):
        self.landmarks = YAOUNDE_LANDMARKS
        self.aliases = ALIASES
        logger.info(
            f"✅ GeoService initialisé avec {len(self.landmarks)} repères "
            f"et {len(self.aliases)} alias pour Yaoundé."
        )

    def _normalize(self, text: str) -> str:
        """Normalise un texte pour la recherche de lieux (minuscules, accents simplifiés)."""
        import unicodedata
        text = text.lower().strip()
        # Suppression des accents pour la comparaison souple
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def validate_location(self, text: str) -> Dict:
        """
        Cherche et valide les noms de lieux de Yaoundé dans un texte libre.

        Args:
            text: Texte brut (alerte, message vocal transcrit, etc.)

        Returns:
            dict avec :
            - ``lieux_trouves`` : liste des lieux reconnus avec leurs métadonnées
            - ``lieu_principal`` : le premier lieu trouvé (le plus probable)
            - ``valide`` : True si au moins un lieu a été identifié
        """
        text_norm = self._normalize(text)
        found: List[Dict] = []

        # 1) Chercher parmi les alias d'abord
        for alias, canonical in self.aliases.items():
            alias_norm = self._normalize(alias)
            if alias_norm in text_norm and canonical in self.landmarks:
                landmark = self.landmarks[canonical]
                if landmark not in found:
                    found.append(landmark)

        # 2) Chercher les noms de lieux directs
        for key, landmark in self.landmarks.items():
            key_norm = self._normalize(key)
            if key_norm in text_norm and landmark not in found:
                found.append(landmark)

        return {
            "lieux_trouves": found,
            "lieu_principal": found[0] if found else None,
            "valide": len(found) > 0,
        }

    def get_nearest_landmark(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Trouve le repère connu le plus proche des coordonnées GPS données.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            dict du repère le plus proche avec la distance en km, ou None.
        """
        best = None
        best_dist = float("inf")

        for _key, landmark in self.landmarks.items():
            dist = self._haversine(lat, lon, landmark["lat"], landmark["lon"])
            if dist < best_dist:
                best_dist = dist
                best = {**landmark, "distance_km": round(dist, 2)}

        return best

    def enrich_location_context(self, lieu: str) -> str:
        """
        Fournit un paragraphe de contexte topographique pour le LLM
        afin d'enrichir la compréhension de la zone.

        Args:
            lieu: Nom du lieu (tel que détecté dans le texte)

        Returns:
            Contexte textuel enrichi pour le system prompt LLM.
        """
        lieu_norm = self._normalize(lieu)

        # Résoudre l'alias éventuel
        canonical = self.aliases.get(lieu_norm, lieu_norm)
        landmark = self.landmarks.get(canonical)

        if not landmark:
            return f"Lieu '{lieu}' non reconnu dans la topographie de Yaoundé."

        return (
            f"   Lieu identifié : {landmark['display']} — Zone {landmark['zone']}.\n"
            f"   Description : {landmark['description']}.\n"
            f"   Coordonnées : {landmark['lat']:.4f}°N, {landmark['lon']:.4f}°E.\n"
            f"   Ce lieu est dans le périmètre couvert par les services d'urgence de Yaoundé."
        )

    def get_all_landmarks(self) -> List[str]:
        """Retourne la liste de tous les noms de lieux connus."""
        return [v["display"] for v in self.landmarks.values()]

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcule la distance en km entre deux points GPS (formule de Haversine)."""
        R = 6371  # Rayon de la Terre en km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Singleton ────────────────────────────────────────────────────────────────
geo_service = GeoService()
