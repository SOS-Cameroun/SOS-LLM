"""
Service Base de Données de Fraude — SQLite
=============================================
Gère les logs d'alertes suspectes / frauduleuses.
Utilise aiosqlite pour les opérations asynchrones.

Schéma :
    fraud_logs — Journal de toutes les alertes analysées avec un score < seuil
    Stocke : texte, hash d'image, score de fiabilité, lieu déclaré vs. détecté, etc.
"""

import os
import uuid
import logging
import aiosqlite
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from utils.config import settings

logger = logging.getLogger("ai-inference.fraud-db")

# Schéma SQL de la table de fraude
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fraud_logs (
    id                TEXT PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    alert_text        TEXT,
    image_hash        TEXT,
    image_path        TEXT,
    description_image TEXT,
    score_fiabilite   INTEGER NOT NULL,
    label             TEXT NOT NULL,
    raison            TEXT NOT NULL,
    lieu_declare      TEXT,
    lieu_detecte      TEXT,
    ip_source         TEXT,
    resolved          INTEGER DEFAULT 0
);
"""

# Index pour les recherches fréquentes
_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_fraud_label ON fraud_logs(label);",
    "CREATE INDEX IF NOT EXISTS idx_fraud_timestamp ON fraud_logs(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_fraud_score ON fraud_logs(score_fiabilite);",
]


class FraudDB:
    """
    Gestionnaire de la base de données de fraude.

    Usage asynchrone :
        db = FraudDB()
        await db.init()
        await db.log_fraud(...)
        logs = await db.get_all_logs()
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.FRAUD_DB_PATH
        self._ensure_directory()

    def _ensure_directory(self):
        """Crée le répertoire parent si nécessaire."""
        dir_path = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(dir_path, exist_ok=True)

    async def init(self):
        """
        Initialise la base de données : crée la table et les index si nécessaire.
        Doit être appelé au démarrage de l'application (lifespan FastAPI).
        """
        logger.info(f"🗄️  Initialisation de la base de fraude : {self.db_path}")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(_CREATE_TABLE_SQL)
            for idx_sql in _CREATE_INDEXES_SQL:
                await db.execute(idx_sql)
            await db.commit()
        logger.info("✅ Base de fraude initialisée.")

    async def log_fraud(
        self,
        score_fiabilite: int,
        label: str,
        raison: str,
        alert_text: Optional[str] = None,
        image_hash: Optional[str] = None,
        image_path: Optional[str] = None,
        description_image: Optional[str] = None,
        lieu_declare: Optional[str] = None,
        lieu_detecte: Optional[str] = None,
        ip_source: Optional[str] = None,
    ) -> str:
        """
        Enregistre une nouvelle entrée dans le journal de fraude.

        Args:
            score_fiabilite: Score de fiabilité (0–100)
            label: Verdict (FIABLE, SUSPECTE, FRAUDE)
            raison: Explication textuelle
            alert_text: Texte brut de l'alerte
            image_hash: Hash perceptuel (pHash) de l'image
            image_path: Chemin du fichier image archivé
            description_image: Description IA de l'image (Groq Vision)
            lieu_declare: Lieu mentionné par l'utilisateur
            lieu_detecte: Lieu identifié automatiquement
            ip_source: Adresse IP de la source

        Returns:
            L'identifiant UUID du log créé.
        """
        log_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO fraud_logs
                    (id, timestamp, alert_text, image_hash, image_path,
                     description_image, score_fiabilite, label, raison,
                     lieu_declare, lieu_detecte, ip_source, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    log_id, timestamp, alert_text, image_hash, image_path,
                    description_image, score_fiabilite, label, raison,
                    lieu_declare, lieu_detecte, ip_source,
                ),
            )
            await db.commit()

        logger.info(f"📝 Fraude loggée : id={log_id}, label={label}, score={score_fiabilite}")
        return log_id

    async def get_all_logs(
        self,
        label: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Récupère les logs de fraude, filtrés optionnellement par label.

        Args:
            label: Filtrer par verdict (FIABLE, SUSPECTE, FRAUDE). None = tous.
            limit: Nombre max de résultats.
            offset: Décalage pour pagination.

        Returns:
            Liste de dictionnaires représentant les logs.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if label:
                cursor = await db.execute(
                    "SELECT * FROM fraud_logs WHERE label = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (label, limit, offset),
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM fraud_logs ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_log_by_id(self, log_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère un log de fraude par son identifiant.

        Args:
            log_id: UUID du log.

        Returns:
            Dictionnaire du log ou None si introuvable.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM fraud_logs WHERE id = ?", (log_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def mark_resolved(self, log_id: str) -> bool:
        """
        Marque un log de fraude comme résolu/traité.

        Args:
            log_id: UUID du log à résoudre.

        Returns:
            True si le log a été mis à jour, False si introuvable.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE fraud_logs SET resolved = 1 WHERE id = ?", (log_id,)
            )
            await db.commit()
            updated = cursor.rowcount > 0

        if updated:
            logger.info(f"✅ Log de fraude {log_id} marqué comme résolu.")
        else:
            logger.warning(f"⚠️ Log de fraude {log_id} introuvable pour résolution.")
        return updated

    async def count_by_label(self) -> Dict[str, int]:
        """
        Compte le nombre de logs par label (pour statistiques dashboard).

        Returns:
            Dict {"FIABLE": n, "SUSPECTE": n, "FRAUDE": n}
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT label, COUNT(*) as cnt FROM fraud_logs GROUP BY label"
            )
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}


# ── Singleton ────────────────────────────────────────────────────────────────
fraud_db = FraudDB()
