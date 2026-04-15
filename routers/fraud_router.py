"""
Router Fraude — Consultation des logs anti-fraude
===================================================
Endpoints pour consulter, filtrer et résoudre les alertes suspectes
détectées par le pipeline multimodal.

Utilisé par le dashboard administrateur du CINU pour :
- Lister les alertes SUSPECTES et FRAUDE
- Consulter les détails d'un log (texte, image, raison)
- Marquer un log comme résolu après investigation
- Obtenir des statistiques globales
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import logging

from services.fraud_db import fraud_db
from models.schemas import FraudLogEntry

router = APIRouter()
logger = logging.getLogger("ai-inference.fraud-router")


@router.get(
    "/logs",
    response_model=List[FraudLogEntry],
    summary="📋 Lister les logs de fraude",
    description="""
Récupère les logs de fraude enregistrés par le pipeline anti-fraude.

Filtrable par label :
- **FIABLE** : Alertes confirmées comme légitimes
- **SUSPECTE** : Alertes avec une incohérence partielle texte-image
- **FRAUDE** : Alertes avec une incohérence majeure (probable fausse alerte)

Paginé avec ``limit`` et ``offset``.
    """,
    response_description="Liste des entrées du journal de fraude",
)
async def list_fraud_logs(
    label: Optional[str] = Query(
        None,
        description="Filtrer par verdict : FIABLE, SUSPECTE ou FRAUDE",
        example="SUSPECTE",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="Nombre max de résultats à retourner",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Décalage pour la pagination",
    ),
):
    """Liste les logs de fraude avec filtrage optionnel."""
    try:
        logs = await fraud_db.get_all_logs(label=label, limit=limit, offset=offset)
        return logs
    except Exception as e:
        logger.error(f"❌ Erreur lecture logs fraude : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/logs/{log_id}",
    response_model=FraudLogEntry,
    summary="🔍 Détail d'un log de fraude",
    description="Récupère les détails complets d'une entrée de fraude par son identifiant UUID.",
    response_description="Détail complet du log de fraude",
)
async def get_fraud_log(log_id: str):
    """Récupère un log de fraude par son ID."""
    log = await fraud_db.get_log_by_id(log_id)
    if not log:
        raise HTTPException(status_code=404, detail=f"Log de fraude '{log_id}' non trouvé")
    return log


@router.patch(
    "/logs/{log_id}/resolve",
    summary="✅ Résoudre un log de fraude",
    description="""
Marque un log de fraude comme **résolu** après investigation par un agent du CINU.

Cela signifie qu'un humain a vérifié l'alerte et a pris une décision :
- Fausse alerte confirmée → sanctionner l'émetteur
- Alerte légitime malgré le score → traiter normalement
    """,
    response_description="Confirmation de la résolution",
)
async def resolve_fraud_log(log_id: str):
    """Marque un log de fraude comme résolu."""
    updated = await fraud_db.mark_resolved(log_id)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Log de fraude '{log_id}' non trouvé")
    return {"status": "resolved", "id": log_id}


@router.get(
    "/stats",
    summary="📊 Statistiques de fraude",
    description="Retourne le nombre de logs par label (FIABLE, SUSPECTE, FRAUDE) pour le dashboard.",
    response_description="Compteurs par label",
)
async def fraud_stats():
    """Retourne les statistiques globales de fraude."""
    try:
        counts = await fraud_db.count_by_label()
        return {
            "total": sum(counts.values()),
            "par_label": counts,
        }
    except Exception as e:
        logger.error(f"❌ Erreur stats fraude : {e}")
        raise HTTPException(status_code=500, detail=str(e))
