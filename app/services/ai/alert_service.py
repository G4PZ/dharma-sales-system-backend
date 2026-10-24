"""
Servicio de Gestión y Auditoría de Alertas IA (Alert Service).

Responsabilidades:
1. Consulta y filtrado paginado de alertas detectadas (nivel, estado de revisión, cliente, fechas).
2. Obtención de estadísticas agregadas (pendientes, confirmadas, falsos positivos, distribución por nivel).
3. Registro de retroalimentación humana (Feedback Loop):
   - Transiciones a CONFIRMADA o FALSO_POSITIVO.
   - Auditoría estricta con registro de correo del revisor y timestamp UTC.
   - Preservación invariable del puntaje original de anomalía (puntaje_anomalia).
4. Verificación de estado de disponibilidad de modelos ML activos (Isolation Forest y K-Means).
"""

from datetime import date, datetime, time, timezone
import logging
import os
from typing import Any, Dict, List, Optional
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.config import ML_MODELS_DIR
from app.models.ai import AlertaVenta, ModeloEntrenamiento
from app.models.sale import Venta
from app.schemas.ai import AlertFeedbackUpdate
from app.services.ai.feature_engineering import build_customer_features
from app.services.ai.trainer import (
    ISOLATION_FOREST_MODEL_TYPE,
    KMEANS_MODEL_TYPE,
    MIN_CUSTOMERS_FOR_KMEANS,
    MIN_PURCHASES_FOR_KMEANS_PROFILE,
    MIN_SALES_FOR_ISOLATION_FOREST,
)

logger = logging.getLogger(__name__)


def list_alerts(
    db: Session,
    nivel: Optional[str] = None,
    estado_revision: Optional[str] = None,
    cliente_id: Optional[int] = None,
    fecha_desde: Optional[date] = None,
    fecha_hasta: Optional[date] = None,
    skip: int = 0,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Retorna el listado paginado de alertas de venta aplicando filtros multicriterio.
    Ordena de forma descendente por fecha de creación (alertas más recientes primero).
    Precarga las relaciones venta y cliente con joinedload para evitar N+1 queries.
    """
    query = db.query(AlertaVenta).options(
        joinedload(AlertaVenta.venta),
        joinedload(AlertaVenta.cliente),
    )

    if nivel:
        query = query.filter(AlertaVenta.nivel_alerta == nivel.upper())

    if estado_revision:
        query = query.filter(AlertaVenta.estado_revision == estado_revision.upper())

    if cliente_id:
        query = query.filter(AlertaVenta.cliente_id == cliente_id)

    if fecha_desde:
        dt_desde = datetime.combine(fecha_desde, time.min).replace(tzinfo=timezone.utc)
        query = query.filter(AlertaVenta.created_at >= dt_desde)

    if fecha_hasta:
        dt_hasta = datetime.combine(fecha_hasta, time.max).replace(tzinfo=timezone.utc)
        query = query.filter(AlertaVenta.created_at <= dt_hasta)

    total = query.count()
    items = query.order_by(AlertaVenta.created_at.desc(), AlertaVenta.id.desc()).offset(skip).limit(limit).all()

    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


def get_alert_by_id(db: Session, alerta_id: int) -> Optional[AlertaVenta]:
    """
    Busca una alerta específica por su ID primario, precargando relaciones de trazabilidad.
    """
    return (
        db.query(AlertaVenta)
        .options(
            joinedload(AlertaVenta.venta),
            joinedload(AlertaVenta.cliente),
        )
        .filter(AlertaVenta.id == alerta_id)
        .first()
    )


def get_alert_stats(db: Session) -> Dict[str, int]:
    """
    Calcula los contadores ejecutivos de alertas:
    - Estado de revisión: pendientes, confirmadas, falsos positivos.
    - Distribución por nivel de atipicidad: bajo, medio, alto.
    - Total de alertas registradas en el sistema.
    """
    total = db.query(func.count(AlertaVenta.id)).scalar() or 0

    pendientes = (
        db.query(func.count(AlertaVenta.id))
        .filter(AlertaVenta.estado_revision == "PENDIENTE")
        .scalar()
        or 0
    )
    confirmadas = (
        db.query(func.count(AlertaVenta.id))
        .filter(AlertaVenta.estado_revision == "CONFIRMADA")
        .scalar()
        or 0
    )
    falsos_positivos = (
        db.query(func.count(AlertaVenta.id))
        .filter(AlertaVenta.estado_revision == "FALSO_POSITIVO")
        .scalar()
        or 0
    )

    bajo = (
        db.query(func.count(AlertaVenta.id))
        .filter(AlertaVenta.nivel_alerta == "BAJO")
        .scalar()
        or 0
    )
    medio = (
        db.query(func.count(AlertaVenta.id))
        .filter(AlertaVenta.nivel_alerta == "MEDIO")
        .scalar()
        or 0
    )
    alto = (
        db.query(func.count(AlertaVenta.id))
        .filter(AlertaVenta.nivel_alerta == "ALTO")
        .scalar()
        or 0
    )

    return {
        "total": total,
        "pendientes": pendientes,
        "confirmadas": confirmadas,
        "falsos_positivos": falsos_positivos,
        "bajo": bajo,
        "medio": medio,
        "alto": alto,
    }


def submit_feedback(
    db: Session,
    alerta_id: int,
    feedback_data: AlertFeedbackUpdate,
    revisado_por: str,
) -> AlertaVenta:
    """
    Registra la retroalimentación de un operador humano sobre una alerta:
    - Actualiza estado_revision a CONFIRMADA o FALSO_POSITIVO.
    - Registra notas u observaciones explicativas.
    - Asigna identidad del usuario autenticado (revisado_por) y timestamp de auditoría.
    - Mantiene estrictamente inalterado el puntaje de anomalía original del modelo.
    """
    alerta = get_alert_by_id(db=db, alerta_id=alerta_id)
    if not alerta:
        raise ValueError(f"No se encontró la alerta con ID {alerta_id}")

    alerta.estado_revision = feedback_data.estado_revision
    alerta.retroalimentacion_notas = feedback_data.retroalimentacion_notas
    alerta.revisado_por = revisado_por
    alerta.fecha_revision = datetime.now(timezone.utc)

    db.commit()
    db.refresh(alerta)
    logger.info(
        f"Feedback registrado para Alerta ID {alerta.id}: estado={alerta.estado_revision}, revisado_por={revisado_por}"
    )
    return alerta


def get_models_status(db: Session) -> Dict[str, Any]:
    """
    Verifica el estado operativo actual de los modelos ML registrados en la base de datos:
    - K-Means (KMEANS_CLIENTES)
    - Isolation Forest (ISOLATION_FOREST_VENTAS)

    Determina disponibilidad real comprobando que el registro esté ACTIVO y que el
    artefacto físico .joblib exista en el almacenamiento local.
    """
    def _inspect_model(tipo: str) -> Dict[str, Any]:
        modelo = (
            db.query(ModeloEntrenamiento)
            .filter(
                ModeloEntrenamiento.tipo_modelo == tipo,
                ModeloEntrenamiento.estado == "ACTIVO",
            )
            .order_by(ModeloEntrenamiento.id.desc())
            .first()
        )
        if not modelo:
            return {
                "disponible": False,
                "estado": "NO DISPONIBLE",
                "version": None,
                "fecha_entrenamiento": None,
                "total_muestras": 0,
                "mensaje": f"No existe un modelo {tipo} activo en el sistema.",
            }

        # Verificar existencia física del archivo
        archivo_existe = os.path.isfile(modelo.ruta_artefacto)
        if not archivo_existe:
            return {
                "disponible": False,
                "estado": "NO DISPONIBLE",
                "version": modelo.version,
                "fecha_entrenamiento": modelo.created_at,
                "total_muestras": modelo.total_muestras,
                "mensaje": f"El registro está ACTIVO pero el archivo físico no fue hallado en '{modelo.ruta_artefacto}'.",
            }

        return {
            "disponible": True,
            "estado": "DISPONIBLE",
            "version": modelo.version,
            "fecha_entrenamiento": modelo.created_at,
            "total_muestras": modelo.total_muestras,
            "mensaje": f"Modelo {tipo} operativo y listo para inferencia.",
        }

    return {
        "isolation_forest": _inspect_model(ISOLATION_FOREST_MODEL_TYPE),
        "kmeans": _inspect_model(KMEANS_MODEL_TYPE),
    }


def get_models_readiness(db: Session) -> Dict[str, Any]:
    """
    Calcula en tiempo real la preparación del sistema para entrenamiento de modelos ML:
    1. K-Means:
       - clientes_totales: total de clientes en el sistema.
       - clientes_elegibles: clientes con al menos 2 compras completadas (MIN_PURCHASES_FOR_KMEANS_PROFILE).
       - compras_minimas_por_cliente: 2.
       - clientes_minimos_requeridos: 15 (MIN_CUSTOMERS_FOR_KMEANS).
       - puede_entrenar: clientes_elegibles >= 15.
       - porcentaje_progreso: min(100.0, (clientes_elegibles / 15) * 100.0).
    2. Isolation Forest:
       - ventas_completadas: total de ventas con estado 'COMPLETADA'.
       - ventas_minimas_requeridas: 50 (MIN_SALES_FOR_ISOLATION_FOREST).
       - puede_entrenar: ventas_completadas >= 50.
       - porcentaje_progreso: min(100.0, (ventas_completadas / 50) * 100.0).
    """
    # 1. K-Means
    df_customers = build_customer_features(db=db)
    clientes_totales = len(df_customers)
    if not df_customers.empty and "total_compras" in df_customers.columns:
        clientes_elegibles = int((df_customers["total_compras"] >= MIN_PURCHASES_FOR_KMEANS_PROFILE).sum())
    else:
        clientes_elegibles = 0
    kmeans_puede = clientes_elegibles >= MIN_CUSTOMERS_FOR_KMEANS
    kmeans_progreso = round(min(100.0, (clientes_elegibles / MIN_CUSTOMERS_FOR_KMEANS) * 100.0), 1)

    # 2. Isolation Forest
    ventas_completadas = (
        db.query(func.count(Venta.id))
        .filter(Venta.estado == "COMPLETADA")
        .scalar()
        or 0
    )
    if_puede = ventas_completadas >= MIN_SALES_FOR_ISOLATION_FOREST
    if_progreso = round(min(100.0, (ventas_completadas / MIN_SALES_FOR_ISOLATION_FOREST) * 100.0), 1)

    return {
        "kmeans": {
            "clientes_totales": clientes_totales,
            "clientes_elegibles": clientes_elegibles,
            "compras_minimas_por_cliente": MIN_PURCHASES_FOR_KMEANS_PROFILE,
            "clientes_minimos_requeridos": MIN_CUSTOMERS_FOR_KMEANS,
            "puede_entrenar": kmeans_puede,
            "porcentaje_progreso": kmeans_progreso,
        },
        "isolation_forest": {
            "ventas_completadas": ventas_completadas,
            "ventas_minimas_requeridas": MIN_SALES_FOR_ISOLATION_FOREST,
            "puede_entrenar": if_puede,
            "porcentaje_progreso": if_progreso,
        },
    }

