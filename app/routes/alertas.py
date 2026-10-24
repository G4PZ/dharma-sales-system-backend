"""
Router de Alertas del Motor IA y Auditoría de Operaciones (Fase 5).

Endpoints Disponibles:
- GET   /api/alertas                 : Listado paginado con filtros de nivel, revisión, cliente y fechas.
- GET   /api/alertas/stats           : Estadísticas agregadas (pendientes, confirmadas, distribución por nivel).
- GET   /api/alertas/modelos/estado  : Disponibilidad operativa de Isolation Forest y K-Means.
- GET   /api/alertas/{alerta_id}     : Detalle explicativo de una alerta con relaciones y factores contextuales.
- PATCH /api/alertas/{alerta_id}/feedback : Retroalimentación humana (CONFIRMADA / FALSO_POSITIVO).
- POST  /api/alertas/analizar/{venta_id}  : Análisis manual bajo demanda de una operación de venta.
"""

from datetime import date
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.sale import Venta
from app.routes.auth import get_current_user
from app.schemas.ai import (
    AlertDetailResponse,
    AlertFeedbackUpdate,
    AlertListResponse,
    AlertStatsResponse,
    ModelsReadinessResponse,
    ModelsStatusResponse,
    SaleAnalysisResponse,
    TrainingExecutionResponse,
)
from app.services.ai import alert_service, detector
from app.services.ai.detector import ModelNotAvailableError
from app.services.ai.trainer import (
    InsufficientDataError,
    TrainingError,
    TrainingInProgressError,
    train_isolation_forest_sales,
    train_kmeans_customers,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("", response_model=AlertListResponse, summary="Listar alertas IA con filtros y paginación")
def list_alerts(
    nivel: Optional[str] = Query(None, description="Filtrar por nivel: BAJO, MEDIO, ALTO"),
    estado_revision: Optional[str] = Query(None, description="Filtrar por revisión: PENDIENTE, CONFIRMADA, FALSO_POSITIVO"),
    cliente_id: Optional[int] = Query(None, description="Filtrar por ID de cliente"),
    fecha_desde: Optional[date] = Query(None, description="Fecha de creación inicial (YYYY-MM-DD)"),
    fecha_hasta: Optional[date] = Query(None, description="Fecha de creación final (YYYY-MM-DD)"),
    skip: int = Query(0, ge=0, description="Cantidad de registros a omitir"),
    limit: int = Query(50, ge=1, le=100, description="Cantidad de registros por página"),
    db: Session = Depends(get_db),
):
    """
    Retorna el listado paginado de alertas detectadas ordenadas cronológicamente en forma descendente.
    """
    return alert_service.list_alerts(
        db=db,
        nivel=nivel,
        estado_revision=estado_revision,
        cliente_id=cliente_id,
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
        skip=skip,
        limit=limit,
    )


@router.get("/stats", response_model=AlertStatsResponse, summary="Obtener estadísticas ejecutivas de alertas")
def get_alert_stats(db: Session = Depends(get_db)):
    """
    Retorna contadores de alertas agrupados por estado de revisión (pendientes, confirmadas,
    falsos positivos) y por nivel operativo de severidad (bajo, medio, alto).
    """
    return alert_service.get_alert_stats(db=db)


@router.get("/modelos/estado", response_model=ModelsStatusResponse, summary="Consultar disponibilidad de modelos ML")
def get_models_status(db: Session = Depends(get_db)):
    """
    Informa la disponibilidad operativa actual de los modelos Isolation Forest y K-Means en el sistema.
    En la base de datos real sin entrenar, retorna 'NO DISPONIBLE' de manera controlada.
    """
    return alert_service.get_models_status(db=db)


@router.get("/modelos/preparacion", response_model=ModelsReadinessResponse, summary="Consultar preparación de datos para entrenamiento")
def get_models_readiness(db: Session = Depends(get_db)):
    """
    Informa en tiempo real si existen suficientes datos en el sistema para entrenar K-Means
    (mínimo 15 clientes con 2+ compras) e Isolation Forest (mínimo 50 ventas completadas),
    junto con los contadores y porcentajes de avance hacia el umbral requerido.
    """
    return alert_service.get_models_readiness(db=db)


@router.post(
    "/modelos/entrenar/kmeans",
    response_model=TrainingExecutionResponse,
    summary="Iniciar entrenamiento administrativo de K-Means",
)
def train_kmeans(
    db: Session = Depends(get_db),
):
    """
    Ejecuta el entrenamiento administrativo del modelo K-Means para segmentación comercial de clientes.
    Requiere al menos 15 clientes elegibles con 2+ compras.
    Si los datos son insuficientes, responde con HTTP 409 Conflict.
    """
    try:
        result = train_kmeans_customers(db=db)
        detector.clear_model_cache()
        return result
    except InsufficientDataError as data_exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(data_exc),
        )
    except TrainingInProgressError as prog_exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(prog_exc),
        )
    except TrainingError as train_exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el entrenamiento de K-Means: {str(train_exc)}",
        )


@router.post(
    "/modelos/entrenar/isolation-forest",
    response_model=TrainingExecutionResponse,
    summary="Iniciar entrenamiento administrativo de Isolation Forest",
)
def train_isolation_forest(
    db: Session = Depends(get_db),
):
    """
    Ejecuta el entrenamiento administrativo del modelo Isolation Forest para detección de ventas inusuales.
    Requiere al menos 50 ventas completadas.
    Si los datos son insuficientes, responde con HTTP 409 Conflict.
    """
    try:
        result = train_isolation_forest_sales(db=db)
        detector.clear_model_cache()
        return result
    except InsufficientDataError as data_exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(data_exc),
        )
    except TrainingInProgressError as prog_exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(prog_exc),
        )
    except TrainingError as train_exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el entrenamiento de Isolation Forest: {str(train_exc)}",
        )


@router.get("/{alerta_id}", response_model=AlertDetailResponse, summary="Obtener detalle de una alerta por ID")
def get_alert(alerta_id: int, db: Session = Depends(get_db)):
    """
    Retorna el detalle completo de la alerta incluyendo los factores explicativos contribuyentes,
    los datos de la venta, cliente y trazabilidad de los modelos ML participantes.
    """
    alerta = alert_service.get_alert_by_id(db=db, alerta_id=alerta_id)
    if not alerta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontró la alerta con ID {alerta_id}",
        )
    return alerta


@router.patch("/{alerta_id}/feedback", response_model=AlertDetailResponse, summary="Registrar retroalimentación humana")
def submit_feedback(
    alerta_id: int,
    body: AlertFeedbackUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Registra la revisión de un operador sobre la alerta (CONFIRMADA o FALSO_POSITIVO).
    La identidad del revisor se obtiene automáticamente de la sesión autenticada.
    """
    reviewer_email = current_user.get("email") or current_user.get("name") or "usuario_autenticado"
    try:
        updated_alert = alert_service.submit_feedback(
            db=db,
            alerta_id=alerta_id,
            feedback_data=body,
            revisado_por=reviewer_email,
        )
        return updated_alert
    except ValueError as val_exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_exc),
        )


@router.post(
    "/analizar/numero/{numero_venta}",
    response_model=SaleAnalysisResponse,
    summary="Ejecutar análisis contextual de una venta por su número correlativo",
)
def analyze_sale_by_number(
    numero_venta: str,
    db: Session = Depends(get_db),
):
    """
    Ejecuta el análisis contextual de una operación de venta identificada por su número correlativo (ej. 'V-000127').
    Normaliza el número de venta eliminando espacios al inicio y final y convirtiéndolo a mayúsculas.
    Si la venta no existe en la base de datos, responde con HTTP 404 Not Found.
    Reutiliza exactamente la lógica del detector existente (detector.analyze_sale).
    """
    clean_num = numero_venta.strip().upper()
    venta = db.query(Venta).filter(Venta.numero_venta == clean_num).first()
    if not venta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontró la venta '{clean_num}'.",
        )

    try:
        result = detector.analyze_sale(
            db=db,
            venta_id=venta.id,
            persist_alert=True,
        )
        if isinstance(result, dict):
            result["numero_venta"] = venta.numero_venta
        return result
    except ModelNotAvailableError as model_exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(model_exc),
        )
    except ValueError as not_found_exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(not_found_exc),
        )
    except Exception as exc:
        logger.error(f"Error inesperado durante el análisis manual de la venta {clean_num} (ID {venta.id}): {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocurrió un error al procesar el análisis de la venta: {exc}",
        )


@router.post(
    "/analizar/{venta_id}",
    response_model=SaleAnalysisResponse,
    summary="Ejecutar análisis contextual de una venta bajo demanda",
)
def analyze_sale_manually(
    venta_id: int,
    db: Session = Depends(get_db),
):
    """
    Ejecuta el análisis contextual de una operación de venta existente utilizando los modelos ML activos.
    Si no existe un modelo Isolation Forest entrenado y activo, responde con HTTP 409 Conflict controlado.
    Si la venta no es anómala (NORMAL), devuelve el score y métricas sin crear una alerta en base de datos.
    """
    try:
        result = detector.analyze_sale(
            db=db,
            venta_id=venta_id,
            persist_alert=True,
        )
        venta = db.query(Venta).filter(Venta.id == venta_id).first()
        if venta and isinstance(result, dict):
            result["numero_venta"] = venta.numero_venta
        return result
    except ModelNotAvailableError as model_exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(model_exc),
        )
    except ValueError as not_found_exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(not_found_exc),
        )
    except Exception as exc:
        logger.error(f"Error inesperado durante el análisis manual de la venta {venta_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocurrió un error al procesar el análisis de la venta: {exc}",
        )
