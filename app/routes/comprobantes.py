from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.routes.auth import get_current_user
from app.schemas.receipt import (
    AvailableSalesListResponse,
    ReceiptCreate,
    ReceiptDetailResponse,
    ReceiptListResponse,
    ReceiptStatsResponse,
)
from app.schemas.summary import (
    DailySummaryCreate,
    DailySummaryDetailResponse,
    DailySummaryListResponse,
    PendingBoletasResponse,
)
from app.services import receipt_service

router = APIRouter(dependencies=[Depends(get_current_user)])


# =============================================================================
# RUTAS ESTÁTICAS DE CONSULTA Y ESTADÍSTICAS
# =============================================================================

@router.get(
    "/ventas-disponibles",
    response_model=AvailableSalesListResponse,
    summary="Listar ventas completadas que aún no cuentan con comprobante emitido",
)
def get_available_sales(
    search: Optional[str] = Query(None, description="Búsqueda por número de venta, RUC o razón social"),
    skip: int = Query(0, ge=0, description="Cantidad de registros a omitir"),
    limit: int = Query(50, ge=1, le=100, description="Cantidad de registros por página"),
    db: Session = Depends(get_db),
):
    return receipt_service.get_available_sales(
        db=db,
        search=search,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/stats",
    response_model=ReceiptStatsResponse,
    summary="Obtener estadísticas operativas del módulo de comprobantes",
)
def get_receipt_stats(db: Session = Depends(get_db)):
    return receipt_service.get_receipt_stats(db=db)


# =============================================================================
# RUTAS DE RESÚMENES DIARIOS DE BOLETAS (DECLARADAS ANTES DE /{receipt_id})
# =============================================================================

@router.get(
    "/boletas-pendientes-resumen",
    response_model=PendingBoletasResponse,
    summary="Listar boletas pendientes de envío agrupadas por fecha de emisión en America/Lima",
)
def get_pending_boletas_by_date(db: Session = Depends(get_db)):
    return receipt_service.get_boletas_pendientes_resumen(db=db)


@router.get(
    "/resumenes-diarios",
    response_model=DailySummaryListResponse,
    summary="Listar historial de Resúmenes Diarios de Boletas generados y enviados",
)
def list_daily_summaries(
    skip: int = Query(0, ge=0, description="Cantidad de registros a omitir"),
    limit: int = Query(50, ge=1, le=100, description="Cantidad de registros por página"),
    db: Session = Depends(get_db),
):
    return receipt_service.get_resumenes_diarios(db=db, skip=skip, limit=limit)


@router.post(
    "/resumenes-diarios",
    response_model=DailySummaryDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generar un nuevo Resumen Diario de Boletas para una fecha seleccionada",
)
def create_daily_summary(
    summary_in: DailySummaryCreate,
    db: Session = Depends(get_db),
):
    return receipt_service.crear_resumen_diario(
        db=db,
        fecha_documentos=summary_in.fecha_documentos,
    )


@router.get(
    "/resumenes-diarios/{resumen_id}",
    response_model=DailySummaryDetailResponse,
    summary="Obtener detalle integral de un Resumen Diario y sus boletas asociadas",
)
def get_daily_summary_detail(
    resumen_id: int,
    db: Session = Depends(get_db),
):
    return receipt_service.get_resumen_diario_by_id(db=db, resumen_id=resumen_id)


@router.post(
    "/resumenes-diarios/{resumen_id}/enviar",
    response_model=DailySummaryDetailResponse,
    summary="Reintentar o enviar a SUNAT un Resumen Diario existente",
)
def send_existing_daily_summary(
    resumen_id: int,
    db: Session = Depends(get_db),
):
    return receipt_service.enviar_resumen_diario(db=db, resumen_id=resumen_id)


@router.post(
    "/resumenes-diarios/{resumen_id}/consultar",
    response_model=DailySummaryDetailResponse,
    summary="Consultar el ticket de un Resumen Diario ante SUNAT",
)
def check_daily_summary_ticket(
    resumen_id: int,
    db: Session = Depends(get_db),
):
    return receipt_service.consultar_resumen_diario(db=db, resumen_id=resumen_id)


# =============================================================================
# RUTAS DE COLECCIÓN Y GENERACIÓN DE COMPROBANTES
# =============================================================================

@router.get(
    "",
    response_model=ReceiptListResponse,
    summary="Listar comprobantes emitidos con filtros y paginación",
)
def list_receipts(
    search: Optional[str] = Query(None, description="Búsqueda por serie, correlativo, número de venta o cliente"),
    tipo: Optional[str] = Query(None, description="Filtrar por tipo: '01' (Factura), '03' (Boleta)"),
    estado: Optional[str] = Query(None, description="Filtrar por estado: 'PENDIENTE_ENVIO', etc."),
    skip: int = Query(0, ge=0, description="Cantidad de registros a omitir"),
    limit: int = Query(50, ge=1, le=100, description="Cantidad de registros por página"),
    db: Session = Depends(get_db),
):
    return receipt_service.get_receipts(
        db=db,
        search=search,
        tipo=tipo,
        estado=estado,
        skip=skip,
        limit=limit,
    )


@router.post(
    "",
    response_model=ReceiptDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generar un nuevo comprobante de pago electrónico para una venta",
)
def create_receipt(
    receipt_in: ReceiptCreate,
    db: Session = Depends(get_db),
):
    comprobante = receipt_service.create_receipt(db=db, receipt_in=receipt_in)
    return receipt_service.get_receipt_by_id(db=db, receipt_id=comprobante.id)


# =============================================================================
# RUTA DE DETALLE INDIVIDUAL Y DESPACHO DE FACTURA
# =============================================================================

@router.get(
    "/{receipt_id}",
    response_model=ReceiptDetailResponse,
    summary="Obtener detalle integral de un comprobante de pago",
)
def get_receipt_detail(
    receipt_id: int,
    db: Session = Depends(get_db),
):
    return receipt_service.get_receipt_by_id(db=db, receipt_id=receipt_id)


@router.post(
    "/{receipt_id}/enviar-sunat",
    summary="Enviar una Factura electrónica ('01') al microservicio SUNAT",
)
def send_invoice_to_sunat(
    receipt_id: int,
    db: Session = Depends(get_db),
):
    return receipt_service.enviar_factura_sunat(db=db, comprobante_id=receipt_id)
