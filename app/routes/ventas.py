from datetime import date
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.routes.auth import get_current_user
from app.schemas.sale import (
    SaleCreate,
    SaleListResponse,
    SaleResponse,
    SaleStatsResponse,
)
from app.services import sale_service
from app.services.ai.background_tasks import process_sale_ai_background

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("", response_model=SaleListResponse, summary="Listar ventas con filtros y paginación")
def list_sales(
    search: Optional[str] = Query(None, description="Búsqueda por número de venta, RUC o razón social"),
    estado: Optional[str] = Query(None, description="Filtrar por estado de la venta"),
    fecha_desde: Optional[date] = Query(None, description="Fecha inicio (YYYY-MM-DD)"),
    fecha_hasta: Optional[date] = Query(None, description="Fecha fin (YYYY-MM-DD)"),
    skip: int = Query(0, ge=0, description="Registros a omitir"),
    limit: int = Query(50, ge=1, le=100, description="Registros por página"),
    db: Session = Depends(get_db),
):
    """
    Retorna el listado paginado de ventas con búsqueda por correlativo/cliente,
    filtros por estado y rango de fechas.
    """
    return sale_service.get_sales(
        db=db,
        search=search,
        estado=estado,
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
        skip=skip,
        limit=limit,
    )


@router.get("/stats", response_model=SaleStatsResponse, summary="Obtener estadísticas globales de ventas")
def get_sale_stats(db: Session = Depends(get_db)):
    """
    Retorna los KPIs reales derivados de las ventas registradas:
    total de ventas, monto facturado, ticket promedio y unidades físicas vendidas.
    """
    return sale_service.get_sale_stats(db=db)


@router.get("/{venta_id}", response_model=SaleResponse, summary="Obtener detalle de una venta por ID")
def get_sale(venta_id: int, db: Session = Depends(get_db)):
    """
    Retorna los datos generales de la venta, cliente y el desglose de productos con precios unitarios históricos.
    """
    sale = sale_service.get_sale_by_id(db=db, venta_id=venta_id)
    if not sale:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Venta con ID {venta_id} no encontrada."
        )
    return sale


@router.post(
    "",
    response_model=SaleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Registrar y procesar una nueva venta",
)
def create_sale(
    sale_in: SaleCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Procesa una operación de venta atómica:
    - Valida cliente activo.
    - Bloquea existencias pesimistamente.
    - Valida stock suficiente.
    - Calcula precios, subtotales, IGV y total con Decimal.
    - Genera correlativo V-XXXXXX único.
    - Descuenta inventario y confirma la transacción.
    - Programa el análisis contextual por IA en segundo plano (no bloqueante e independiente).
    """
    sale = sale_service.create_sale(db=db, sale_in=sale_in)

    # Disparar análisis IA en segundo plano únicamente si la venta fue confirmada
    if sale and sale.get("estado") == "COMPLETADA":
        background_tasks.add_task(process_sale_ai_background, sale["id"])

    return sale

