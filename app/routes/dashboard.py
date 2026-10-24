"""
Router del Dashboard Dharma (Business Intelligence).

Provee métricas ejecutivas, indicadores clave de rendimiento (KPIs), series
temporales agregadas y registros recientes para el tablero principal del sistema.
"""

from datetime import date
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.routes.auth import get_current_user
from app.schemas.dashboard import DashboardResumenResponse
from app.services import dashboard_service

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get(
    "/resumen",
    response_model=DashboardResumenResponse,
    summary="Obtener resumen ejecutivo e indicadores de Business Intelligence para el dashboard",
)
def get_dashboard_summary(
    fecha_desde: Optional[date] = Query(
        None, description="Fecha inicial del rango (YYYY-MM-DD)"
    ),
    fecha_hasta: Optional[date] = Query(
        None, description="Fecha final del rango (YYYY-MM-DD)"
    ),
    db: Session = Depends(get_db),
):
    """
    Retorna el resumen consolidado de Business Intelligence recalculado dinámicamente
    para el rango de fechas seleccionado (por defecto últimos 7 días):
    - Ventas e ingresos correspondientes al rango en hora peruana.
    - Estado real de los comprobantes electrónicos emitidos en el rango.
    - Métricas ejecutivas y distribución de severidad de Alertas IA en el rango.
    - Evolución de ventas día a día dentro del rango.
    - Distribución porcentual de comprobantes en el rango.
    - Alertas operativas más recientes generadas dentro del rango.
    - Ventas más recientes registradas dentro del rango.
    """
    return dashboard_service.get_dashboard_summary(
        db=db,
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
    )
