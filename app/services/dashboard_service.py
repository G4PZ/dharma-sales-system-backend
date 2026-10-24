from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo
from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.ai import AlertaVenta
from app.models.receipt import Comprobante
from app.models.sale import Venta
from app.schemas.dashboard import (
    AlertaNivelItem,
    AlertaRecienteItem,
    AlertasIAResumen,
    ComprobanteEstadoItem,
    ComprobantesResumen,
    DashboardResumenResponse,
    RangoFechasInfo,
    VentaDiaEvolucion,
    VentaRecienteItem,
    VentasResumen,
)

LIMA_TZ = ZoneInfo("America/Lima")
TWO_PLACES = Decimal("0.01")
DIAS_ABREV = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def get_dashboard_summary(
    db: Session,
    fecha_desde: Optional[date] = None,
    fecha_hasta: Optional[date] = None,
) -> DashboardResumenResponse:
    """
    Consolida de manera atómica y eficiente los indicadores ejecutivos de Business Intelligence
    para el Dashboard Dharma, aplicando de forma global el filtro por rango de fechas:
    1. Si no se especifican fechas, se toma por defecto los últimos 7 días (día actual y 6 días previos).
    2. Valida que fecha_desde <= fecha_hasta.
    3. Interpreta las fechas en America/Lima garantizando que fecha_hasta incluya el día completo (hasta 23:59:59.999999).
    4. Recalcula todos los KPIs, visualizaciones y tablas recientes usando exclusivamente ese rango.
    """
    now_lima = datetime.now(LIMA_TZ)
    hoy = now_lima.date()

    # 1. Determinación de fechas objetivo
    if fecha_desde is None and fecha_hasta is None:
        target_desde = hoy - timedelta(days=6)
        target_hasta = hoy
    elif fecha_desde is not None and fecha_hasta is None:
        target_desde = fecha_desde
        target_hasta = hoy
    elif fecha_desde is None and fecha_hasta is not None:
        target_desde = fecha_hasta - timedelta(days=6)
        target_hasta = fecha_hasta
    else:
        target_desde = fecha_desde
        target_hasta = fecha_hasta

    # 2. Validación de rango temporal
    if target_desde > target_hasta:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La fecha inicial (fecha_desde) no puede ser posterior a la fecha final (fecha_hasta).",
        )

    # 3. Límites en zona horaria America/Lima convertidos a UTC
    inicio_local = datetime.combine(target_desde, time.min, tzinfo=LIMA_TZ)
    inicio_utc = inicio_local.astimezone(timezone.utc)
    fin_local = datetime.combine(target_hasta + timedelta(days=1), time.min, tzinfo=LIMA_TZ)
    fin_utc = fin_local.astimezone(timezone.utc)

    num_days = (target_hasta - target_desde).days + 1
    rango_info = RangoFechasInfo(
        fecha_desde=target_desde.isoformat(),
        fecha_hasta=target_hasta.isoformat(),
        dias_totales=num_days,
    )

    # -------------------------------------------------------------------------
    # 4. KPI 1 y KPI 2: Ventas e ingresos en el rango
    # -------------------------------------------------------------------------
    ventas_cantidad = (
        db.query(func.count(Venta.id))
        .filter(Venta.created_at >= inicio_utc, Venta.created_at < fin_utc)
        .scalar()
        or 0
    )

    ingresos_val = (
        db.query(func.coalesce(func.sum(Venta.total), 0))
        .filter(
            Venta.estado == "COMPLETADA",
            Venta.created_at >= inicio_utc,
            Venta.created_at < fin_utc,
        )
        .scalar()
    )
    ingresos = Decimal(str(ingresos_val or 0)).quantize(
        TWO_PLACES, rounding=ROUND_HALF_UP
    )

    ventas_resumen = VentasResumen(
        cantidad=ventas_cantidad,
        ingresos=ingresos,
    )

    # -------------------------------------------------------------------------
    # 5. KPI 3: Comprobantes electrónicos emitidos en el rango
    #    Estados soportados: 'PENDIENTE_ENVIO', 'ENVIADO', 'ACEPTADO', 'RECHAZADO', 'ERROR_ENVIO'
    # -------------------------------------------------------------------------
    filtro_comprobantes_rango = (
        Comprobante.fecha_emision >= inicio_utc,
        Comprobante.fecha_emision < fin_utc,
    )

    total_comprobantes = (
        db.query(func.count(Comprobante.id))
        .filter(*filtro_comprobantes_rango)
        .scalar()
        or 0
    )
    aceptados_count = (
        db.query(func.count(Comprobante.id))
        .filter(Comprobante.estado == "ACEPTADO", *filtro_comprobantes_rango)
        .scalar()
        or 0
    )
    pendientes_count = (
        db.query(func.count(Comprobante.id))
        .filter(
            Comprobante.estado.in_(["PENDIENTE_ENVIO", "ENVIADO"]),
            *filtro_comprobantes_rango,
        )
        .scalar()
        or 0
    )
    rechazados_count = (
        db.query(func.count(Comprobante.id))
        .filter(
            Comprobante.estado.in_(["RECHAZADO", "ERROR_ENVIO"]),
            *filtro_comprobantes_rango,
        )
        .scalar()
        or 0
    )

    comprobantes_resumen = ComprobantesResumen(
        total=total_comprobantes,
        aceptados=aceptados_count,
        pendientes=pendientes_count,
        rechazados=rechazados_count,
    )

    # Distribución porcentual para gráfico de dona
    def _calc_pct(count: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return round((count / total) * 100.0, 1)

    comprobantes_distribucion: List[ComprobanteEstadoItem] = [
        ComprobanteEstadoItem(
            estado="ACEPTADO",
            nombre="Aceptados",
            cantidad=aceptados_count,
            porcentaje=_calc_pct(aceptados_count, total_comprobantes),
        ),
        ComprobanteEstadoItem(
            estado="PENDIENTE_ENVIO",
            nombre="Pendientes",
            cantidad=pendientes_count,
            porcentaje=_calc_pct(pendientes_count, total_comprobantes),
        ),
        ComprobanteEstadoItem(
            estado="RECHAZADO",
            nombre="Rechazados",
            cantidad=rechazados_count,
            porcentaje=_calc_pct(rechazados_count, total_comprobantes),
        ),
    ]

    # -------------------------------------------------------------------------
    # 6. KPI 4: Alertas del Módulo IA creadas en el rango
    # -------------------------------------------------------------------------
    filtro_alertas_rango = (
        AlertaVenta.created_at >= inicio_utc,
        AlertaVenta.created_at < fin_utc,
    )

    total_alertas = (
        db.query(func.count(AlertaVenta.id))
        .filter(*filtro_alertas_rango)
        .scalar()
        or 0
    )
    alertas_pendientes = (
        db.query(func.count(AlertaVenta.id))
        .filter(
            AlertaVenta.estado_revision == "PENDIENTE",
            *filtro_alertas_rango,
        )
        .scalar()
        or 0
    )
    alertas_alto = (
        db.query(func.count(AlertaVenta.id))
        .filter(
            AlertaVenta.nivel_alerta == "ALTO",
            *filtro_alertas_rango,
        )
        .scalar()
        or 0
    )
    alertas_medio = (
        db.query(func.count(AlertaVenta.id))
        .filter(
            AlertaVenta.nivel_alerta == "MEDIO",
            *filtro_alertas_rango,
        )
        .scalar()
        or 0
    )
    alertas_bajo = (
        db.query(func.count(AlertaVenta.id))
        .filter(
            AlertaVenta.nivel_alerta == "BAJO",
            *filtro_alertas_rango,
        )
        .scalar()
        or 0
    )

    alertas_ia_resumen = AlertasIAResumen(
        total=total_alertas,
        pendientes=alertas_pendientes,
        nivel_alto=alertas_alto,
        bajo=alertas_bajo,
        medio=alertas_medio,
        alto=alertas_alto,
    )

    alertas_por_nivel: List[AlertaNivelItem] = [
        AlertaNivelItem(nivel="ALTO", cantidad=alertas_alto),
        AlertaNivelItem(nivel="MEDIO", cantidad=alertas_medio),
        AlertaNivelItem(nivel="BAJO", cantidad=alertas_bajo),
    ]

    # -------------------------------------------------------------------------
    # 7. Visualización 1: Evolución de ventas en el rango seleccionado
    # -------------------------------------------------------------------------
    dias_dict: Dict[str, Dict[str, any]] = {}
    for d_idx in range(num_days):
        cur_date = target_desde + timedelta(days=d_idx)
        date_iso = cur_date.isoformat()
        if num_days <= 14:
            dia_etiqueta = f"{DIAS_ABREV[cur_date.weekday()]} {cur_date.day:02d}"
        else:
            dia_etiqueta = f"{cur_date.day:02d}/{cur_date.month:02d}"

        dias_dict[date_iso] = {
            "fecha": date_iso,
            "dia": dia_etiqueta,
            "total": Decimal("0.00"),
            "cantidad": 0,
        }

    # Consulta única indexada de ventas completadas en el rango
    ventas_rango = (
        db.query(Venta.total, Venta.created_at)
        .filter(
            Venta.estado == "COMPLETADA",
            Venta.created_at >= inicio_utc,
            Venta.created_at < fin_utc,
        )
        .all()
    )

    for total_val, created_at in ventas_rango:
        created_at_lima = created_at.astimezone(LIMA_TZ)
        sale_iso = created_at_lima.date().isoformat()
        if sale_iso in dias_dict:
            dias_dict[sale_iso]["total"] += Decimal(str(total_val or 0))
            dias_dict[sale_iso]["cantidad"] += 1

    evolucion_ventas: List[VentaDiaEvolucion] = [
        VentaDiaEvolucion(
            fecha=item["fecha"],
            dia=item["dia"],
            total=item["total"].quantize(TWO_PLACES, rounding=ROUND_HALF_UP),
            cantidad=item["cantidad"],
        )
        for item in dias_dict.values()
    ]

    # -------------------------------------------------------------------------
    # 8. Alertas recientes dentro del rango seleccionado (máximo 5)
    # -------------------------------------------------------------------------
    alertas_recientes_db = (
        db.query(AlertaVenta)
        .filter(*filtro_alertas_rango)
        .order_by(AlertaVenta.created_at.desc(), AlertaVenta.id.desc())
        .limit(5)
        .all()
    )

    alertas_recientes: List[AlertaRecienteItem] = [
        AlertaRecienteItem(
            id=a.id,
            numero_venta=a.numero_venta,
            cliente_razon_social=a.cliente_razon_social,
            nivel_alerta=a.nivel_alerta,
            estado_revision=a.estado_revision,
            descripcion=a.descripcion,
            created_at=a.created_at,
        )
        for a in alertas_recientes_db
    ]

    # -------------------------------------------------------------------------
    # 9. Ventas recientes dentro del rango seleccionado (máximo 5)
    # -------------------------------------------------------------------------
    ventas_recientes_db = (
        db.query(Venta)
        .filter(Venta.created_at >= inicio_utc, Venta.created_at < fin_utc)
        .order_by(Venta.created_at.desc(), Venta.id.desc())
        .limit(5)
        .all()
    )

    ventas_recientes: List[VentaRecienteItem] = [
        VentaRecienteItem(
            id=v.id,
            numero_venta=v.numero_venta,
            cliente_razon_social=v.cliente.razon_social if v.cliente else None,
            fecha=v.created_at,
            total=v.total,
            estado=v.estado,
        )
        for v in ventas_recientes_db
    ]

    return DashboardResumenResponse(
        rango_fechas=rango_info,
        ventas=ventas_resumen,
        ventas_hoy=ventas_resumen,
        comprobantes=comprobantes_resumen,
        alertas_ia=alertas_ia_resumen,
        evolucion_ventas=evolucion_ventas,
        comprobantes_distribucion=comprobantes_distribucion,
        alertas_por_nivel=alertas_por_nivel,
        alertas_recientes=alertas_recientes,
        ventas_recientes=ventas_recientes,
    )
