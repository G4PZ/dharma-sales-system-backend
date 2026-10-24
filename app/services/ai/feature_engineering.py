"""
Servicio de Ingeniería de Características (Feature Engineering) para el Motor IA.

Responsabilidades y Principios de Diseño:
1. Separación Estricta de Unidades de Análisis:
   - DATASET K-MEANS: Unidad de análisis = CLIENTE.
     Descubre perfiles de comportamiento a partir del historial comercial agregado.
   - DATASET ISOLATION FOREST: Unidad de análisis = VENTA.
     Detecta operaciones atípicas en el espacio multidimensional de transacciones.

2. REGLA CRÍTICA DE NO DATA LEAKAGE:
   - Para cualquier venta analizada, sus características históricas se calculan utilizando
     EXCLUSIVAMENTE operaciones con estado COMPLETADA anteriores a dicha venta:
     (created_at < fecha_objetivo) OR (created_at == fecha_objetivo AND id < id_objetivo).
   - NUNCA se utiliza la propia venta analizada ni ventas posteriores para construir su historial.

3. Clientes Nuevos (Cold Start):
   - Se conserva semánticamente la ausencia de historial asignando np.nan a promedios,
     máximos, intervalos y ratios relativos, con compras_previas = 0 y primera_compra = 1.
   - NO se imputan prematuramente ceros ni medias en esta fase.

4. Desacoplamiento de Facturación SUNAT y Metadata:
   - No se utilizan RUC, número de comprobante, series ni respuestas SUNAT como variables ML.
   - Los identificadores (cliente_id, venta_id, timestamps) se mantienen únicamente como metadata.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.models.client import Cliente
from app.models.sale import DetalleVenta, Venta

# Zona horaria oficial para operaciones comerciales de Dharma E.I.R.L.
LIMA_TZ = ZoneInfo("America/Lima")

# Estado de negocio válido para considerar operaciones efectivas
SALE_STATUS_COMPLETED = "COMPLETADA"

# =============================================================================
# DEFINICIÓN ESTRICTA DE CARACTERÍSTICAS NUMÉRICAS PARA MODELOS ML
# (Excluye identificadores, strings, RUC, nombres y metadatos)
# =============================================================================

KMEANS_FEATURE_COLUMNS: List[str] = [
    "total_compras",
    "monto_promedio",
    "monto_total_acumulado",
    "unidades_promedio",
    "productos_distintos_historicos",
    "frecuencia_30d",
    "recencia_dias",
    "antiguedad_cliente_dias",
    "dias_entre_compras_promedio",
]

ISOLATION_FOREST_FEATURE_COLUMNS: List[str] = [
    # Características propias de la transacción
    "monto_total",
    "unidades_totales",
    "productos_distintos",
    "lineas_detalle",
    "concentracion_importe",
    "concentracion_unidades",
    # Características temporales locales
    "hora_operacion",
    "dia_semana",
    "es_fin_semana",
    # Contexto histórico del cliente (estrictamente anterior)
    "compras_previas",
    "compras_previas_30d",
    "monto_promedio_previo",
    "monto_maximo_previo",
    "unidades_promedio_previas",
    "dias_desde_compra_anterior",
    "antiguedad_cliente_dias",
    "primera_compra",
    "diversidad_productos_previa",
    # Ratios contextuales relativos
    "ratio_monto_vs_promedio_cliente",
    "ratio_unidades_vs_promedio_cliente",
]


def _to_lima_tz(dt: datetime) -> datetime:
    """
    Normaliza un timestamp a la zona horaria local de Trujillo / Lima (America/Lima).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LIMA_TZ)


def _get_default_reference_datetime() -> datetime:
    """
    Retorna la fecha/hora de referencia actual en UTC de forma centralizada.
    """
    return datetime.now(timezone.utc)


# =============================================================================
# DATASET PARA K-MEANS — UNIDAD DE ANÁLISIS: CLIENTE
# =============================================================================

def build_customer_features(
    db: Session,
    reference_datetime: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Construye el dataset de características comerciales históricas por CLIENTE para K-Means.

    Estrategia de Optimización:
    - Evita el patrón N+1 precargando todos los clientes, ventas completadas y sus detalles
      en 3 consultas SQL consolidadas, realizando las agregaciones temporales en memoria.

    Regla Temporal:
    - Utiliza `reference_datetime` para reconstruir cómo se comportaba el cliente hasta ese momento.
      Cualquier venta posterior a `reference_datetime` es excluida.
    """
    if reference_datetime is None:
        ref_dt = _get_default_reference_datetime()
    else:
        ref_dt = reference_datetime if reference_datetime.tzinfo else reference_datetime.replace(tzinfo=timezone.utc)

    # 1. Cargar clientes activos/registrados
    clientes = db.query(Cliente).order_by(Cliente.id.asc()).all()
    if not clientes:
        cols = ["cliente_id"] + KMEANS_FEATURE_COLUMNS
        return pd.DataFrame(columns=cols)

    cliente_ids = [c.id for c in clientes]

    # 2. Cargar todas las ventas completadas hasta la fecha de referencia
    ventas = (
        db.query(Venta)
        .filter(
            Venta.cliente_id.in_(cliente_ids),
            Venta.estado == SALE_STATUS_COMPLETED,
            Venta.created_at <= ref_dt,
        )
        .order_by(Venta.cliente_id.asc(), Venta.created_at.asc(), Venta.id.asc())
        .all()
    )

    # 3. Cargar todos los detalles de las ventas filtradas en una sola consulta
    venta_ids = [v.id for v in ventas]
    detalles_por_venta: Dict[int, List[DetalleVenta]] = defaultdict(list)
    if venta_ids:
        detalles = (
            db.query(DetalleVenta)
            .filter(DetalleVenta.venta_id.in_(venta_ids))
            .all()
        )
        for d in detalles:
            detalles_por_venta[d.venta_id].append(d)

    # 4. Agrupar ventas por cliente
    ventas_por_cliente: Dict[int, List[Venta]] = defaultdict(list)
    for v in ventas:
        ventas_por_cliente[v.cliente_id].append(v)

    ventana_30d = ref_dt - timedelta(days=30)
    rows: List[Dict[str, Any]] = []

    for c in clientes:
        client_sales = ventas_por_cliente.get(c.id, [])
        total_compras = len(client_sales)

        # Cálculo de antigüedad del cliente en días
        c_created_at = c.created_at if c.created_at.tzinfo else c.created_at.replace(tzinfo=timezone.utc)
        antiguedad_cliente_dias = max(0.0, (ref_dt - c_created_at).total_seconds() / 86400.0)

        if total_compras == 0:
            # Cliente sin compras previas hasta la fecha de referencia
            rows.append({
                "cliente_id": c.id,
                "total_compras": 0,
                "monto_promedio": np.nan,
                "monto_total_acumulado": 0.0,
                "unidades_promedio": np.nan,
                "productos_distintos_historicos": 0,
                "frecuencia_30d": 0,
                "recencia_dias": np.nan,
                "antiguedad_cliente_dias": float(antiguedad_cliente_dias),
                "dias_entre_compras_promedio": np.nan,
            })
            continue

        # Totales monetarios
        montos = [float(v.total) for v in client_sales]
        monto_total_acumulado = float(sum(montos))
        monto_promedio = float(monto_total_acumulado / total_compras)

        # Unidades físicas y diversidad de catálogo
        unidades_por_venta: List[int] = []
        productos_set: Set[int] = set()
        for v in client_sales:
            items = detalles_por_venta.get(v.id, [])
            unidades_venta = sum(item.cantidad for item in items)
            unidades_por_venta.append(unidades_venta)
            for item in items:
                productos_set.add(item.producto_id)

        unidades_promedio = float(sum(unidades_por_venta) / total_compras)
        productos_distintos_historicos = len(productos_set)

        # Frecuencia en ventana reciente de 30 días
        frecuencia_30d = sum(
            1 for v in client_sales
            if (v.created_at if v.created_at.tzinfo else v.created_at.replace(tzinfo=timezone.utc)) >= ventana_30d
        )

        # Recencia: días desde la última compra registrada
        last_sale = client_sales[-1]
        last_sale_dt = last_sale.created_at if last_sale.created_at.tzinfo else last_sale.created_at.replace(tzinfo=timezone.utc)
        recencia_dias = max(0.0, (ref_dt - last_sale_dt).total_seconds() / 86400.0)

        # Intervalo promedio entre compras consecutivas (requiere al menos 2 compras)
        if total_compras >= 2:
            first_sale = client_sales[0]
            first_sale_dt = first_sale.created_at if first_sale.created_at.tzinfo else first_sale.created_at.replace(tzinfo=timezone.utc)
            total_intervalo_dias = max(0.0, (last_sale_dt - first_sale_dt).total_seconds() / 86400.0)
            dias_entre_compras_promedio = float(total_intervalo_dias / (total_compras - 1))
        else:
            dias_entre_compras_promedio = np.nan

        rows.append({
            "cliente_id": c.id,
            "total_compras": total_compras,
            "monto_promedio": monto_promedio,
            "monto_total_acumulado": monto_total_acumulado,
            "unidades_promedio": unidades_promedio,
            "productos_distintos_historicos": productos_distintos_historicos,
            "frecuencia_30d": frecuencia_30d,
            "recencia_dias": float(recencia_dias),
            "antiguedad_cliente_dias": float(antiguedad_cliente_dias),
            "dias_entre_compras_promedio": dias_entre_compras_promedio,
        })

    cols = ["cliente_id"] + KMEANS_FEATURE_COLUMNS
    df = pd.DataFrame(rows, columns=cols)
    return df


# =============================================================================
# DATASET PARA ISOLATION FOREST — UNIDAD DE ANÁLISIS: VENTA
# =============================================================================

class _SaleAccumulatorItem:
    """
    Estructura ligera en memoria para acumular el historial previo de un cliente
    sin incurrir en data leakage ni consultas repetitivas a la base de datos.
    """
    __slots__ = ("id", "created_at", "total", "units", "product_ids")

    def __init__(self, id: int, created_at: datetime, total: float, units: int, product_ids: Set[int]):
        self.id = id
        self.created_at = created_at
        self.total = total
        self.units = units
        self.product_ids = product_ids


def build_sales_features(db: Session) -> pd.DataFrame:
    """
    Construye el dataset de características a nivel de VENTA para Isolation Forest.

    Garantía de No Data Leakage:
    - Las ventas completadas se ordenan cronológicamente por (created_at ASC, id ASC).
    - Al procesar cada venta, su contexto histórico se deriva exclusivamente de las ventas
      previas acumuladas para ese cliente. La venta actual no forma parte de su propio historial.

    Estrategia de Optimización:
    - Evita el patrón N+1 precargando ventas, detalles y clientes en 3 consultas consolidadas.
    """
    ventas = (
        db.query(Venta)
        .filter(Venta.estado == SALE_STATUS_COMPLETED)
        .order_by(Venta.created_at.asc(), Venta.id.asc())
        .all()
    )

    columns_order = ["venta_id", "cliente_id", "created_at"] + ISOLATION_FOREST_FEATURE_COLUMNS
    if not ventas:
        return pd.DataFrame(columns=columns_order)

    venta_ids = [v.id for v in ventas]
    cliente_ids = list({v.cliente_id for v in ventas})

    # Cargar clientes relacionados
    clientes = db.query(Cliente).filter(Cliente.id.in_(cliente_ids)).all()
    cliente_map = {c.id: c for c in clientes}

    # Cargar detalles agrupados por venta
    detalles = db.query(DetalleVenta).filter(DetalleVenta.venta_id.in_(venta_ids)).all()
    detalles_por_venta: Dict[int, List[DetalleVenta]] = defaultdict(list)
    for d in detalles:
        detalles_por_venta[d.venta_id].append(d)

    # Acumulador histórico por cliente: cliente_id -> List[_SaleAccumulatorItem]
    historial_clientes: Dict[int, List[_SaleAccumulatorItem]] = defaultdict(list)
    rows: List[Dict[str, Any]] = []

    for v in ventas:
        cliente = cliente_map.get(v.cliente_id)
        c_items = detalles_por_venta.get(v.id, [])

        # 1. Características intrínsecas de la venta actual
        monto_total = float(v.total)
        unidades_totales = sum(item.cantidad for item in c_items)
        productos_distintos = len({item.producto_id for item in c_items})
        lineas_detalle = len(c_items)

        # Concentración de importe (mayor subtotal / suma de subtotales)
        subtotales = [float(item.subtotal) for item in c_items]
        sum_subtotales = sum(subtotales)
        concentracion_importe = float(max(subtotales) / sum_subtotales) if sum_subtotales > 0 else 0.0

        # Concentración de unidades (mayor cantidad de línea / total de unidades)
        cantidades = [item.cantidad for item in c_items]
        concentracion_unidades = float(max(cantidades) / unidades_totales) if unidades_totales > 0 else 0.0

        # 2. Características temporales en zona horaria America/Lima
        dt_local = _to_lima_tz(v.created_at)
        hora_operacion = dt_local.hour
        dia_semana = dt_local.weekday()  # 0: Lunes, 6: Domingo
        es_fin_semana = 1 if dia_semana >= 5 else 0

        # 3. Contexto histórico del cliente (estrictamente anterior a esta venta)
        prior_sales = historial_clientes[v.cliente_id]
        compras_previas = len(prior_sales)

        # Antigüedad del cliente al momento de la venta
        v_created_utc = v.created_at if v.created_at.tzinfo else v.created_at.replace(tzinfo=timezone.utc)
        if cliente and cliente.created_at:
            c_created_utc = cliente.created_at if cliente.created_at.tzinfo else cliente.created_at.replace(tzinfo=timezone.utc)
            antiguedad_cliente_dias = max(0.0, (v_created_utc - c_created_utc).total_seconds() / 86400.0)
        else:
            antiguedad_cliente_dias = 0.0

        if compras_previas == 0:
            # PRIMERA COMPRA DEL CLIENTE (Cold Start)
            # Se conservan semánticamente como NaN las variables sin sustento histórico
            compras_previas_30d = 0
            monto_promedio_previo = np.nan
            monto_maximo_previo = np.nan
            unidades_promedio_previas = np.nan
            dias_desde_compra_anterior = np.nan
            primera_compra = 1
            diversidad_productos_previa = 0
            ratio_monto_vs_promedio_cliente = np.nan
            ratio_unidades_vs_promedio_cliente = np.nan
        else:
            # CLIENTE CON HISTORIAL PREVIO EFECTIVO
            primera_compra = 0

            # Ventana de 30 días anterior a la fecha de la venta
            ventana_30d = v_created_utc - timedelta(days=30)
            compras_previas_30d = sum(1 for p in prior_sales if p.created_at >= ventana_30d)

            # Métricas monetarias previas
            prior_totals = [p.total for p in prior_sales]
            monto_promedio_previo = float(np.mean(prior_totals))
            monto_maximo_previo = float(np.max(prior_totals))

            # Métricas de volumen previo
            prior_units = [p.units for p in prior_sales]
            unidades_promedio_previas = float(np.mean(prior_units))

            # Días desde la venta anterior inmediata
            last_prior = prior_sales[-1]
            dias_desde_compra_anterior = max(0.0, (v_created_utc - last_prior.created_at).total_seconds() / 86400.0)

            # Diversidad de catálogo acumulada previamente
            all_prior_products: Set[int] = set()
            for p in prior_sales:
                all_prior_products.update(p.product_ids)
            diversidad_productos_previa = len(all_prior_products)

            # Ratios relativos contextuales
            ratio_monto_vs_promedio_cliente = (
                float(monto_total / monto_promedio_previo) if monto_promedio_previo > 0 else np.nan
            )
            ratio_unidades_vs_promedio_cliente = (
                float(unidades_totales / unidades_promedio_previas) if unidades_promedio_previas > 0 else np.nan
            )

        rows.append({
            "venta_id": v.id,
            "cliente_id": v.cliente_id,
            "created_at": v.created_at,
            "monto_total": monto_total,
            "unidades_totales": unidades_totales,
            "productos_distintos": productos_distintos,
            "lineas_detalle": lineas_detalle,
            "concentracion_importe": concentracion_importe,
            "concentracion_unidades": concentracion_unidades,
            "hora_operacion": hora_operacion,
            "dia_semana": dia_semana,
            "es_fin_semana": es_fin_semana,
            "compras_previas": compras_previas,
            "compras_previas_30d": compras_previas_30d,
            "monto_promedio_previo": monto_promedio_previo,
            "monto_maximo_previo": monto_maximo_previo,
            "unidades_promedio_previas": unidades_promedio_previas,
            "dias_desde_compra_anterior": dias_desde_compra_anterior,
            "antiguedad_cliente_dias": float(antiguedad_cliente_dias),
            "primera_compra": primera_compra,
            "diversidad_productos_previa": diversidad_productos_previa,
            "ratio_monto_vs_promedio_cliente": ratio_monto_vs_promedio_cliente,
            "ratio_unidades_vs_promedio_cliente": ratio_unidades_vs_promedio_cliente,
        })

        # Agregar la venta actual al acumulador del cliente para las operaciones SUBSECUENTES
        current_product_ids = {item.producto_id for item in c_items}
        historial_clientes[v.cliente_id].append(
            _SaleAccumulatorItem(
                id=v.id,
                created_at=v_created_utc,
                total=monto_total,
                units=unidades_totales,
                product_ids=current_product_ids,
            )
        )

    df = pd.DataFrame(rows, columns=columns_order)
    return df


# =============================================================================
# EXTRACCIÓN DE CARACTERÍSTICAS PARA UNA SOLA VENTA (INFERENCIA EN TIEMPO REAL)
# =============================================================================

def build_sale_features_for_sale(db: Session, venta_id: int) -> pd.DataFrame:
    """
    Construye exactamente el vector de características de UNA sola venta ya persistida.

    Regla Crítica de No Data Leakage:
    - La venta `venta_id` ya existe en la base de datos PostgreSQL, pero al calcular su historial
      previo se excluye explícitamente a sí misma y a cualquier operación posterior:
      (created_at < venta.created_at) OR (created_at == venta.created_at AND id < venta.id).
    - Retorna un DataFrame de 1 fila con exactamente las mismas columnas y tipos que `build_sales_features`.
    """
    venta = db.query(Venta).filter(Venta.id == venta_id).first()
    if not venta:
        raise ValueError(f"Venta con ID {venta_id} no encontrada en la base de datos.")

    cliente = db.query(Cliente).filter(Cliente.id == venta.cliente_id).first()
    c_items = db.query(DetalleVenta).filter(DetalleVenta.venta_id == venta.id).all()

    # 1. Características intrínsecas de la venta
    monto_total = float(venta.total)
    unidades_totales = sum(item.cantidad for item in c_items)
    productos_distintos = len({item.producto_id for item in c_items})
    lineas_detalle = len(c_items)

    subtotales = [float(item.subtotal) for item in c_items]
    sum_subtotales = sum(subtotales)
    concentracion_importe = float(max(subtotales) / sum_subtotales) if sum_subtotales > 0 else 0.0

    cantidades = [item.cantidad for item in c_items]
    concentracion_unidades = float(max(cantidades) / unidades_totales) if unidades_totales > 0 else 0.0

    # 2. Características temporales en hora local America/Lima
    dt_local = _to_lima_tz(venta.created_at)
    hora_operacion = dt_local.hour
    dia_semana = dt_local.weekday()
    es_fin_semana = 1 if dia_semana >= 5 else 0

    # 3. Consulta de ventas completadas estrictamente ANTERIORES del mismo cliente
    v_created = venta.created_at
    prior_sales = (
        db.query(Venta)
        .filter(
            Venta.cliente_id == venta.cliente_id,
            Venta.estado == SALE_STATUS_COMPLETED,
            or_(
                Venta.created_at < v_created,
                and_(Venta.created_at == v_created, Venta.id < venta.id),
            ),
        )
        .order_by(Venta.created_at.asc(), Venta.id.asc())
        .all()
    )

    compras_previas = len(prior_sales)

    # Antigüedad del cliente al momento de la venta
    v_created_utc = v_created if v_created.tzinfo else v_created.replace(tzinfo=timezone.utc)
    if cliente and cliente.created_at:
        c_created_utc = cliente.created_at if cliente.created_at.tzinfo else cliente.created_at.replace(tzinfo=timezone.utc)
        antiguedad_cliente_dias = max(0.0, (v_created_utc - c_created_utc).total_seconds() / 86400.0)
    else:
        antiguedad_cliente_dias = 0.0

    if compras_previas == 0:
        compras_previas_30d = 0
        monto_promedio_previo = np.nan
        monto_maximo_previo = np.nan
        unidades_promedio_previas = np.nan
        dias_desde_compra_anterior = np.nan
        primera_compra = 1
        diversidad_productos_previa = 0
        ratio_monto_vs_promedio_cliente = np.nan
        ratio_unidades_vs_promedio_cliente = np.nan
    else:
        primera_compra = 0
        ventana_30d = v_created_utc - timedelta(days=30)
        compras_previas_30d = sum(
            1 for p in prior_sales
            if (p.created_at if p.created_at.tzinfo else p.created_at.replace(tzinfo=timezone.utc)) >= ventana_30d
        )

        prior_totals = [float(p.total) for p in prior_sales]
        monto_promedio_previo = float(np.mean(prior_totals))
        monto_maximo_previo = float(np.max(prior_totals))

        # Cargar detalles de las ventas previas en una sola consulta
        prior_ids = [p.id for p in prior_sales]
        prior_details = (
            db.query(DetalleVenta)
            .filter(DetalleVenta.venta_id.in_(prior_ids))
            .all()
        )
        units_by_sale: Dict[int, int] = defaultdict(int)
        all_prior_products: Set[int] = set()
        for pd_item in prior_details:
            units_by_sale[pd_item.venta_id] += pd_item.cantidad
            all_prior_products.add(pd_item.producto_id)

        prior_units_list = [units_by_sale[pid] for pid in prior_ids]
        unidades_promedio_previas = float(np.mean(prior_units_list)) if prior_units_list else 0.0

        last_prior = prior_sales[-1]
        last_prior_dt = last_prior.created_at if last_prior.created_at.tzinfo else last_prior.created_at.replace(tzinfo=timezone.utc)
        dias_desde_compra_anterior = max(0.0, (v_created_utc - last_prior_dt).total_seconds() / 86400.0)
        diversidad_productos_previa = len(all_prior_products)

        ratio_monto_vs_promedio_cliente = (
            float(monto_total / monto_promedio_previo) if monto_promedio_previo > 0 else np.nan
        )
        ratio_unidades_vs_promedio_cliente = (
            float(unidades_totales / unidades_promedio_previas) if unidades_promedio_previas > 0 else np.nan
        )

    row = {
        "venta_id": venta.id,
        "cliente_id": venta.cliente_id,
        "created_at": venta.created_at,
        "monto_total": monto_total,
        "unidades_totales": unidades_totales,
        "productos_distintos": productos_distintos,
        "lineas_detalle": lineas_detalle,
        "concentracion_importe": concentracion_importe,
        "concentracion_unidades": concentracion_unidades,
        "hora_operacion": hora_operacion,
        "dia_semana": dia_semana,
        "es_fin_semana": es_fin_semana,
        "compras_previas": compras_previas,
        "compras_previas_30d": compras_previas_30d,
        "monto_promedio_previo": monto_promedio_previo,
        "monto_maximo_previo": monto_maximo_previo,
        "unidades_promedio_previas": unidades_promedio_previas,
        "dias_desde_compra_anterior": dias_desde_compra_anterior,
        "antiguedad_cliente_dias": float(antiguedad_cliente_dias),
        "primera_compra": primera_compra,
        "diversidad_productos_previa": diversidad_productos_previa,
        "ratio_monto_vs_promedio_cliente": ratio_monto_vs_promedio_cliente,
        "ratio_unidades_vs_promedio_cliente": ratio_unidades_vs_promedio_cliente,
    }

    columns_order = ["venta_id", "cliente_id", "created_at"] + ISOLATION_FOREST_FEATURE_COLUMNS
    return pd.DataFrame([row], columns=columns_order)


def build_customer_features_before_sale(
    db: Session,
    venta_id: int,
) -> pd.DataFrame:
    """
    Construye las características comerciales históricas del CLIENTE asociado a una venta,
    evaluando estricta y exclusivamente las operaciones COMPLETADAS registradas antes de dicha venta:
    (created_at < venta.created_at) OR (created_at == venta.created_at AND id < venta.id).

    Regla Crítica Anti-Data-Leakage:
    - La venta actual analizada y cualquier venta posterior quedan TOTALMENTE EXCLUIDAS del cálculo.
    - Se utiliza para contextualizar la operación con el perfil K-Means sin contaminar el historial.
    """
    venta = db.query(Venta).filter(Venta.id == venta_id).first()
    if not venta:
        raise ValueError(f"No se encontró la venta con ID {venta_id}")

    cliente = db.query(Cliente).filter(Cliente.id == venta.cliente_id).first()
    if not cliente:
        raise ValueError(f"No se encontró el cliente asociado a la venta con ID {venta_id}")

    v_created_utc = venta.created_at if venta.created_at.tzinfo else venta.created_at.replace(tzinfo=timezone.utc)
    c_created_utc = cliente.created_at if cliente.created_at.tzinfo else cliente.created_at.replace(tzinfo=timezone.utc)
    antiguedad_cliente_dias = max(0.0, (v_created_utc - c_created_utc).total_seconds() / 86400.0)

    # Filtrar ventas del cliente anteriores a la venta objetivo
    prior_sales = (
        db.query(Venta)
        .filter(
            Venta.cliente_id == cliente.id,
            Venta.estado == SALE_STATUS_COMPLETED,
            or_(
                Venta.created_at < v_created_utc,
                and_(Venta.created_at == v_created_utc, Venta.id < venta.id),
            ),
        )
        .order_by(Venta.created_at.asc(), Venta.id.asc())
        .all()
    )

    total_compras = len(prior_sales)

    if total_compras == 0:
        row = {
            "cliente_id": cliente.id,
            "total_compras": 0,
            "monto_promedio": np.nan,
            "monto_total_acumulado": 0.0,
            "unidades_promedio": np.nan,
            "productos_distintos_historicos": 0,
            "frecuencia_30d": 0,
            "recencia_dias": np.nan,
            "antiguedad_cliente_dias": float(antiguedad_cliente_dias),
            "dias_entre_compras_promedio": np.nan,
        }
    else:
        montos = [float(p.total) for p in prior_sales]
        monto_total_acumulado = float(sum(montos))
        monto_promedio = float(monto_total_acumulado / total_compras)

        prior_ids = [p.id for p in prior_sales]
        prior_details = (
            db.query(DetalleVenta)
            .filter(DetalleVenta.venta_id.in_(prior_ids))
            .all()
        )
        detalles_por_venta: Dict[int, List[DetalleVenta]] = defaultdict(list)
        for d in prior_details:
            detalles_por_venta[d.venta_id].append(d)

        unidades_por_venta: List[int] = []
        productos_set: Set[int] = set()
        for p in prior_sales:
            items = detalles_por_venta.get(p.id, [])
            unidades_por_venta.append(sum(item.cantidad for item in items))
            for item in items:
                productos_set.add(item.producto_id)

        unidades_promedio = float(sum(unidades_por_venta) / total_compras) if total_compras > 0 else 0.0
        productos_distintos_historicos = len(productos_set)

        ventana_30d = v_created_utc - timedelta(days=30)
        frecuencia_30d = sum(
            1 for p in prior_sales
            if (p.created_at if p.created_at.tzinfo else p.created_at.replace(tzinfo=timezone.utc)) >= ventana_30d
        )

        last_prior = prior_sales[-1]
        last_prior_dt = last_prior.created_at if last_prior.created_at.tzinfo else last_prior.created_at.replace(tzinfo=timezone.utc)
        recencia_dias = max(0.0, (v_created_utc - last_prior_dt).total_seconds() / 86400.0)

        if total_compras >= 2:
            first_prior = prior_sales[0]
            first_prior_dt = first_prior.created_at if first_prior.created_at.tzinfo else first_prior.created_at.replace(tzinfo=timezone.utc)
            total_intervalo_dias = max(0.0, (last_prior_dt - first_prior_dt).total_seconds() / 86400.0)
            dias_entre_compras_promedio = float(total_intervalo_dias / (total_compras - 1))
        else:
            dias_entre_compras_promedio = np.nan

        row = {
            "cliente_id": cliente.id,
            "total_compras": total_compras,
            "monto_promedio": monto_promedio,
            "monto_total_acumulado": monto_total_acumulado,
            "unidades_promedio": unidades_promedio,
            "productos_distintos_historicos": productos_distintos_historicos,
            "frecuencia_30d": frecuencia_30d,
            "recencia_dias": float(recencia_dias),
            "antiguedad_cliente_dias": float(antiguedad_cliente_dias),
            "dias_entre_compras_promedio": dias_entre_compras_promedio,
        }

    cols = ["cliente_id"] + KMEANS_FEATURE_COLUMNS
    return pd.DataFrame([row], columns=cols)


# Alias compatibles para importaciones genéricas
extract_client_features = build_customer_features
extract_sale_features = build_sales_features

