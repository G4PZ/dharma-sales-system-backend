from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo
from fastapi import HTTPException, status
from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from app.models.client import Cliente
from app.models.product import Producto
from app.models.sale import DetalleVenta, Venta
from app.schemas.sale import SaleCreate

TWO_PLACES = Decimal("0.01")
IGV_RATE = Decimal("0.18")
LIMA_TZ = ZoneInfo("America/Lima")



def _generate_numero_venta(db: Session) -> str:
    """
    Genera el siguiente número correlativo seguro para ventas utilizando la secuencia
    nativa de PostgreSQL venta_numero_seq. Garantiza unicidad e inmunidad ante concurrencia.
    """
    seq_val = db.execute(text("SELECT nextval('venta_numero_seq')")).scalar()
    return f"V-{seq_val:06d}"


def _format_sale_response(venta: Venta) -> dict:
    """
    Construye la estructura compatible con SaleResponse incluyendo datos
    del cliente y los datos históricos y descriptivos de cada ítem de detalle.
    """
    detalles_formateados = []
    for d in venta.detalles:
        detalles_formateados.append({
            "id": d.id,
            "producto_id": d.producto_id,
            "codigo_producto": d.producto.codigo if d.producto else "N/A",
            "nombre_producto": d.producto.nombre if d.producto else "Producto no disponible",
            "unidad_medida": d.producto.unidad_medida if d.producto else "und",
            "cantidad": d.cantidad,
            "precio_unitario": d.precio_unitario,
            "subtotal": d.subtotal,
        })

    return {
        "id": venta.id,
        "numero_venta": venta.numero_venta,
        "cliente_id": venta.cliente_id,
        "cliente": {
            "id": venta.cliente.id,
            "numero_documento": venta.cliente.numero_documento,
            "razon_social": venta.cliente.razon_social,
            "nombre_contacto": venta.cliente.nombre_contacto,
            "telefono": venta.cliente.telefono,
            "email": venta.cliente.email,
            "direccion": venta.cliente.direccion,
            "ciudad": venta.cliente.ciudad,
        },
        "detalles": detalles_formateados,
        "subtotal": venta.subtotal,
        "igv": venta.igv,
        "total": venta.total,
        "estado": venta.estado,
        "created_at": venta.created_at,
        "updated_at": venta.updated_at,
    }


def create_sale(db: Session, sale_in: SaleCreate) -> dict:
    """
    Registra y procesa una venta completa como una transacción atómica:
    1. Verifica existencia y estado activo del cliente.
    2. Consolida cantidades por producto y valida enteros positivos.
    3. Bloquea pesimistamente las filas de los productos con SELECT ... FOR UPDATE.
    4. Verifica existencia, estado activo y suficiencia de stock.
    5. Aplica cálculos monetarios exactos con Decimal (subtotales, IGV 18% y total).
    6. Genera numero_venta correlativo único con venta_numero_seq.
    7. Persiste Venta y DetalleVenta.
    8. Descuenta el stock de cada producto.
    """
    try:
        # 1. Verificar existencia y estado del cliente
        cliente = db.query(Cliente).filter(Cliente.id == sale_in.cliente_id).first()
        if not cliente:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cliente con ID {sale_in.cliente_id} no encontrado."
            )
        if not cliente.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"El cliente '{cliente.razon_social}' (RUC: {cliente.numero_documento}) se encuentra inactivo y no puede realizar compras."
            )

        # 2. Validar que existan ítems y consolidar productos duplicados
        if not sale_in.productos:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La venta debe incluir al menos un producto."
            )

        cantidades_por_producto: Dict[int, int] = defaultdict(int)
        for item in sale_in.productos:
            if item.cantidad <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"La cantidad para el producto ID {item.producto_id} debe ser un entero mayor a cero."
                )
            cantidades_por_producto[item.producto_id] += item.cantidad

        producto_ids = sorted(list(cantidades_por_producto.keys()))

        # 3. Bloqueo pesimista de filas en orden ascendente para evitar race conditions y deadlocks
        productos_db = (
            db.query(Producto)
            .filter(Producto.id.in_(producto_ids))
            .order_by(Producto.id.asc())
            .with_for_update()
            .all()
        )

        productos_map = {p.id: p for p in productos_db}

        # 4. Verificar que todos los productos existan
        for pid in producto_ids:
            if pid not in productos_map:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Producto con ID {pid} no encontrado en el catálogo."
                )

        # 5. Verificar que todos los productos estén activos
        for pid, prod in productos_map.items():
            if not prod.is_active:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"El producto '{prod.nombre}' (Código: {prod.codigo}) se encuentra inactivo."
                )

        # 6. Verificar stock suficiente para cada producto antes de cualquier mutación
        for pid, cant_solicitada in cantidades_por_producto.items():
            prod = productos_map[pid]
            if prod.stock < cant_solicitada:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Stock insuficiente para el producto '{prod.nombre}' (Código: {prod.codigo}). "
                        f"Disponible: {prod.stock}, solicitado: {cant_solicitada}."
                    )
                )

        # 7. Cálculos monetarios exactos con Decimal y cuantización ROUND_HALF_UP a 2 decimales
        detalles_data = []
        subtotal_acumulado = Decimal("0.00")

        for pid, cant in cantidades_por_producto.items():
            prod = productos_map[pid]
            precio_unitario = Decimal(str(prod.precio)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
            subtotal_linea = (precio_unitario * Decimal(cant)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
            subtotal_acumulado += subtotal_linea

            detalles_data.append({
                "producto": prod,
                "cantidad": cant,
                "precio_unitario": precio_unitario,
                "subtotal": subtotal_linea,
            })

        subtotal = subtotal_acumulado.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
        igv = (subtotal * IGV_RATE).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
        total = (subtotal + igv).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

        # 8. Generar número de venta correlativo único
        numero_venta = _generate_numero_venta(db)

        # 9. Crear cabecera de la venta
        nueva_venta = Venta(
            numero_venta=numero_venta,
            cliente_id=cliente.id,
            subtotal=subtotal,
            igv=igv,
            total=total,
            estado="COMPLETADA",
        )
        db.add(nueva_venta)
        db.flush()  # Genera nueva_venta.id dentro de la transacción activa

        # 10. Crear detalles de la venta y descontar stock
        for item in detalles_data:
            prod = item["producto"]
            cant = item["cantidad"]

            detalle = DetalleVenta(
                venta_id=nueva_venta.id,
                producto_id=prod.id,
                cantidad=cant,
                precio_unitario=item["precio_unitario"],
                subtotal=item["subtotal"],
            )
            db.add(detalle)

            # Descontar stock atómicamente
            prod.stock -= cant

        # 11. Confirmar toda la transacción atómica
        db.commit()
        db.refresh(nueva_venta)

        return _format_sale_response(nueva_venta)

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado al procesar la venta: {str(e)}"
        )


def get_sales(
    db: Session,
    search: Optional[str] = None,
    estado: Optional[str] = None,
    fecha_desde: Optional[date] = None,
    fecha_hasta: Optional[date] = None,
    skip: int = 0,
    limit: int = 50,
) -> dict:
    """
    Listar ventas con filtros por búsqueda (número de venta, RUC o razón social del cliente),
    filtro por estado, rango de fechas y paginación.
    """
    query = db.query(Venta).join(Venta.cliente)

    # Filtro de búsqueda textual
    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Venta.numero_venta.ilike(term),
                Cliente.numero_documento.ilike(term),
                Cliente.razon_social.ilike(term),
            )
        )

    # Filtro de estado
    if estado and estado.strip().lower() not in ("todos", ""):
        query = query.filter(Venta.estado.ilike(estado.strip()))

    # Filtro por rango de fechas (días calendario en zona horaria America/Lima convertidos a UTC)
    if fecha_desde:
        inicio_local = datetime.combine(fecha_desde, time.min, tzinfo=LIMA_TZ)
        inicio_utc = inicio_local.astimezone(timezone.utc)
        query = query.filter(Venta.created_at >= inicio_utc)

    if fecha_hasta:
        siguiente_dia_local = datetime.combine(fecha_hasta + timedelta(days=1), time.min, tzinfo=LIMA_TZ)
        siguiente_dia_utc = siguiente_dia_local.astimezone(timezone.utc)
        query = query.filter(Venta.created_at < siguiente_dia_utc)


    total = query.count()
    ventas = query.order_by(Venta.created_at.desc(), Venta.id.desc()).offset(skip).limit(limit).all()

    items = []
    for v in ventas:
        items.append({
            "id": v.id,
            "numero_venta": v.numero_venta,
            "cliente": {
                "id": v.cliente.id,
                "numero_documento": v.cliente.numero_documento,
                "razon_social": v.cliente.razon_social,
            },
            "fecha": v.created_at,
            "cantidad_items": len(v.detalles),
            "subtotal": v.subtotal,
            "igv": v.igv,
            "total": v.total,
            "estado": v.estado,
        })

    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


def get_sale_by_id(db: Session, venta_id: int) -> Optional[dict]:
    """
    Obtiene el detalle completo de una venta por su ID.
    """
    venta = db.query(Venta).filter(Venta.id == venta_id).first()
    if not venta:
        return None
    return _format_sale_response(venta)


def get_sale_stats(db: Session) -> dict:
    """
    Calcula las estadísticas del módulo Ventas correspondientes exclusivamente
    al mes calendario actual en hora de Perú (America/Lima):
    - ventas_mes: cantidad de ventas con estado COMPLETADA en el mes actual.
    - monto_mes: suma del total de ventas completadas en el mes actual.
    - producto_mas_vendido: producto con mayor suma de DetalleVenta.cantidad en ventas completadas del mes actual.
    - unidades_mes: suma de DetalleVenta.cantidad de todas las ventas completadas del mes actual.
    """
    now_lima = datetime.now(LIMA_TZ)
    inicio_mes_local = datetime(now_lima.year, now_lima.month, 1, 0, 0, 0, tzinfo=LIMA_TZ)
    inicio_mes_utc = inicio_mes_local.astimezone(timezone.utc)

    if now_lima.month == 12:
        siguiente_mes_local = datetime(now_lima.year + 1, 1, 1, 0, 0, 0, tzinfo=LIMA_TZ)
    else:
        siguiente_mes_local = datetime(now_lima.year, now_lima.month + 1, 1, 0, 0, 0, tzinfo=LIMA_TZ)
    siguiente_mes_utc = siguiente_mes_local.astimezone(timezone.utc)

    # Filtro base para ventas completadas del mes calendario actual
    filtro_mes = (
        Venta.estado == "COMPLETADA",
        Venta.created_at >= inicio_mes_utc,
        Venta.created_at < siguiente_mes_utc,
    )

    # 1. Ventas del mes
    ventas_mes = db.query(func.count(Venta.id)).filter(*filtro_mes).scalar() or 0

    # 2. Monto vendido del mes
    monto_val = (
        db.query(func.coalesce(func.sum(Venta.total), 0))
        .filter(*filtro_mes)
        .scalar()
    )
    monto_mes = Decimal(str(monto_val)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

    # 3. Unidades vendidas del mes
    unidades_val = (
        db.query(func.coalesce(func.sum(DetalleVenta.cantidad), 0))
        .join(Venta, DetalleVenta.venta_id == Venta.id)
        .filter(*filtro_mes)
        .scalar()
    )
    unidades_mes = int(unidades_val or 0)

    # 4. Producto más vendido del mes (con criterio de desempate determinístico Producto.id.asc())
    producto_top = (
        db.query(
            Producto.nombre,
            func.sum(DetalleVenta.cantidad).label("total_unidades")
        )
        .join(DetalleVenta, DetalleVenta.producto_id == Producto.id)
        .join(Venta, DetalleVenta.venta_id == Venta.id)
        .filter(*filtro_mes)
        .group_by(Producto.id, Producto.nombre)
        .order_by(func.sum(DetalleVenta.cantidad).desc(), Producto.id.asc())
        .first()
    )

    if producto_top and producto_top[0] is not None:
        producto_mas_vendido = {
            "nombre": producto_top[0],
            "unidades": int(producto_top[1] or 0),
        }
    else:
        producto_mas_vendido = {
            "nombre": None,
            "unidades": 0,
        }

    return {
        "ventas_mes": ventas_mes,
        "monto_mes": monto_mes,
        "producto_mas_vendido": producto_mas_vendido,
        "unidades_mes": unidades_mes,
    }

