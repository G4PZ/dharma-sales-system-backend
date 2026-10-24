from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
from fastapi import HTTPException, status
from sqlalchemy import String, func, or_, select, text
from sqlalchemy.orm import Session

from app.models.client import Cliente
from app.models.receipt import Comprobante, RespuestaSunat
from app.models.sale import Venta
from app.models.summary import ResumenDiario, ResumenDiarioComprobante
from app.schemas.receipt import ReceiptCreate
from app.services.sunat_client import (
    SunatAuthError,
    SunatClientError,
    SunatConnectionError,
    SunatPayloadError,
    sunat_client,
)

LIMA_TZ = ZoneInfo("America/Lima")
MAX_RESUMEN_BOLETAS = 500

# Constantes de negocio para tipos de comprobante y series de Dharma
TIPO_FACTURA = "01"
TIPO_BOLETA = "03"

SERIE_POR_TIPO = {
    TIPO_FACTURA: "F001",
    TIPO_BOLETA: "B001",
}

SECUENCIA_POR_TIPO = {
    TIPO_FACTURA: "factura_correlativo_seq",
    TIPO_BOLETA: "boleta_correlativo_seq",
}

NOMBRE_TIPO = {
    TIPO_FACTURA: "Factura",
    TIPO_BOLETA: "Boleta",
}


def create_receipt(db: Session, receipt_in: ReceiptCreate) -> Comprobante:
    """
    Emite un comprobante electrónico para una venta de forma transaccional y atómica:
    1. Valida el tipo de comprobante ('01' o '03').
    2. Bloquea pesimistamente la venta con SELECT ... FOR UPDATE.
    3. Valida existencia y estado COMPLETADA de la venta.
    4. Valida que la venta no cuente con un comprobante previo (regla 1 venta = máx 1 comprobante).
    5. Asigna la serie fija (F001 o B001) y obtiene el correlativo atómico desde la secuencia de PostgreSQL.
    6. Persiste el comprobante en estado 'PENDIENTE_ENVIO'.
    """
    tipo = receipt_in.tipo_comprobante
    if tipo not in SERIE_POR_TIPO:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Tipo de comprobante '{tipo}' inválido. Solo se permite '01' (Factura) o '03' (Boleta).",
        )

    try:
        # 1. Bloqueo pesimista de la fila de venta (OF ventas para joins)
        venta = (
            db.query(Venta)
            .filter(Venta.id == receipt_in.venta_id)
            .with_for_update(of=Venta)
            .first()
        )

        if not venta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Venta con ID {receipt_in.venta_id} no encontrada.",
            )

        if venta.estado != "COMPLETADA":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Solo se pueden emitir comprobantes para ventas en estado 'COMPLETADA'. Estado actual: '{venta.estado}'.",
            )

        # 2. Verificar que no tenga comprobante previo mediante relación 1:1
        existing_receipt = (
            db.query(Comprobante)
            .filter(Comprobante.venta_id == venta.id)
            .first()
        )
        if existing_receipt:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"La venta '{venta.numero_venta}' ya cuenta con un comprobante emitido ({existing_receipt.numero_comprobante}).",
            )

        # 3. Asignar serie y extraer correlativo desde secuencia PostgreSQL independiente
        serie = SERIE_POR_TIPO[tipo]
        seq_name = SECUENCIA_POR_TIPO[tipo]
        correlativo = db.execute(text(f"SELECT nextval('{seq_name}')")).scalar()

        # 4. Crear registro de comprobante
        comprobante = Comprobante(
            venta_id=venta.id,
            tipo_comprobante=tipo,
            serie=serie,
            correlativo=correlativo,
            estado="PENDIENTE_ENVIO",
        )
        db.add(comprobante)
        db.commit()
        db.refresh(comprobante)
        return comprobante

    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado al generar el comprobante: {str(exc)}",
        ) from exc


def get_available_sales(
    db: Session,
    search: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
) -> Dict:
    """
    Obtiene las ventas completadas que aún no tienen comprobante emitido,
    utilizando LEFT OUTER JOIN con Comprobante y filtrando comprobantes.id IS NULL.
    """
    query = (
        db.query(Venta)
        .outerjoin(Comprobante, Venta.id == Comprobante.venta_id)
        .join(Venta.cliente)
        .filter(Venta.estado == "COMPLETADA")
        .filter(Comprobante.id.is_(None))
    )

    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Venta.numero_venta.ilike(term),
                Cliente.numero_documento.ilike(term),
                Cliente.razon_social.ilike(term),
            )
        )

    total = query.count()
    items = (
        query.order_by(Venta.created_at.desc(), Venta.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    formatted_items = [
        {
            "id": v.id,
            "numero_venta": v.numero_venta,
            "cliente_id": v.cliente_id,
            "cliente_numero_documento": v.cliente.numero_documento,
            "cliente_razon_social": v.cliente.razon_social,
            "subtotal": v.subtotal,
            "igv": v.igv,
            "total": v.total,
            "created_at": v.created_at,
        }
        for v in items
    ]

    return {
        "items": formatted_items,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


def get_receipts(
    db: Session,
    search: Optional[str] = None,
    tipo: Optional[str] = None,
    estado: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
) -> Dict:
    """
    Listado paginado de comprobantes con filtros por tipo, estado y búsqueda textual.
    """
    query = (
        db.query(Comprobante)
        .join(Comprobante.venta)
        .join(Venta.cliente)
    )

    if tipo and tipo.strip() not in ("todos", "todas", ""):
        query = query.filter(Comprobante.tipo_comprobante == tipo.strip())

    if estado and estado.strip() not in ("todos", "todas", ""):
        query = query.filter(Comprobante.estado == estado.strip())

    if search and search.strip():
        term = f"%{search.strip()}%"
        # Permite buscar por serie, correlativo compuesto F001-00000001, N° venta, RUC o razón social
        query = query.filter(
            or_(
                Comprobante.serie.ilike(term),
                func.concat(
                    Comprobante.serie,
                    "-",
                    func.lpad(func.cast(Comprobante.correlativo, String), 8, "0"),
                ).ilike(term),
                Venta.numero_venta.ilike(term),
                Cliente.numero_documento.ilike(term),
                Cliente.razon_social.ilike(term),
            )
        )

    total = query.count()
    items = (
        query.order_by(Comprobante.fecha_emision.desc(), Comprobante.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    formatted_items = [
        {
            "id": c.id,
            "venta_id": c.venta_id,
            "numero_venta": c.venta.numero_venta,
            "cliente_numero_documento": c.venta.cliente.numero_documento,
            "cliente_razon_social": c.venta.cliente.razon_social,
            "tipo_comprobante": c.tipo_comprobante,
            "nombre_tipo": NOMBRE_TIPO.get(c.tipo_comprobante, "Comprobante"),
            "serie": c.serie,
            "correlativo": c.correlativo,
            "numero_comprobante": c.numero_comprobante,
            "total": c.venta.total,
            "fecha_emision": c.fecha_emision,
            "estado": c.estado,
            "created_at": c.created_at,
        }
        for c in items
    ]

    return {
        "items": formatted_items,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


def get_receipt_by_id(db: Session, receipt_id: int) -> Dict:
    """
    Obtiene la información integral de un comprobante:
    datos de emisión, datos de la venta, cliente y desglose de ítems históricos.
    """
    c = db.query(Comprobante).filter(Comprobante.id == receipt_id).first()
    if not c:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comprobante con ID {receipt_id} no encontrado.",
        )

    venta = c.venta
    cliente = venta.cliente

    detalles_formateados = []
    for d in venta.detalles:
        detalles_formateados.append({
            "id": d.id,
            "producto_id": d.producto_id,
            "codigo_producto": d.producto.codigo if d.producto else "N/A",
            "nombre_producto": d.producto.nombre if d.producto else "Producto",
            "unidad_medida": d.producto.unidad_medida if d.producto else "und",
            "cantidad": d.cantidad,
            "precio_unitario": d.precio_unitario,
            "subtotal": d.subtotal,
        })

    respuestas_formateadas = [
        {
            "id": r.id,
            "codigo": r.codigo,
            "estado": r.estado,
            "descripcion": r.descripcion,
            "observaciones": r.observaciones,
            "ticket": r.ticket,
            "fecha_respuesta": r.fecha_respuesta,
            "created_at": r.created_at,
        }
        for r in c.respuestas_sunat
    ]

    receipt_data = {
        "id": c.id,
        "venta_id": c.venta_id,
        "tipo_comprobante": c.tipo_comprobante,
        "nombre_tipo": NOMBRE_TIPO.get(c.tipo_comprobante, "Comprobante"),
        "serie": c.serie,
        "correlativo": c.correlativo,
        "numero_comprobante": c.numero_comprobante,
        "fecha_emision": c.fecha_emision,
        "estado": c.estado,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
        "venta": {
            "id": venta.id,
            "numero_venta": venta.numero_venta,
            "subtotal": venta.subtotal,
            "igv": venta.igv,
            "total": venta.total,
            "created_at": venta.created_at,
        },
        "cliente": {
            "id": cliente.id,
            "numero_documento": cliente.numero_documento,
            "razon_social": cliente.razon_social,
            "direccion": cliente.direccion,
            "email": cliente.email,
            "telefono": cliente.telefono,
        },
        "detalles": detalles_formateados,
        "respuestas_sunat": respuestas_formateadas,
        "resumen_diario": None,
    }

    if c.tipo_comprobante == TIPO_BOLETA and c.resumenes_diarios:
        ultimo_resumen = sorted(c.resumenes_diarios, key=lambda r: r.id, reverse=True)[0]
        receipt_data["resumen_diario"] = {
            "id": ultimo_resumen.id,
            "identificador": ultimo_resumen.identificador,
            "fecha_resumen": ultimo_resumen.fecha_resumen,
            "estado": ultimo_resumen.estado,
            "ticket": ultimo_resumen.ticket,
            "codigo_respuesta": ultimo_resumen.codigo_respuesta,
            "descripcion_respuesta": ultimo_resumen.descripcion_respuesta,
            "observaciones": ultimo_resumen.observaciones,
            "fecha_envio": ultimo_resumen.fecha_envio,
            "fecha_respuesta": ultimo_resumen.fecha_respuesta,
        }

    return receipt_data


def get_receipt_stats(db: Session) -> Dict:
    """
    Retorna métricas operativas del módulo de comprobantes.
    """
    total_comprobantes = db.query(Comprobante).count()
    facturas_count = db.query(Comprobante).filter(Comprobante.tipo_comprobante == TIPO_FACTURA).count()
    boletas_count = db.query(Comprobante).filter(Comprobante.tipo_comprobante == TIPO_BOLETA).count()
    pendientes_envio = db.query(Comprobante).filter(Comprobante.estado == "PENDIENTE_ENVIO").count()

    return {
        "total_comprobantes": total_comprobantes,
        "facturas_count": facturas_count,
        "boletas_count": boletas_count,
        "pendientes_envio": pendientes_envio,
    }


# =============================================================================
# INTEGRACIÓN FASE 4: EMISIÓN Y FLUJO DE FACTURAS ('01')
# =============================================================================

def enviar_factura_sunat(db: Session, comprobante_id: int) -> Dict[str, Any]:
    """
    Despacha una Factura electrónica hacia el microservicio PHP/Greenter.
    Estrategia de concurrencia e idempotencia en 3 pasos:
    1. Transacción rápida DB: Bloquea con SELECT FOR UPDATE, valida estado, construye payload histórico y marca ENVIADO. Commit.
    2. Llamada HTTP fuera de transacción PostgreSQL.
    3. Manejo de resultado: si modo local -> PENDIENTE_ENVIO. Si SUNAT -> ACEPTADO/RECHAZADO con auditoría en respuesta_sunat.
    """
    # 1. Transacción rápida DB con bloqueo pesimista
    try:
        c = (
            db.query(Comprobante)
            .filter(Comprobante.id == comprobante_id)
            .with_for_update(of=Comprobante)
            .first()
        )
        if not c:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Comprobante con ID {comprobante_id} no encontrado.",
            )

        if c.tipo_comprobante != TIPO_FACTURA:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Solo se pueden enviar Facturas ('01') individualmente. Las Boletas ('03') deben comunicarse mediante Resumen Diario.",
            )

        if c.estado == "ACEPTADO":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"La factura {c.numero_comprobante} ya fue ACEPTADA por SUNAT.",
            )

        if c.estado == "ENVIADO":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"La factura {c.numero_comprobante} se encuentra actualmente en proceso de envío.",
            )

        venta = c.venta
        cliente = venta.cliente

        # Construir payload histórico (Precios inmutables de detalle_ventas)
        items_payload = []
        for d in venta.detalles:
            items_payload.append({
                "codigo": d.producto.codigo if d.producto else f"PROD-{d.producto_id}",
                "descripcion": d.producto.nombre if d.producto else "Producto",
                "unidad": "NIU",  # SUNAT Catálogo 03: UNIDAD (BIENES) para todos los productos de Dharma
                "cantidad": d.cantidad,
                "valor_unitario": str(d.precio_unitario),  # PRECIO HISTÓRICO SIN IGV
                "subtotal": str(d.subtotal),
            })

        payload = {
            "comprobante": {
                "id": c.id,
                "tipo": "01",
                "serie": c.serie,
                "correlativo": c.correlativo,
                "fecha_emision": c.fecha_emision.isoformat(),
                "moneda": "PEN",
            },
            "cliente": {
                "tipo_documento": "6",  # RUC exclusivo en Dharma
                "numero_documento": cliente.numero_documento,
                "razon_social": cliente.razon_social,
            },
            "totales": {
                "subtotal": str(venta.subtotal),
                "igv": str(venta.igv),
                "total": str(venta.total),
            },
            "items": items_payload,
        }

        # Transición transitoria antes de la llamada de red
        c.estado = "ENVIADO"
        db.commit()

    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error preparando el envío de la factura: {str(exc)}",
        ) from exc

    # 2. Llamada HTTP a PHP fuera de la transacción de BD
    try:
        result = sunat_client.emitir_factura(payload)
    except SunatConnectionError as conn_err:
        c.estado = "ERROR_ENVIO"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ERROR_ENVIO: {conn_err.message}",
        ) from conn_err
    except SunatAuthError as auth_err:
        c.estado = "ERROR_ENVIO"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CONFIGURATION_ERROR: {auth_err.message}",
        ) from auth_err
    except SunatPayloadError as pay_err:
        c.estado = "ERROR_ENVIO"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{pay_err.code}: {pay_err.message}",
        ) from pay_err
    except Exception as exc:
        c.estado = "ERROR_ENVIO"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado en comunicación con el microservicio: {str(exc)}",
        ) from exc

    # 3. Procesar respuesta técnica
    if "enviado_sunat" not in result:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Respuesta inválida del microservicio SUNAT: falta la propiedad requerida 'enviado_sunat'.",
        )

    if not isinstance(result["enviado_sunat"], bool):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Respuesta inválida del microservicio SUNAT: 'enviado_sunat' debe ser booleano.",
        )

    enviado_sunat = result["enviado_sunat"]

    # Modo Local (SUNAT_SEND_ENABLED=false)
    if not enviado_sunat:
        c.estado = "PENDIENTE_ENVIO"
        db.commit()
        return {
            "success": True,
            "comprobante_id": c.id,
            "numero_comprobante": c.numero_comprobante,
            "estado": "PENDIENTE_ENVIO",
            "enviado_sunat": False,
            "xml_generado": result.get("xml_generado", True),
            "xml_firmado": result.get("xml_firmado", True),
            "xml_file": result.get("xml_file"),
            "mensaje": "XML generado y firmado correctamente en el microservicio. El envío a SUNAT se encuentra deshabilitado (modo local).",
        }

    # Modo SUNAT Real (SUNAT_SEND_ENABLED=true)
    estado_sunat = result.get("estado")
    if estado_sunat == "ACEPTADO":
        c.estado = "ACEPTADO"
        respuesta = RespuestaSunat(
            comprobante_id=c.id,
            codigo=result.get("codigo"),
            estado="ACEPTADO",
            descripcion=result.get("descripcion"),
            observaciones=result.get("observaciones"),
            ticket=None,
            fecha_respuesta=func.now(),
        )
        db.add(respuesta)
        db.commit()
        return {
            "success": True,
            "comprobante_id": c.id,
            "numero_comprobante": c.numero_comprobante,
            "estado": "ACEPTADO",
            "enviado_sunat": True,
            "codigo": result.get("codigo"),
            "descripcion": result.get("descripcion"),
            "observaciones": result.get("observaciones"),
            "xml_file": result.get("xml_file"),
            "cdr_file": result.get("cdr_file"),
            "mensaje": "Factura aceptada por SUNAT.",
        }
    else:
        c.estado = "RECHAZADO"
        respuesta = RespuestaSunat(
            comprobante_id=c.id,
            codigo=result.get("codigo"),
            estado="RECHAZADO",
            descripcion=result.get("descripcion"),
            observaciones=result.get("observaciones"),
            ticket=None,
            fecha_respuesta=func.now(),
        )
        db.add(respuesta)
        db.commit()
        return {
            "success": False,
            "comprobante_id": c.id,
            "numero_comprobante": c.numero_comprobante,
            "estado": "RECHAZADO",
            "enviado_sunat": True,
            "codigo": result.get("codigo"),
            "descripcion": result.get("descripcion"),
            "mensaje": "Factura rechazada por SUNAT.",
        }


# =============================================================================
# INTEGRACIÓN FASE 4: RESÚMENES DIARIOS DE BOLETAS ('03')
# =============================================================================

def get_boletas_pendientes_resumen(db: Session) -> Dict[str, Any]:
    """
    Obtiene las boletas 'PENDIENTE_ENVIO' agrupadas por fecha de emisión en America/Lima,
    excluyendo aquellas que ya pertenecen a un resumen en estado GENERADO, ENVIADO, EN_PROCESO o ACEPTADO.
    """
    # Subconsulta de comprobantes ya asociados a resúmenes activos
    subq_asociadas = (
        select(ResumenDiarioComprobante.comprobante_id)
        .join(ResumenDiario, ResumenDiarioComprobante.resumen_id == ResumenDiario.id)
        .where(ResumenDiario.estado.in_(["GENERADO", "ENVIADO", "EN_PROCESO", "ACEPTADO"]))
    )

    fecha_lima_col = func.date(func.timezone("America/Lima", Comprobante.fecha_emision))

    boletas = (
        db.query(Comprobante)
        .join(Comprobante.venta)
        .join(Venta.cliente)
        .filter(Comprobante.tipo_comprobante == TIPO_BOLETA)
        .filter(Comprobante.estado == "PENDIENTE_ENVIO")
        .filter(~Comprobante.id.in_(subq_asociadas))
        .order_by(fecha_lima_col.desc(), Comprobante.correlativo.asc())
        .all()
    )

    # Agrupar en memoria por fecha local (America/Lima)
    grupos_dict: Dict[date, List[Dict[str, Any]]] = {}
    for b in boletas:
        # Calcular fecha en America/Lima
        fecha_local = b.fecha_emision.astimezone(LIMA_TZ).date()
        if fecha_local not in grupos_dict:
            grupos_dict[fecha_local] = []

        grupos_dict[fecha_local].append({
            "id": b.id,
            "comprobante_id": b.id,
            "numero_comprobante": b.numero_comprobante,
            "tipo_comprobante": b.tipo_comprobante,
            "serie": b.serie,
            "correlativo": b.correlativo,
            "numero_venta": b.venta.numero_venta,
            "cliente_numero_documento": b.venta.cliente.numero_documento,
            "cliente_razon_social": b.venta.cliente.razon_social,
            "total": b.venta.total,
            "fecha_emision": b.fecha_emision,
            "estado": b.estado,
        })

    grupos_formateados = []
    total_pendientes = 0
    for f_date, items in grupos_dict.items():
        total_monto = sum((item["total"] for item in items), Decimal("0.00"))
        total_pendientes += len(items)
        grupos_formateados.append({
            "fecha": f_date,
            "cantidad_boletas": len(items),
            "monto_total": total_monto,
            "boletas": items,
        })

    return {
        "grupos": grupos_formateados,
        "total_pendientes": total_pendientes,
    }


def crear_resumen_diario(db: Session, fecha_documentos: date) -> Dict[str, Any]:
    """
    Crea y emite un Resumen Diario para las boletas de una fecha específica en America/Lima.
    - Bloquea pesimistamente las boletas seleccionadas.
    - Aplica límite estricto de hasta 500 boletas por resumen.
    - Asigna correlativo seguro desde secuencia PostgreSQL (1 a 99999).
    - Persiste ResumenDiario y asociaciones antes de la llamada de red.
    - Invoca microservicio PHP /api/resumen.php.
    """
    subq_asociadas = (
        select(ResumenDiarioComprobante.comprobante_id)
        .join(ResumenDiario, ResumenDiarioComprobante.resumen_id == ResumenDiario.id)
        .where(ResumenDiario.estado.in_(["GENERADO", "ENVIADO", "EN_PROCESO", "ACEPTADO"]))
    )

    fecha_lima_col = func.date(func.timezone("America/Lima", Comprobante.fecha_emision))

    try:
        # 1. Bloqueo pesimista de las boletas candidatas de la fecha seleccionada
        boletas = (
            db.query(Comprobante)
            .join(Comprobante.venta)
            .join(Venta.cliente)
            .filter(Comprobante.tipo_comprobante == TIPO_BOLETA)
            .filter(Comprobante.estado == "PENDIENTE_ENVIO")
            .filter(fecha_lima_col == fecha_documentos)
            .filter(~Comprobante.id.in_(subq_asociadas))
            .with_for_update(of=Comprobante)
            .order_by(Comprobante.correlativo.asc())
            .limit(MAX_RESUMEN_BOLETAS)
            .all()
        )

        if not boletas:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No hay boletas pendientes disponibles para la fecha {fecha_documentos}.",
            )

        # 2. Generar correlativo desde la secuencia de PostgreSQL (límite 99999)
        correlativo = db.execute(text("SELECT nextval('resumen_diario_correlativo_seq')")).scalar()
        if correlativo > 99999:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="El correlativo del Resumen Diario ha superado el límite de 5 dígitos permitido (99999).",
            )

        fecha_resumen = datetime.now(LIMA_TZ).date()
        identificador = f"RC-{fecha_resumen.strftime('%Y%m%d')}-{correlativo:03d}"

        # 3. Crear entidad ResumenDiario en estado 'GENERADO'
        resumen = ResumenDiario(
            identificador=identificador,
            fecha_documentos=fecha_documentos,
            fecha_resumen=fecha_resumen,
            correlativo=correlativo,
            estado="GENERADO",
        )
        db.add(resumen)
        db.flush()

        # 4. Asociar boletas
        for b in boletas:
            asoc = ResumenDiarioComprobante(
                resumen_id=resumen.id,
                comprobante_id=b.id,
                estado_item="1",
            )
            db.add(asoc)

        db.commit()
        db.refresh(resumen)

    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creando el registro del Resumen Diario: {str(exc)}",
        ) from exc

    # 5. Construir payload para PHP
    comprobantes_payload = []
    for b in boletas:
        comprobantes_payload.append({
            "tipo": "03",
            "serie": b.serie,
            "correlativo": b.correlativo,
            "cliente_tipo": "6",
            "cliente_numero": b.venta.cliente.numero_documento,
            "total": str(b.venta.total),
            "gravadas": str(b.venta.subtotal),
            "igv": str(b.venta.igv),
            "estado": "1",
        })

    payload = {
        "fecha_documentos": str(fecha_documentos),
        "fecha_resumen": str(fecha_resumen),
        "correlativo": f"{correlativo:03d}",
        "comprobantes": comprobantes_payload,
    }

    # 6. Llamar a PHP
    try:
        result = sunat_client.emitir_resumen(payload)
    except SunatConnectionError as conn_err:
        resumen.estado = "ERROR_ENVIO"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ERROR_ENVIO: {conn_err.message}",
        ) from conn_err
    except SunatPayloadError as pay_err:
        resumen.estado = "ERROR_ENVIO"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{pay_err.code}: {pay_err.message}",
        ) from pay_err
    except Exception as exc:
        resumen.estado = "ERROR_ENVIO"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en comunicación con microservicio PHP: {str(exc)}",
        ) from exc

    # 7. Actualizar entidad ResumenDiario según respuesta
    enviado_sunat = result.get("enviado_sunat", False)
    resumen.xml_file = result.get("xml_file")

    if enviado_sunat:
        ticket = result.get("ticket")
        resumen.estado = "ENVIADO"
        resumen.ticket = ticket
        resumen.fecha_envio = func.now()
        # En modo SUNAT con ticket, las boletas pasan a ENVIADO
        for b in boletas:
            b.estado = "ENVIADO"
    else:
        # Modo local: queda GENERADO y boletas continúan PENDIENTE_ENVIO
        resumen.estado = "GENERADO"

    db.commit()
    db.refresh(resumen)
    return get_resumen_diario_by_id(db=db, resumen_id=resumen.id)


def enviar_resumen_diario(db: Session, resumen_id: int) -> Dict[str, Any]:
    """
    Reintenta el envío de un Resumen Diario existente en estado 'GENERADO' o 'ERROR_ENVIO'.
    Reutiliza el mismo resumen_id, correlativo, identificador RC y las mismas boletas.
    """
    resumen = db.query(ResumenDiario).filter(ResumenDiario.id == resumen_id).first()
    if not resumen:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resumen Diario con ID {resumen_id} no encontrado.",
        )

    if resumen.estado not in ("GENERADO", "ERROR_ENVIO"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"El resumen {resumen.identificador} ya se encuentra en estado '{resumen.estado}'. No requiere reenvío.",
        )

    boletas = resumen.comprobantes
    comprobantes_payload = []
    for b in boletas:
        comprobantes_payload.append({
            "tipo": "03",
            "serie": b.serie,
            "correlativo": b.correlativo,
            "cliente_tipo": "6",
            "cliente_numero": b.venta.cliente.numero_documento,
            "total": str(b.venta.total),
            "gravadas": str(b.venta.subtotal),
            "igv": str(b.venta.igv),
            "estado": "1",
        })

    payload = {
        "fecha_documentos": str(resumen.fecha_documentos),
        "fecha_resumen": str(resumen.fecha_resumen),
        "correlativo": f"{resumen.correlativo:03d}",
        "comprobantes": comprobantes_payload,
    }

    try:
        result = sunat_client.emitir_resumen(payload)
    except Exception as exc:
        resumen.estado = "ERROR_ENVIO"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al enviar Resumen Diario: {str(exc)}",
        ) from exc

    enviado_sunat = result.get("enviado_sunat", False)
    resumen.xml_file = result.get("xml_file")

    if enviado_sunat:
        resumen.estado = "ENVIADO"
        resumen.ticket = result.get("ticket")
        resumen.fecha_envio = func.now()
        for b in boletas:
            b.estado = "ENVIADO"
    else:
        resumen.estado = "GENERADO"

    db.commit()
    db.refresh(resumen)
    return get_resumen_diario_by_id(db=db, resumen_id=resumen.id)


def consultar_resumen_diario(db: Session, resumen_id: int) -> Dict[str, Any]:
    """
    Consulta el estado de un ticket ante SUNAT mediante api/consultar.php.
    - Si ACEPTADO: Resumen -> ACEPTADO, boletas -> ACEPTADO.
    - Si EN_PROCESO: Resumen -> EN_PROCESO.
    - Si RECHAZADO: Resumen -> RECHAZADO, boletas permanecen sin aceptar.
    """
    resumen = db.query(ResumenDiario).filter(ResumenDiario.id == resumen_id).first()
    if not resumen:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resumen Diario con ID {resumen_id} no encontrado.",
        )

    if not resumen.ticket:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"El resumen {resumen.identificador} no cuenta con un ticket emitido por SUNAT para consultar.",
        )

    try:
        result = sunat_client.consultar_ticket(resumen.ticket)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error consultando ticket ante SUNAT: {str(exc)}",
        ) from exc

    estado_consulta = result.get("estado")
    if estado_consulta == "ACEPTADO":
        resumen.estado = "ACEPTADO"
        resumen.codigo_respuesta = result.get("codigo")
        resumen.descripcion_respuesta = result.get("descripcion")
        resumen.observaciones = result.get("observaciones")
        resumen.cdr_file = result.get("cdr_file")
        resumen.fecha_respuesta = func.now()

        # Marcar todas las boletas incluidas como ACEPTADAS
        for b in resumen.comprobantes:
            b.estado = "ACEPTADO"

    elif estado_consulta == "EN_PROCESO":
        resumen.estado = "EN_PROCESO"
        resumen.codigo_respuesta = "98"
        resumen.descripcion_respuesta = result.get("descripcion")
    elif estado_consulta == "RECHAZADO":
        resumen.estado = "RECHAZADO"
        resumen.codigo_respuesta = result.get("codigo")
        resumen.descripcion_respuesta = result.get("descripcion")

    db.commit()
    db.refresh(resumen)
    return get_resumen_diario_by_id(db=db, resumen_id=resumen.id)


def get_resumenes_diarios(db: Session, skip: int = 0, limit: int = 50) -> Dict[str, Any]:
    """
    Listado paginado de resúmenes diarios con cantidad y montos acumulados.
    """
    query = db.query(ResumenDiario)
    total = query.count()
    items = (
        query.order_by(ResumenDiario.created_at.desc(), ResumenDiario.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    formatted = []
    for r in items:
        monto_total = sum((b.venta.total for b in r.comprobantes), Decimal("0.00"))
        formatted.append({
            "id": r.id,
            "identificador": r.identificador,
            "fecha_documentos": r.fecha_documentos,
            "fecha_resumen": r.fecha_resumen,
            "correlativo": r.correlativo,
            "estado": r.estado,
            "ticket": r.ticket,
            "codigo_respuesta": r.codigo_respuesta,
            "descripcion_respuesta": r.descripcion_respuesta,
            "total_boletas": len(r.comprobantes),
            "monto_total": monto_total,
            "fecha_envio": r.fecha_envio,
            "fecha_respuesta": r.fecha_respuesta,
            "created_at": r.created_at,
        })

    return {
        "items": formatted,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


def get_resumen_diario_by_id(db: Session, resumen_id: int) -> Dict[str, Any]:
    """
    Detalle completo de un Resumen Diario con sus boletas asociadas.
    """
    r = db.query(ResumenDiario).filter(ResumenDiario.id == resumen_id).first()
    if not r:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resumen Diario con ID {resumen_id} no encontrado.",
        )

    boletas_formateadas = []
    monto_total = Decimal("0.00")
    for b in r.comprobantes:
        monto_total += b.venta.total
        boletas_formateadas.append({
            "id": b.id,
            "comprobante_id": b.id,
            "numero_comprobante": b.numero_comprobante,
            "tipo_comprobante": b.tipo_comprobante,
            "serie": b.serie,
            "correlativo": b.correlativo,
            "numero_venta": b.venta.numero_venta,
            "cliente_numero_documento": b.venta.cliente.numero_documento,
            "cliente_razon_social": b.venta.cliente.razon_social,
            "total": b.venta.total,
            "fecha_emision": b.fecha_emision,
            "estado": b.estado,
        })

    return {
        "id": r.id,
        "identificador": r.identificador,
        "fecha_documentos": r.fecha_documentos,
        "fecha_resumen": r.fecha_resumen,
        "correlativo": r.correlativo,
        "estado": r.estado,
        "ticket": r.ticket,
        "codigo_respuesta": r.codigo_respuesta,
        "descripcion_respuesta": r.descripcion_respuesta,
        "observaciones": r.observaciones,
        "xml_file": r.xml_file,
        "cdr_file": r.cdr_file,
        "fecha_envio": r.fecha_envio,
        "fecha_respuesta": r.fecha_respuesta,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
        "total_boletas": len(boletas_formateadas),
        "monto_total": monto_total,
        "boletas": boletas_formateadas,
    }

