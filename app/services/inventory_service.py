from typing import Dict, Optional
from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.inventory import EntradaStock, HistorialPrecio
from app.models.product import Producto
from app.schemas.inventory import StockEntryCreate


def create_stock_entry(
    db: Session,
    product_id: int,
    entry_in: StockEntryCreate,
) -> EntradaStock:
    """
    Registra una entrada de stock de forma transaccional y atómica:
    1. Bloquea la fila del producto con SELECT ... FOR UPDATE (pesimista).
    2. Valida la existencia del producto y reglas de negocio.
    3. Registra el ingreso en la tabla entradas_stock.
    4. Incrementa el stock total del producto.
    5. Si se indica nuevo_precio_venta y difiere del actual, inserta el
       registro en historial_precios y actualiza productos.precio.
    6. Confirma la transacción con commit, garantizando atomicidad total ante fallos.
    """
    try:
        # 1. Bloqueo pesimista con FOR UPDATE
        producto = (
            db.query(Producto)
            .filter(Producto.id == product_id)
            .with_for_update()
            .first()
        )

        if not producto:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Producto con ID {product_id} no encontrado."
            )

        # 2. Validaciones de cantidad y costo
        if entry_in.cantidad <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="La cantidad recibida debe ser un entero estrictamente mayor a 0."
            )

        if entry_in.costo_unitario < 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="El costo unitario no puede ser negativo."
            )

        # 3. Insertar registro de entrada
        nueva_entrada = EntradaStock(
            producto_id=producto.id,
            cantidad=entry_in.cantidad,
            costo_unitario=entry_in.costo_unitario,
            fecha_ingreso=entry_in.fecha_ingreso if entry_in.fecha_ingreso else func.now(),
            observaciones=entry_in.observaciones.strip() if entry_in.observaciones else None,
        )
        db.add(nueva_entrada)

        # 4. Incrementar stock
        producto.stock += entry_in.cantidad

        # 5. Evaluación de cambio de precio de venta comercial
        if (
            entry_in.nuevo_precio_venta is not None
            and entry_in.nuevo_precio_venta != producto.precio
        ):
            nuevo_historial = HistorialPrecio(
                producto_id=producto.id,
                precio_anterior=producto.precio,
                precio_nuevo=entry_in.nuevo_precio_venta,
                motivo="Actualización de precio por reposición de mercadería",
                fecha_cambio=func.now(),
            )
            db.add(nuevo_historial)
            producto.precio = entry_in.nuevo_precio_venta

        # 6. Commit de la transacción atómica
        db.commit()
        db.refresh(nueva_entrada)
        db.refresh(producto)
        return nueva_entrada

    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado al registrar el ingreso de stock: {str(exc)}"
        ) from exc


def get_stock_entries(
    db: Session,
    product_id: int,
    skip: int = 0,
    limit: int = 50,
) -> Dict:
    """
    Obtiene el listado cronológico de entradas de stock para un producto específico,
    ordenado desde la más reciente hacia la más antigua.
    """
    producto = db.query(Producto).filter(Producto.id == product_id).first()
    if not producto:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Producto con ID {product_id} no encontrado."
        )

    query = db.query(EntradaStock).filter(EntradaStock.producto_id == product_id)
    total = query.count()
    items = (
        query.order_by(EntradaStock.fecha_ingreso.desc(), EntradaStock.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


def get_price_history(
    db: Session,
    product_id: int,
    skip: int = 0,
    limit: int = 50,
) -> Dict:
    """
    Obtiene el historial de modificaciones de precio comercial para un producto,
    ordenado desde el cambio más reciente.
    """
    producto = db.query(Producto).filter(Producto.id == product_id).first()
    if not producto:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Producto con ID {product_id} no encontrado."
        )

    query = db.query(HistorialPrecio).filter(HistorialPrecio.producto_id == product_id)
    total = query.count()
    items = (
        query.order_by(HistorialPrecio.fecha_cambio.desc(), HistorialPrecio.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
    }
