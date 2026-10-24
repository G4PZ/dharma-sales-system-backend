from decimal import Decimal
from typing import Optional
from fastapi import HTTPException, status
from sqlalchemy import distinct, func, or_
from sqlalchemy.orm import Session

from app.models.inventory import HistorialPrecio
from app.models.product import Producto
from app.schemas.product import ProductCreate, ProductUpdate


def get_products(
    db: Session,
    search: Optional[str] = None,
    categoria: Optional[str] = None,
    estado: Optional[str] = None,
    nivel_stock: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
) -> dict:
    query = db.query(Producto)

    # Filtro de búsqueda textual (código, nombre o categoría)
    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Producto.codigo.ilike(term),
                Producto.nombre.ilike(term),
                Producto.categoria.ilike(term),
                Producto.descripcion.ilike(term),
            )
        )

    # Filtro de categoría
    if categoria and categoria.strip().lower() not in ("todas", "todas las categorías", ""):
        query = query.filter(Producto.categoria.ilike(categoria.strip()))

    # Filtro de estado
    if estado:
        est = estado.strip().lower()
        if est in ("activo", "activos", "true"):
            query = query.filter(Producto.is_active.is_(True))
        elif est in ("inactivo", "inactivos", "false"):
            query = query.filter(Producto.is_active.is_(False))

    # Filtro de nivel de stock
    if nivel_stock:
        ns = nivel_stock.strip().lower()
        if ns in ("bajo", "stock bajo"):
            query = query.filter(Producto.stock <= Producto.stock_minimo, Producto.stock > 0)
        elif ns in ("agotado", "sin stock"):
            query = query.filter(Producto.stock == 0)
        elif ns in ("normal", "stock normal"):
            query = query.filter(Producto.stock > Producto.stock_minimo)

    total = query.count()
    items = query.order_by(Producto.id.asc()).offset(skip).limit(limit).all()

    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


def get_product_by_id(db: Session, product_id: int) -> Optional[Producto]:
    return db.query(Producto).filter(Producto.id == product_id).first()


def get_product_by_code(db: Session, codigo: str) -> Optional[Producto]:
    return db.query(Producto).filter(Producto.codigo.ilike(codigo.strip())).first()


def create_product(db: Session, product_in: ProductCreate) -> Producto:
    existing = get_product_by_code(db, product_in.codigo)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ya existe un producto registrado con el código '{product_in.codigo}'."
        )

    product_data = product_in.model_dump()
    # Todo nuevo producto inicia estrictamente con stock = 0
    product = Producto(**product_data, stock=0)
    db.add(product)
    db.commit()
    db.refresh(product)
    return product


# Lista blanca de campos actualizables (stock estrictamente excluido)
ALLOWED_UPDATE_FIELDS = {
    "descripcion",
    "categoria",
    "precio",
    "stock_minimo",
    "unidad_medida",
}


def update_product(db: Session, db_product: Producto, product_in: ProductUpdate) -> Producto:
    update_data = product_in.model_dump(exclude_unset=True)

    # Si se actualiza el precio comercial y difiere del actual, generar trazabilidad histórica
    nuevo_precio = update_data.get("precio")
    if nuevo_precio is not None and nuevo_precio != db_product.precio:
        historial = HistorialPrecio(
            producto_id=db_product.id,
            precio_anterior=db_product.precio,
            precio_nuevo=nuevo_precio,
            motivo="Actualización manual desde catálogo",
            fecha_cambio=func.now(),
        )
        db.add(historial)

    # Segunda capa de seguridad: aplicar únicamente campos permitidos por la lista blanca
    for field, value in update_data.items():
        if field in ALLOWED_UPDATE_FIELDS:
            setattr(db_product, field, value)

    db.commit()
    db.refresh(db_product)
    return db_product


def toggle_product_status(db: Session, db_product: Producto) -> Producto:
    db_product.is_active = not db_product.is_active
    db.commit()
    db.refresh(db_product)
    return db_product


def get_product_stats(db: Session) -> dict:
    # 1. Total de productos activos
    productos_activos = db.query(Producto).filter(Producto.is_active.is_(True)).count()

    # 2. Productos con stock bajo (stock <= stock_minimo)
    stock_bajo_count = db.query(Producto).filter(
        Producto.stock <= Producto.stock_minimo,
        Producto.is_active.is_(True)
    ).count()

    # 3. Conteo de categorías distintas
    categorias_count = db.query(func.count(distinct(Producto.categoria))).scalar() or 0

    # 4. Valor total del inventario: sum(stock * precio)
    valor_inv = db.query(func.coalesce(func.sum(Producto.stock * Producto.precio), 0)).scalar()
    valor_inventario = Decimal(str(valor_inv)) if valor_inv is not None else Decimal("0.00")

    # 5. Lista de productos con stock bajo para el panel lateral de alertas
    alertas_stock = (
        db.query(Producto)
        .filter(Producto.stock <= Producto.stock_minimo)
        .order_by(Producto.stock.asc())
        .limit(10)
        .all()
    )

    return {
        "productos_activos": productos_activos,
        "stock_bajo_count": stock_bajo_count,
        "categorias_count": categorias_count,
        "valor_inventario": valor_inventario,
        "alertas_stock": alertas_stock,
    }
