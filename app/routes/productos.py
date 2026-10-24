from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.routes.auth import get_current_user
from app.schemas.inventory import (
    PriceHistoryListResponse,
    StockEntryCreate,
    StockEntryListResponse,
    StockEntryResponse,
)
from app.schemas.product import (
    ProductCreate,
    ProductListResponse,
    ProductResponse,
    ProductStatsResponse,
    ProductUpdate,
)
from app.services import inventory_service, product_service

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("", response_model=ProductListResponse, summary="Listar productos con filtros y paginación")
def list_products(
    search: Optional[str] = Query(None, description="Búsqueda por código, nombre, categoría o descripción"),
    categoria: Optional[str] = Query(None, description="Filtrar por categoría"),
    estado: Optional[str] = Query(None, description="Filtrar por estado: 'activo', 'inactivo', 'todos'"),
    nivel_stock: Optional[str] = Query(None, description="Filtrar por nivel: 'normal', 'bajo', 'agotado', 'todos'"),
    skip: int = Query(0, ge=0, description="Cantidad de registros a omitir"),
    limit: int = Query(50, ge=1, le=100, description="Cantidad de registros por página"),
    db: Session = Depends(get_db),
):
    return product_service.get_products(
        db=db,
        search=search,
        categoria=categoria,
        estado=estado,
        nivel_stock=nivel_stock,
        skip=skip,
        limit=limit,
    )


@router.get("/stats", response_model=ProductStatsResponse, summary="Obtener estadísticas y alertas de inventario")
def get_product_stats(db: Session = Depends(get_db)):
    return product_service.get_product_stats(db=db)


@router.get("/{product_id}", response_model=ProductResponse, summary="Obtener producto por ID")
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = product_service.get_product_by_id(db=db, product_id=product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Producto con ID {product_id} no encontrado."
        )
    return product


@router.post("", response_model=ProductResponse, status_code=status.HTTP_201_CREATED, summary="Registrar nuevo producto")
def create_product(product_in: ProductCreate, db: Session = Depends(get_db)):
    return product_service.create_product(db=db, product_in=product_in)


@router.put("/{product_id}", response_model=ProductResponse, summary="Actualizar información de un producto")
def update_product(
    product_id: int,
    product_in: ProductUpdate,
    db: Session = Depends(get_db),
):
    product = product_service.get_product_by_id(db=db, product_id=product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Producto con ID {product_id} no encontrado."
        )
    return product_service.update_product(db=db, db_product=product, product_in=product_in)


@router.patch("/{product_id}/toggle-status", response_model=ProductResponse, summary="Activar o desactivar producto")
def toggle_product_status(product_id: int, db: Session = Depends(get_db)):
    product = product_service.get_product_by_id(db=db, product_id=product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Producto con ID {product_id} no encontrado."
        )
    return product_service.toggle_product_status(db=db, db_product=product)


# =============================================================================
# ENDPOINTS DE FASE 1: ENTRADAS DE STOCK E HISTORIAL DE PRECIOS
# =============================================================================

@router.post(
    "/{product_id}/entradas-stock",
    response_model=StockEntryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Registrar reposición de stock y actualización comercial opcional",
)
def create_stock_entry(
    product_id: int,
    entry_in: StockEntryCreate,
    db: Session = Depends(get_db),
):
    return inventory_service.create_stock_entry(
        db=db,
        product_id=product_id,
        entry_in=entry_in,
    )


@router.get(
    "/{product_id}/entradas-stock",
    response_model=StockEntryListResponse,
    summary="Consultar historial de entradas / reposiciones de un producto",
)
def get_stock_entries(
    product_id: int,
    skip: int = Query(0, ge=0, description="Cantidad de registros a omitir"),
    limit: int = Query(50, ge=1, le=100, description="Cantidad de registros por página"),
    db: Session = Depends(get_db),
):
    return inventory_service.get_stock_entries(
        db=db,
        product_id=product_id,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{product_id}/historial-precios",
    response_model=PriceHistoryListResponse,
    summary="Consultar historial de cambios en el precio de venta comercial",
)
def get_price_history(
    product_id: int,
    skip: int = Query(0, ge=0, description="Cantidad de registros a omitir"),
    limit: int = Query(50, ge=1, le=100, description="Cantidad de registros por página"),
    db: Session = Depends(get_db),
):
    return inventory_service.get_price_history(
        db=db,
        product_id=product_id,
        skip=skip,
        limit=limit,
    )
