from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ProductBase(BaseModel):
    codigo: str = Field(..., min_length=1, max_length=50, description="Código único del producto")
    nombre: str = Field(..., min_length=1, max_length=255, description="Nombre del producto")
    descripcion: Optional[str] = Field(None, description="Descripción o presentación del producto")
    categoria: str = Field(..., min_length=1, max_length=100, description="Categoría a la que pertenece")
    precio: Decimal = Field(..., ge=0, description="Precio de venta unitario en Soles")
    stock_minimo: int = Field(10, ge=0, description="Nivel mínimo requerido para alertas")
    unidad_medida: str = Field("und", max_length=50, description="Unidad de medida (und, galón, rollo, etc.)")
    is_active: bool = Field(True, description="Estado de disponibilidad")


class ProductCreate(ProductBase):
    model_config = ConfigDict(extra="forbid")


class ProductUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    descripcion: Optional[str] = None
    categoria: Optional[str] = Field(None, min_length=1, max_length=100)
    precio: Optional[Decimal] = Field(None, ge=0)
    stock_minimo: Optional[int] = Field(None, ge=0)
    unidad_medida: Optional[str] = Field(None, max_length=50)


# Alias en español
ProductoUpdate = ProductUpdate


class ProductResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    codigo: str
    nombre: str
    descripcion: Optional[str] = None
    categoria: str
    precio: Decimal
    stock: int
    stock_minimo: int
    unidad_medida: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class ProductListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[ProductResponse]
    total: int
    skip: int
    limit: int


class ProductAlertItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    codigo: str
    nombre: str
    categoria: str
    stock: int
    stock_minimo: int
    unidad_medida: str


class ProductStatsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    productos_activos: int
    stock_bajo_count: int
    categorias_count: int
    valor_inventario: Decimal
    alertas_stock: List[ProductAlertItem]
