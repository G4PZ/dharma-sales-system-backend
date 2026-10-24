from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class StockEntryBase(BaseModel):
    cantidad: int = Field(..., gt=0, description="Cantidad recibida de unidades (debe ser mayor a 0)")
    costo_unitario: Decimal = Field(..., ge=0, description="Costo unitario de compra del proveedor (S/)")
    nuevo_precio_venta: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Nuevo precio comercial opcional para todo el producto en Soles"
    )
    observaciones: Optional[str] = Field(
        None,
        max_length=500,
        description="Número de factura, guía de remisión o notas del proveedor"
    )
    fecha_ingreso: Optional[datetime] = Field(
        None,
        description="Fecha de ingreso personalizada (si se omite se usa la fecha actual)"
    )


class StockEntryCreate(StockEntryBase):
    model_config = ConfigDict(extra="forbid")


class StockEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    producto_id: int
    cantidad: int
    costo_unitario: Decimal
    fecha_ingreso: datetime
    observaciones: Optional[str] = None
    created_at: datetime


class StockEntryListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[StockEntryResponse]
    total: int
    skip: int
    limit: int


class PriceHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    producto_id: int
    precio_anterior: Decimal
    precio_nuevo: Decimal
    motivo: str
    fecha_cambio: datetime
    created_at: datetime


class PriceHistoryListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[PriceHistoryResponse]
    total: int
    skip: int
    limit: int
