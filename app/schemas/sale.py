from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SaleItemCreate(BaseModel):
    producto_id: int = Field(..., gt=0, description="Identificador único del producto")
    cantidad: int = Field(..., gt=0, description="Cantidad entera mayor a cero")


class SaleCreate(BaseModel):
    cliente_id: int = Field(..., gt=0, description="Identificador único del cliente")
    productos: List[SaleItemCreate] = Field(
        ...,
        min_length=1,
        description="Lista de productos y cantidades a incluir en la venta"
    )


class SaleClientSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    numero_documento: str
    razon_social: str


class SaleClientDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    numero_documento: str
    razon_social: str
    nombre_contacto: Optional[str] = None
    telefono: Optional[str] = None
    email: Optional[str] = None
    direccion: Optional[str] = None
    ciudad: Optional[str] = None


class SaleDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    producto_id: int
    codigo_producto: str
    nombre_producto: str
    unidad_medida: str
    cantidad: int
    precio_unitario: Decimal
    subtotal: Decimal


class SaleListItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    numero_venta: str
    cliente: SaleClientSummary
    fecha: datetime
    cantidad_items: int
    subtotal: Decimal
    igv: Decimal
    total: Decimal
    estado: str


class SaleListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[SaleListItemResponse]
    total: int
    skip: int
    limit: int


class SaleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    numero_venta: str
    cliente_id: int
    cliente: SaleClientDetail
    detalles: List[SaleDetailResponse]
    subtotal: Decimal
    igv: Decimal
    total: Decimal
    estado: str
    created_at: datetime
    updated_at: datetime


class TopProductStats(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    nombre: Optional[str] = Field(None, description="Nombre del producto más vendido del mes")
    unidades: int = Field(0, description="Cantidad de unidades vendidas del producto en el mes")


class SaleStatsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ventas_mes: int = Field(..., description="Cantidad de ventas completadas del mes actual")
    monto_mes: Decimal = Field(..., description="Monto total vendido en Soles durante el mes actual")
    producto_mas_vendido: TopProductStats = Field(..., description="Producto con mayor cantidad vendida en el mes actual")
    unidades_mes: int = Field(..., description="Total de unidades físicas vendidas durante el mes actual")

