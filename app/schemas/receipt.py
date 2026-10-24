from datetime import datetime
from decimal import Decimal
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ReceiptCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    venta_id: int = Field(..., gt=0, description="Identificador único de la venta a comprobar")
    tipo_comprobante: str = Field(
        ...,
        pattern="^(01|03)$",
        description="Código de tipo de comprobante electrónico: '01' (Factura) o '03' (Boleta)",
    )


class ReceiptClientSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    numero_documento: str
    razon_social: str
    direccion: Optional[str] = None
    email: Optional[str] = None
    telefono: Optional[str] = None


class ReceiptItemDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    producto_id: int
    codigo_producto: str
    nombre_producto: str
    unidad_medida: str
    cantidad: int
    precio_unitario: Decimal
    subtotal: Decimal


class ReceiptSunatResponseItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    codigo: Optional[str] = None
    estado: Optional[str] = None
    descripcion: Optional[str] = None
    observaciones: Optional[Any] = None
    ticket: Optional[str] = None
    fecha_respuesta: Optional[datetime] = None
    created_at: datetime


class ReceiptListItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    venta_id: int
    numero_venta: str
    cliente_numero_documento: str
    cliente_razon_social: str
    tipo_comprobante: str
    nombre_tipo: str
    serie: str
    correlativo: int
    numero_comprobante: str
    total: Decimal
    fecha_emision: datetime
    estado: str
    created_at: datetime


class ReceiptListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[ReceiptListItemResponse]
    total: int
    skip: int
    limit: int


class ReceiptSaleDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    numero_venta: str
    subtotal: Decimal
    igv: Decimal
    total: Decimal
    created_at: datetime


class ReceiptDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    venta_id: int
    tipo_comprobante: str
    nombre_tipo: str
    serie: str
    correlativo: int
    numero_comprobante: str
    fecha_emision: datetime
    estado: str
    created_at: datetime
    updated_at: datetime

    venta: ReceiptSaleDetail
    cliente: ReceiptClientSummary
    detalles: List[ReceiptItemDetail]
    respuestas_sunat: List[ReceiptSunatResponseItem]
    resumen_diario: Optional[Any] = None


class ReceiptStatsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total_comprobantes: int
    facturas_count: int
    boletas_count: int
    pendientes_envio: int


class AvailableSaleItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    numero_venta: str
    cliente_id: int
    cliente_numero_documento: str
    cliente_razon_social: str
    subtotal: Decimal
    igv: Decimal
    total: Decimal
    created_at: datetime


class AvailableSalesListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[AvailableSaleItem]
    total: int
    skip: int
    limit: int
