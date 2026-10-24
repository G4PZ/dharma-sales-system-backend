from datetime import date, datetime
from decimal import Decimal
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DailySummaryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fecha_documentos: date = Field(
        ...,
        description="Fecha de emisión de las Boletas a incluir en el Resumen Diario (formato YYYY-MM-DD)",
    )


class DailySummaryReceiptItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    comprobante_id: int
    numero_comprobante: str
    tipo_comprobante: str
    serie: str
    correlativo: int
    numero_venta: str
    cliente_numero_documento: str
    cliente_razon_social: str
    total: Decimal
    fecha_emision: datetime
    estado: str


class DailySummaryListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    identificador: str
    fecha_documentos: date
    fecha_resumen: date
    correlativo: int
    estado: str
    ticket: Optional[str] = None
    codigo_respuesta: Optional[str] = None
    descripcion_respuesta: Optional[str] = None
    total_boletas: int
    monto_total: Decimal
    fecha_envio: Optional[datetime] = None
    fecha_respuesta: Optional[datetime] = None
    created_at: datetime


class DailySummaryDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    identificador: str
    fecha_documentos: date
    fecha_resumen: date
    correlativo: int
    estado: str
    ticket: Optional[str] = None
    codigo_respuesta: Optional[str] = None
    descripcion_respuesta: Optional[str] = None
    observaciones: Optional[Any] = None
    xml_file: Optional[str] = None
    cdr_file: Optional[str] = None
    fecha_envio: Optional[datetime] = None
    fecha_respuesta: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    total_boletas: int
    monto_total: Decimal
    boletas: List[DailySummaryReceiptItem]


class DailySummaryListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[DailySummaryListItem]
    total: int
    skip: int
    limit: int


class PendingBoletaGroup(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    fecha: date
    cantidad_boletas: int
    monto_total: Decimal
    boletas: List[DailySummaryReceiptItem]


class PendingBoletasResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    grupos: List[PendingBoletaGroup]
    total_pendientes: int


class SummaryReceiptInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    identificador: str
    fecha_resumen: date
    estado: str
    ticket: Optional[str] = None
    codigo_respuesta: Optional[str] = None
    descripcion_respuesta: Optional[str] = None
    observaciones: Optional[Any] = None
    fecha_envio: Optional[datetime] = None
    fecha_respuesta: Optional[datetime] = None
