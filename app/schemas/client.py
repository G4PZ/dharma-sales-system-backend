from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ClientBase(BaseModel):
    numero_documento: str = Field(
        ...,
        min_length=11,
        max_length=11,
        pattern=r"^\d{11}$",
        description="Número de RUC (exactamente 11 dígitos numéricos)"
    )
    razon_social: str = Field(..., min_length=1, max_length=255, description="Razón Social de la empresa")
    nombre_contacto: Optional[str] = Field(None, max_length=150, description="Persona de contacto")
    telefono: Optional[str] = Field(None, max_length=50, description="Teléfono de contacto")
    email: Optional[str] = Field(None, max_length=150, description="Correo electrónico")
    direccion: Optional[str] = Field(None, max_length=255, description="Dirección fiscal o de entrega en Trujillo")


class ClientCreate(ClientBase):
    pass


class ClientUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nombre_contacto: Optional[str] = Field(None, max_length=150, description="Persona de contacto")
    telefono: Optional[str] = Field(None, max_length=50, description="Teléfono de contacto")
    email: Optional[str] = Field(None, max_length=150, description="Correo electrónico")
    direccion: Optional[str] = Field(None, max_length=255, description="Dirección fiscal o de entrega en Trujillo")


class ClientResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    tipo_documento: str
    numero_documento: str
    razon_social: str
    nombre_contacto: Optional[str] = None
    telefono: Optional[str] = None
    email: Optional[str] = None
    direccion: Optional[str] = None
    ciudad: Optional[str] = None
    tipo_cliente: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class ClientListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[ClientResponse]
    total: int
    skip: int
    limit: int


class ClientStatsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    clientes_activos: int
    nuevos_este_mes: int
    empresas_count: int
    clientes_inactivos: int
    ultimos_clientes: List[ClientResponse]
