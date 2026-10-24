from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class RangoFechasInfo(BaseModel):
    """Metadatos del rango de fechas activo evaluado en America/Lima."""
    fecha_desde: str
    fecha_hasta: str
    dias_totales: int

    model_config = ConfigDict(from_attributes=True)


class VentasResumen(BaseModel):
    """Métricas de ventas e ingresos correspondientes al rango de fechas seleccionado."""
    cantidad: int
    ingresos: Decimal

    model_config = ConfigDict(from_attributes=True)


# Alias para compatibilidad
VentasHoyResumen = VentasResumen


class ComprobantesResumen(BaseModel):
    """Consolidado de comprobantes según su estado real ante SUNAT dentro del rango seleccionado."""
    total: int
    aceptados: int
    pendientes: int
    rechazados: int

    model_config = ConfigDict(from_attributes=True)


class AlertasIAResumen(BaseModel):
    """Métricas ejecutivas de alertas emitidas por el Motor de IA dentro del rango seleccionado."""
    total: int
    pendientes: int
    nivel_alto: int
    bajo: int
    medio: int
    alto: int

    model_config = ConfigDict(from_attributes=True)


class VentaDiaEvolucion(BaseModel):
    """Punto de datos diario para el gráfico de evolución temporal dentro del rango seleccionado."""
    fecha: str
    dia: str
    total: Decimal
    cantidad: int

    model_config = ConfigDict(from_attributes=True)


class ComprobanteEstadoItem(BaseModel):
    """Segmento para visualización tipo dona/pie de distribución de comprobantes."""
    estado: str
    nombre: str
    cantidad: int
    porcentaje: float

    model_config = ConfigDict(from_attributes=True)


class AlertaNivelItem(BaseModel):
    """Distribución de alertas por nivel de severidad para gráfico de barras."""
    nivel: str
    cantidad: int

    model_config = ConfigDict(from_attributes=True)


class AlertaRecienteItem(BaseModel):
    """Alerta del motor IA dentro del rango seleccionado para el widget informativo."""
    id: int
    numero_venta: Optional[str] = None
    cliente_razon_social: Optional[str] = None
    nivel_alerta: str
    estado_revision: str
    descripcion: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VentaRecienteItem(BaseModel):
    """Venta registrada dentro del rango seleccionado para la tabla de ventas recientes."""
    id: int
    numero_venta: str
    cliente_razon_social: Optional[str] = None
    fecha: datetime
    total: Decimal
    estado: str

    model_config = ConfigDict(from_attributes=True)


class DashboardResumenResponse(BaseModel):
    """Estructura centralizada de Business Intelligence para el Dashboard Dharma."""
    rango_fechas: RangoFechasInfo
    ventas: VentasResumen
    ventas_hoy: VentasResumen  # Mantenido para retrocompatibilidad
    comprobantes: ComprobantesResumen
    alertas_ia: AlertasIAResumen
    evolucion_ventas: List[VentaDiaEvolucion]
    comprobantes_distribucion: List[ComprobanteEstadoItem]
    alertas_por_nivel: List[AlertaNivelItem]
    alertas_recientes: List[AlertaRecienteItem]
    ventas_recientes: List[VentaRecienteItem]

    model_config = ConfigDict(from_attributes=True)
