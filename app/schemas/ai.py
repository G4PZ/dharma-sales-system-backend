from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# SCHEMAS DE MODELO DE ENTRENAMIENTO (METADATOS)
# =============================================================================

class TrainingModelBase(BaseModel):
    tipo_modelo: str = Field(
        ...,
        pattern="^(KMEANS_CLIENTES|ISOLATION_FOREST_VENTAS)$",
        description="Identificador del algoritmo: KMEANS_CLIENTES o ISOLATION_FOREST_VENTAS",
    )
    version: str = Field(..., min_length=1, max_length=50, description="Identificador único de versión del modelo")
    parametros: Dict[str, Any] = Field(..., description="Hiperparámetros de scikit-learn utilizados")
    metricas: Dict[str, Any] = Field(..., description="Métricas de evaluación del entrenamiento")
    total_muestras: int = Field(..., ge=0, description="Cantidad de registros utilizados para entrenar")
    ruta_artefacto: str = Field(..., description="Ruta de persistencia del artefacto serializado .joblib")
    estado: str = Field(
        "ACTIVO",
        pattern="^(ACTIVO|OBSOLETO|ERROR)$",
        description="Estado operativo del modelo",
    )
    resumen_modelo: Optional[Dict[str, Any]] = Field(
        None,
        description="Estructura resumen de centroides o distribuciones del modelo",
    )


class TrainingModelResponse(TrainingModelBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


# =============================================================================
# SCHEMAS AUXILIARES PARA RELACIONES DE ALERTA
# =============================================================================

class AlertSaleSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    numero_venta: str
    total: Decimal
    created_at: datetime


class AlertClientSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    numero_documento: str
    razon_social: str


class AlertModelSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    tipo_modelo: str
    version: str


# =============================================================================
# SCHEMAS DE ALERTAS IA
# =============================================================================

class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    venta_id: int
    numero_venta: Optional[str] = Field(None, description="Número correlativo de la venta (ej. V-000128)")
    cliente_id: int
    cliente_razon_social: Optional[str] = Field(None, description="Razón social del cliente asociado")
    modelo_isolation_forest_id: int = Field(
        ...,
        description="ID del modelo Isolation Forest (ISOLATION_FOREST_VENTAS) utilizado para la detección",
    )
    modelo_kmeans_id: Optional[int] = Field(
        None,
        description="ID del modelo K-Means (KMEANS_CLIENTES) utilizado para el perfil contextual; null para clientes sin historial",
    )
    puntaje_anomalia: Decimal = Field(
        ...,
        description="Índice o puntaje de anomalía generado por el modelo; no representa probabilidad de fraude",
    )
    nivel_alerta: str = Field(..., description="Nivel de alerta: BAJO, MEDIO, ALTO")
    cluster_perfil_id: Optional[int] = None
    perfil_nombre: Optional[str] = None
    descripcion: str
    estado_revision: str = Field(..., description="Estado de revisión: PENDIENTE, CONFIRMADA, FALSO_POSITIVO")
    revisado_por: Optional[str] = None
    fecha_revision: Optional[datetime] = None
    created_at: datetime


class AlertDetailResponse(AlertResponse):
    model_config = ConfigDict(from_attributes=True)

    factores_contribuyentes: Any = Field(
        ...,
        description="Desglose explicable de las variables y desviaciones que motivaron la alerta",
    )
    retroalimentacion_notas: Optional[str] = None

    # Entidades relacionadas enriquecidas con trazabilidad dual de modelos
    venta: Optional[AlertSaleSummary] = None
    cliente: Optional[AlertClientSummary] = None
    modelo_isolation_forest: Optional[AlertModelSummary] = None
    modelo_kmeans: Optional[AlertModelSummary] = None


class AlertListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[AlertResponse]
    total: int
    skip: int
    limit: int


# =============================================================================
# SCHEMA DE RETROALIMENTACIÓN HUMANA (FEEDBACK)
# =============================================================================

class AlertFeedbackUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estado_revision: str = Field(
        ...,
        pattern="^(CONFIRMADA|FALSO_POSITIVO)$",
        description="Resolución de la revisión: CONFIRMADA o FALSO_POSITIVO",
    )
    retroalimentacion_notas: Optional[str] = Field(
        None,
        max_length=1000,
        description="Observaciones justificativas de la retroalimentación",
    )


# =============================================================================
# SCHEMAS DE ESTADÍSTICAS Y ESTADO DE MODELOS
# =============================================================================

class AlertStatsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total: int
    pendientes: int
    confirmadas: int
    falsos_positivos: int
    bajo: int
    medio: int
    alto: int


class ModelAvailabilityInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    disponible: bool
    estado: str
    version: Optional[str] = None
    fecha_entrenamiento: Optional[datetime] = None
    total_muestras: int = 0
    mensaje: Optional[str] = None


class ModelsStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    isolation_forest: ModelAvailabilityInfo
    kmeans: ModelAvailabilityInfo


# =============================================================================
# SCHEMA DE RESPUESTA PARA ANÁLISIS DE VENTA
# =============================================================================

class SaleAnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    venta_id: int
    numero_venta: Optional[str] = Field(None, description="Número correlativo visible de la venta (ej. V-000127)")
    cliente_id: int
    nivel_alerta: str = Field(..., description="Nivel: NORMAL, BAJO, MEDIO, ALTO")
    puntaje_anomalia: float = Field(..., description="Índice relativo percentilar [0.0, 1.0]")
    raw_anomaly_score: float = Field(..., description="Score bruto continuo del estimador (-decision_function)")
    alerta_generada: bool = Field(..., description="True si se persistió una AlertaVenta (nivel != NORMAL)")
    alerta_id: Optional[int] = None
    ya_analizada: bool = Field(False, description="True si la venta ya contaba con alerta para los mismos modelos")
    modelo_isolation_forest_id: int
    modelo_isolation_forest_version: str
    modelo_kmeans_id: Optional[int] = None
    modelo_kmeans_version: Optional[str] = None
    cluster_perfil_id: Optional[int] = None
    perfil_nombre: Optional[str] = None
    descripcion: str
    factores_contribuyentes: List[Any]


# =============================================================================
# SCHEMAS DE PREPARACIÓN DE ENTRENAMIENTO Y EJECUCIÓN ADMINISTRATIVA
# =============================================================================

class KMeansReadinessResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    clientes_totales: int = Field(..., description="Total de clientes registrados en el sistema")
    clientes_elegibles: int = Field(..., description="Clientes con al menos 2 compras completadas")
    compras_minimas_por_cliente: int = Field(2, description="Criterio de elegibilidad individual")
    clientes_minimos_requeridos: int = Field(15, description="Guard de volumen mínimo para K-Means")
    puede_entrenar: bool = Field(..., description="True si se alcanzan al menos 15 clientes elegibles")
    porcentaje_progreso: float = Field(..., description="Progreso porcentual hacia el requisito (0-100%)")


class IsolationForestReadinessResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ventas_completadas: int = Field(..., description="Total de ventas con estado COMPLETADA")
    ventas_minimas_requeridas: int = Field(50, description="Guard de volumen mínimo para Isolation Forest")
    puede_entrenar: bool = Field(..., description="True si se alcanzan al menos 50 ventas completadas")
    porcentaje_progreso: float = Field(..., description="Progreso porcentual hacia el requisito (0-100%)")


class ModelsReadinessResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kmeans: KMeansReadinessResponse
    isolation_forest: IsolationForestReadinessResponse


class TrainingExecutionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: str = Field(..., description="Estado del proceso: ENTRENADO_Y_ACTIVO")
    modelo_id: int
    version: str
    total_muestras: int
    ruta_artefacto: str
    metricas: Dict[str, Any]
    resumen_modelo: Dict[str, Any]


