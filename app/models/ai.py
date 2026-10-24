from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from typing import Optional

from sqlalchemy.orm import relationship

from app.database import Base


class ModeloEntrenamiento(Base):
    """
    Entidad que almacena los metadatos, versionado, métricas de entrenamiento
    y ubicación de los artefactos serializados independientes para los modelos
    de Inteligencia Artificial (K-Means para clientes e Isolation Forest para operaciones de venta).
    """
    __tablename__ = "modelos_entrenamiento"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    tipo_modelo = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Tipo de modelo: KMEANS_CLIENTES o ISOLATION_FOREST_VENTAS",
    )
    version = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Identificador único de versión del modelo",
    )
    parametros = Column(
        JSON,
        nullable=False,
        comment="Hiperparámetros de scikit-learn utilizados en el entrenamiento",
    )
    metricas = Column(
        JSON,
        nullable=False,
        comment="Métricas cuantitativas obtenidas durante el entrenamiento",
    )
    total_muestras = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Cantidad total de registros utilizados en el dataset de entrenamiento",
    )
    ruta_artefacto = Column(
        String(255),
        nullable=False,
        comment="Ruta física de persistencia del artefacto serializado .joblib",
    )
    estado = Column(
        String(30),
        nullable=False,
        default="ACTIVO",
        index=True,
        comment="Estado operativo del modelo: ACTIVO, OBSOLETO, ERROR",
    )
    resumen_modelo = Column(
        JSON,
        nullable=True,
        comment="Estructura resumida del modelo (ej. centroides/perfiles para K-Means o distribuciones para Isolation Forest)",
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "tipo_modelo",
            "version",
            name="uq_modelos_tipo_version",
        ),
        CheckConstraint(
            "tipo_modelo IN ('KMEANS_CLIENTES', 'ISOLATION_FOREST_VENTAS')",
            name="chk_modelos_tipo_modelo",
        ),
        CheckConstraint(
            "estado IN ('ACTIVO', 'OBSOLETO', 'ERROR')",
            name="chk_modelos_estado",
        ),
        CheckConstraint(
            "total_muestras >= 0",
            name="chk_modelos_total_muestras_positivo",
        ),
    )

    # Relaciones explícitas e independientes para evitar ambigüedad con AlertaVenta
    alertas_isolation_forest = relationship(
        "AlertaVenta",
        foreign_keys="[AlertaVenta.modelo_isolation_forest_id]",
        back_populates="modelo_isolation_forest",
    )
    alertas_kmeans = relationship(
        "AlertaVenta",
        foreign_keys="[AlertaVenta.modelo_kmeans_id]",
        back_populates="modelo_kmeans",
    )


class AlertaVenta(Base):
    """
    Entidad que almacena las alertas operativas detectadas por el Motor IA.
    Registra trazabilidad exacta de los dos modelos participantes:
    1. modelo_isolation_forest_id (obligatorio): modelo de detección de operaciones.
    2. modelo_kmeans_id (opcional): modelo de segmentación de perfil del cliente (nullable para clientes nuevos).
    """
    __tablename__ = "alertas_venta"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    venta_id = Column(
        Integer,
        ForeignKey("ventas.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="Operación de venta analizada",
    )
    cliente_id = Column(
        Integer,
        ForeignKey("clientes.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="Cliente asociado a la venta",
    )
    modelo_isolation_forest_id = Column(
        Integer,
        ForeignKey("modelos_entrenamiento.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="Modelo Isolation Forest (ISOLATION_FOREST_VENTAS) utilizado para la detección",
    )
    modelo_kmeans_id = Column(
        Integer,
        ForeignKey("modelos_entrenamiento.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
        comment="Modelo K-Means (KMEANS_CLIENTES) utilizado para el perfil contextual; null para clientes sin historial",
    )
    puntaje_anomalia = Column(
        Numeric(6, 4),
        nullable=False,
        comment="Índice/puntaje de anomalía generado por el modelo; no representa probabilidad de fraude",
    )
    nivel_alerta = Column(
        String(20),
        nullable=False,
        index=True,
        comment="Nivel categorizado: BAJO, MEDIO, ALTO",
    )
    cluster_perfil_id = Column(
        Integer,
        nullable=True,
        comment="Clúster de comportamiento del cliente asignado por K-Means (si aplica)",
    )
    perfil_nombre = Column(
        String(100),
        nullable=True,
        comment="Etiqueta descriptiva del perfil de cliente asignado",
    )
    descripcion = Column(
        Text,
        nullable=False,
        comment="Explicación legible de la anomalía identificada",
    )
    factores_contribuyentes = Column(
        JSON,
        nullable=False,
        comment="Desglose estructurado de las características y desviaciones que motivaron la alerta",
    )
    estado_revision = Column(
        String(30),
        nullable=False,
        default="PENDIENTE",
        index=True,
        comment="Estado de revisión humana: PENDIENTE, CONFIRMADA, FALSO_POSITIVO",
    )
    retroalimentacion_notas = Column(
        Text,
        nullable=True,
        comment="Comentarios u observaciones registradas durante la revisión humana",
    )
    revisado_por = Column(
        String(150),
        nullable=True,
        comment="Correo del usuario que auditó la alerta",
    )
    fecha_revision = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Fecha y hora de la revisión",
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        # Restricción UNIQUE parcial cuando se utiliza perfil K-Means (modelo_kmeans_id IS NOT NULL)
        Index(
            "uq_alertas_venta_if_kmeans",
            "venta_id",
            "modelo_isolation_forest_id",
            "modelo_kmeans_id",
            unique=True,
            postgresql_where=text("modelo_kmeans_id IS NOT NULL"),
        ),
        # Restricción UNIQUE parcial cuando NO se utiliza K-Means (modelo_kmeans_id IS NULL)
        # Impide duplicados para la misma venta y mismo modelo Isolation Forest cuando el cliente es nuevo
        Index(
            "uq_alertas_venta_if_sin_kmeans",
            "venta_id",
            "modelo_isolation_forest_id",
            unique=True,
            postgresql_where=text("modelo_kmeans_id IS NULL"),
        ),
        CheckConstraint(
            "nivel_alerta IN ('BAJO', 'MEDIO', 'ALTO')",
            name="chk_alertas_nivel_alerta",
        ),
        CheckConstraint(
            "estado_revision IN ('PENDIENTE', 'CONFIRMADA', 'FALSO_POSITIVO')",
            name="chk_alertas_estado_revision",
        ),
    )

    # Relaciones SQLAlchemy explícitas sin ambigüedad
    venta = relationship("Venta", foreign_keys=[venta_id], lazy="joined")
    cliente = relationship("Cliente", foreign_keys=[cliente_id], lazy="joined")
    modelo_isolation_forest = relationship(
        "ModeloEntrenamiento",
        foreign_keys=[modelo_isolation_forest_id],
        back_populates="alertas_isolation_forest",
        lazy="joined",
    )
    modelo_kmeans = relationship(
        "ModeloEntrenamiento",
        foreign_keys=[modelo_kmeans_id],
        back_populates="alertas_kmeans",
        lazy="joined",
    )

    @property
    def numero_venta(self) -> Optional[str]:
        return self.venta.numero_venta if self.venta else None

    @property
    def cliente_razon_social(self) -> Optional[str]:
        return self.cliente.razon_social if self.cliente else None
