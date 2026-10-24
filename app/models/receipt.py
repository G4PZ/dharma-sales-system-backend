from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Sequence,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from app.database import Base

# Secuencias nativas registradas en Base.metadata para portabilidad con create_all()
factura_correlativo_seq = Sequence("factura_correlativo_seq", metadata=Base.metadata)
boleta_correlativo_seq = Sequence("boleta_correlativo_seq", metadata=Base.metadata)


class Comprobante(Base):
    __tablename__ = "comprobantes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    venta_id = Column(
        Integer,
        ForeignKey("ventas.id", ondelete="RESTRICT"),
        unique=True,
        nullable=False,
        index=True,
    )
    tipo_comprobante = Column(String(2), nullable=False)
    serie = Column(String(4), nullable=False)
    correlativo = Column(Integer, nullable=False)
    fecha_emision = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    estado = Column(
        String(30),
        nullable=False,
        default="PENDIENTE_ENVIO",
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "tipo_comprobante",
            "serie",
            "correlativo",
            name="uq_comprobantes_tipo_serie_correlativo",
        ),
        CheckConstraint(
            "tipo_comprobante IN ('01', '03')",
            name="chk_comprobantes_tipo_comprobante",
        ),
        CheckConstraint(
            "correlativo > 0",
            name="chk_comprobantes_correlativo_positivo",
        ),
        CheckConstraint(
            "estado IN ('PENDIENTE_ENVIO', 'ENVIADO', 'ACEPTADO', 'RECHAZADO', 'ERROR_ENVIO')",
            name="chk_comprobantes_estado",
        ),
    )

    # Relaciones SQLAlchemy
    venta = relationship("Venta", back_populates="comprobante", lazy="joined")
    respuestas_sunat = relationship(
        "RespuestaSunat",
        back_populates="comprobante",
        cascade="all, delete-orphan",
        order_by="desc(RespuestaSunat.created_at)",
    )
    resumenes_diarios = relationship(
        "ResumenDiario",
        secondary="resumen_diario_comprobantes",
        back_populates="comprobantes",
        lazy="selectin",
    )

    @property
    def numero_comprobante(self) -> str:
        """Formato visual estándar de comprobante electrónico (ej. F001-00000001)"""
        return f"{self.serie}-{self.correlativo:08d}"


class RespuestaSunat(Base):
    __tablename__ = "respuesta_sunat"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    comprobante_id = Column(
        Integer,
        ForeignKey("comprobantes.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    codigo = Column(String(50), nullable=True)
    estado = Column(String(50), nullable=True)
    descripcion = Column(Text, nullable=True)
    observaciones = Column(JSON, nullable=True)
    ticket = Column(String(100), nullable=True)
    fecha_respuesta = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # Relación con comprobante
    comprobante = relationship("Comprobante", back_populates="respuestas_sunat")
