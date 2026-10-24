from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
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

resumen_diario_correlativo_seq = Sequence(
    "resumen_diario_correlativo_seq",
    metadata=Base.metadata,
    minvalue=1,
    maxvalue=99999,
    cycle=True,
)


class ResumenDiario(Base):
    __tablename__ = "resumenes_diarios"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    identificador = Column(String(30), unique=True, nullable=False, index=True)
    fecha_documentos = Column(Date, nullable=False, index=True)
    fecha_resumen = Column(Date, nullable=False)
    correlativo = Column(Integer, nullable=False)
    estado = Column(
        String(30),
        nullable=False,
        default="GENERADO",
        index=True,
    )
    ticket = Column(String(100), nullable=True)
    codigo_respuesta = Column(String(50), nullable=True)
    descripcion_respuesta = Column(Text, nullable=True)
    observaciones = Column(JSON, nullable=True)
    xml_file = Column(String(255), nullable=True)
    cdr_file = Column(String(255), nullable=True)
    fecha_envio = Column(DateTime(timezone=True), nullable=True)
    fecha_respuesta = Column(DateTime(timezone=True), nullable=True)
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
            "fecha_resumen",
            "correlativo",
            name="uq_resumenes_fecha_correlativo",
        ),
        CheckConstraint(
            "correlativo >= 1 AND correlativo <= 99999",
            name="chk_resumenes_correlativo_rango",
        ),
        CheckConstraint(
            "estado IN ('GENERADO', 'ENVIADO', 'EN_PROCESO', 'ACEPTADO', 'RECHAZADO', 'ERROR_ENVIO')",
            name="chk_resumenes_estado",
        ),
    )

    # Relación muchos a muchos con Comprobante
    comprobantes = relationship(
        "Comprobante",
        secondary="resumen_diario_comprobantes",
        back_populates="resumenes_diarios",
        lazy="selectin",
    )


class ResumenDiarioComprobante(Base):
    __tablename__ = "resumen_diario_comprobantes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    resumen_id = Column(
        Integer,
        ForeignKey("resumenes_diarios.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    comprobante_id = Column(
        Integer,
        ForeignKey("comprobantes.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    estado_item = Column(
        String(2),
        nullable=False,
        default="1",
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "resumen_id",
            "comprobante_id",
            name="uq_resumen_comprobante",
        ),
        CheckConstraint(
            "estado_item IN ('1', '2', '3')",
            name="chk_resumen_comp_estado_item",
        ),
    )
