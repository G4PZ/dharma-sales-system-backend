from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    Sequence,
    String,
    func,
)
from sqlalchemy.orm import relationship

from app.database import Base

# Secuencia nativa registrada en Base.metadata para creación automática por create_all()
venta_numero_seq = Sequence("venta_numero_seq", metadata=Base.metadata)


class Venta(Base):
    __tablename__ = "ventas"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    numero_venta = Column(String(50), unique=True, index=True, nullable=False)
    cliente_id = Column(
        Integer,
        ForeignKey("clientes.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    subtotal = Column(Numeric(12, 2), nullable=False)
    igv = Column(Numeric(12, 2), nullable=False)
    total = Column(Numeric(12, 2), nullable=False)
    estado = Column(String(30), nullable=False, default="COMPLETADA", index=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("subtotal >= 0", name="chk_ventas_subtotal_positivo"),
        CheckConstraint("igv >= 0", name="chk_ventas_igv_positivo"),
        CheckConstraint("total >= 0", name="chk_ventas_total_positivo"),
    )

    # Relaciones SQLAlchemy
    cliente = relationship("Cliente", back_populates="ventas", lazy="joined")
    detalles = relationship(
        "DetalleVenta",
        back_populates="venta",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    comprobante = relationship(
        "Comprobante",
        back_populates="venta",
        uselist=False,
    )


class DetalleVenta(Base):
    __tablename__ = "detalle_ventas"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    venta_id = Column(
        Integer,
        ForeignKey("ventas.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    producto_id = Column(
        Integer,
        ForeignKey("productos.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    cantidad = Column(Integer, nullable=False)
    precio_unitario = Column(Numeric(10, 2), nullable=False)
    subtotal = Column(Numeric(12, 2), nullable=False)

    __table_args__ = (
        CheckConstraint("cantidad > 0", name="chk_detalle_ventas_cantidad_positiva"),
        CheckConstraint("precio_unitario >= 0", name="chk_detalle_ventas_precio_positivo"),
        CheckConstraint("subtotal >= 0", name="chk_detalle_ventas_subtotal_positivo"),
    )

    # Relaciones SQLAlchemy
    venta = relationship("Venta", back_populates="detalles")
    producto = relationship("Producto", back_populates="detalles_venta", lazy="joined")
