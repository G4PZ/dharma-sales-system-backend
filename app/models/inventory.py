from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.orm import relationship
from app.database import Base


class EntradaStock(Base):
    __tablename__ = "entradas_stock"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    producto_id = Column(
        Integer,
        ForeignKey("productos.id", ondelete="RESTRICT"),
        nullable=False,
        index=True
    )
    cantidad = Column(Integer, nullable=False)
    costo_unitario = Column(Numeric(10, 2), nullable=False)
    fecha_ingreso = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    observaciones = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relación con el producto principal
    producto = relationship("Producto", back_populates="entradas_stock")


class HistorialPrecio(Base):
    __tablename__ = "historial_precios"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    producto_id = Column(
        Integer,
        ForeignKey("productos.id", ondelete="RESTRICT"),
        nullable=False,
        index=True
    )
    precio_anterior = Column(Numeric(10, 2), nullable=False)
    precio_nuevo = Column(Numeric(10, 2), nullable=False)
    motivo = Column(String(255), nullable=False)
    fecha_cambio = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relación con el producto principal
    producto = relationship("Producto", back_populates="historial_precios")
