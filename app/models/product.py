from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.orm import relationship
from app.database import Base


class Producto(Base):
    __tablename__ = "productos"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    codigo = Column(String(50), unique=True, index=True, nullable=False)
    nombre = Column(String(255), index=True, nullable=False)
    descripcion = Column(Text, nullable=True)
    categoria = Column(String(100), index=True, nullable=False)
    precio = Column(Numeric(10, 2), nullable=False)
    stock = Column(Integer, nullable=False, default=0)
    stock_minimo = Column(Integer, nullable=False, default=10)
    unidad_medida = Column(String(50), nullable=False, default="und")
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relación con detalles de ventas
    detalles_venta = relationship("DetalleVenta", back_populates="producto")

    # Relaciones de inventario e historial de precios
    entradas_stock = relationship("EntradaStock", back_populates="producto", order_by="desc(EntradaStock.fecha_ingreso)")
    historial_precios = relationship("HistorialPrecio", back_populates="producto", order_by="desc(HistorialPrecio.fecha_cambio)")


