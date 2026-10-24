from sqlalchemy import Boolean, Column, DateTime, Integer, String, func
from sqlalchemy.orm import relationship
from app.database import Base


class Cliente(Base):
    __tablename__ = "clientes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    tipo_documento = Column(String(20), nullable=False, default="RUC")
    numero_documento = Column(String(20), unique=True, index=True, nullable=False)
    razon_social = Column(String(255), index=True, nullable=False)
    nombre_contacto = Column(String(150), nullable=True)
    telefono = Column(String(50), nullable=True)
    email = Column(String(150), index=True, nullable=True)
    direccion = Column(String(255), nullable=True)
    ciudad = Column(String(100), nullable=False, default="Trujillo")
    tipo_cliente = Column(String(50), nullable=False, default="Empresa")
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relación con ventas
    ventas = relationship("Venta", back_populates="cliente")

