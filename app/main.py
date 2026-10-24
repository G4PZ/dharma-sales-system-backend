from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.database import Base, engine
import app.models  # noqa: F401 - Asegura registro de modelos en SQLAlchemy Base
from app.routes.auth import router as auth_router
from app.routes.alertas import router as alertas_router
from app.routes.clientes import router as clientes_router
from app.routes.comprobantes import router as comprobantes_router
from app.routes.dashboard import router as dashboard_router
from app.routes.productos import router as productos_router
from app.routes.ventas import router as ventas_router

# Creación automática de tablas mediante SQLAlchemy en desarrollo
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Dharma Sales API",
    version="0.1.0",
    description="API para la gestión de ventas e inventario de Dharma E.I.R.L."
)

# Configuración de CORS para permitir comunicación con el Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-Requested-With"],
)

# Registro de routers
app.include_router(auth_router, prefix="/api/auth", tags=["Autenticación"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(alertas_router, prefix="/api/alertas", tags=["Alertas IA"])
app.include_router(clientes_router, prefix="/api/clientes", tags=["Clientes"])
app.include_router(comprobantes_router, prefix="/api/comprobantes", tags=["Comprobantes"])
app.include_router(productos_router, prefix="/api/productos", tags=["Productos"])
app.include_router(ventas_router, prefix="/api/ventas", tags=["Ventas"])





@app.get("/")
def root():
    return {
        "message": "Dharma Sales API funcionando correctamente"
    }


@app.get("/database-test")
def database_test():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))

        return {
            "database": "PostgreSQL conectado correctamente",
            "result": result.scalar()
        }