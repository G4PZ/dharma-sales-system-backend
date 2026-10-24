import os
from contextlib import asynccontextmanager
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sincronización de tablas en SQLAlchemy sin bloquear la importación del módulo
    if engine is not None:
        try:
            Base.metadata.create_all(bind=engine)
        except Exception as e:
            print(f"[ALERTA] No se pudieron sincronizar las tablas de SQLAlchemy al inicio: {e}")
    yield

app = FastAPI(
    title="Dharma Sales API",
    version="0.1.0",
    description="API para la gestión de ventas e inventario de Dharma E.I.R.L.",
    lifespan=lifespan
)

# Configuración de CORS para desarrollo local y producción
allowed_origins = [
    "http://localhost:3000",
    "https://dharma-sales-system-frontend.vercel.app",
]
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    for origin in allowed_origins_env.split(","):
        origin_clean = origin.strip()
        if origin_clean and origin_clean not in allowed_origins:
            allowed_origins.append(origin_clean)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allowed_origins_env == "*" else allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-Requested-With", "Authorization"],
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
    if engine is None:
        return {
            "status": "error",
            "message": "DATABASE_URL no está configurada en las variables de entorno"
        }
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return {
                "database": "PostgreSQL conectado correctamente",
                "result": result.scalar()
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error conectando a PostgreSQL: {str(e)}"
        }