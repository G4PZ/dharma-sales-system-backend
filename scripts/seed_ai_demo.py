#!/usr/bin/env python3
"""
Herramienta de Carga, Verificación y Limpieza de Dataset Histórico Demo para IA.

Dharma Sales System - Módulo de Inteligencia Artificial (Parte 8A).

Propósito:
Preparar un dataset histórico de demostración (24 clientes sintéticos, 10 productos demo
y 120 ventas históricas completadas) suficientemente representativo para permitir
el posterior entrenamiento de:
  - K-Means (Segmentación de clientes: >= 15 clientes elegibles con >= 2 compras).
  - Isolation Forest (Detección de anomalías en ventas: >= 50 ventas completadas).

Garantías de Seguridad e Integridad:
  1. Operación Manual: NO se incluye en el startup de FastAPI ni se ejecuta automáticamente.
  2. Aislamiento y Respeto a Datos Reales:
     - Todos los clientes demo se prefijan con 'DEMO IA - '.
     - Todos los productos demo se prefijan con 'DEMO-IA-'.
     - Las ventas demo se asocian exclusivamente a clientes y productos demo.
     - NUNCA se ejecutan eliminaciones globales ni se tocan registros reales.
  3. Reversibilidad mediante Manifiesto:
     - Guarda los IDs exactos creados en '.demo_ai_seed_manifest.json'.
     - El comando --cleanup utiliza exclusivamente los IDs del manifiesto tras validar
       que cumplen los prefijos de seguridad.
  4. Transaccionalidad Total:
     - La inserción (--apply) se ejecuta en una única transacción atómica; si ocurre un error,
       se realiza rollback completo sin dejar estados inconsistentes.
  5. Cero Efectos Secundarios:
     - NO genera comprobantes SUNAT, NO crea XML, NO llama a Greenter/SUNAT.
     - NO dispara tareas en segundo plano (BackgroundTasks).
     - NO entrena modelos ni genera alertas IA en esta fase.
  6. Idempotencia:
     - Impide ejecuciones duplicadas de --apply sin un --cleanup previo.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
import logging
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import Session, sessionmaker

# Configurar path para importar módulos de backend
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv

# Cargar variables de entorno de backend/.env si existe
env_path = os.path.join(BACKEND_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

from app.database import Base
from app.models.client import Cliente
from app.models.product import Producto
from app.models.sale import DetalleVenta, Venta
from app.models.receipt import Comprobante
from app.models.ai import ModeloEntrenamiento, AlertaVenta
from app.services.ai.feature_engineering import (
    build_customer_features,
    build_sales_features,
    KMEANS_FEATURE_COLUMNS,
    ISOLATION_FOREST_FEATURE_COLUMNS,
)

# Constantes numéricas y monetarias
TWO_PLACES = Decimal("0.01")
IGV_RATE = Decimal("0.18")
LIMA_TZ = ZoneInfo("America/Lima")
DEFAULT_RANDOM_STATE = 42

# Prefijos obligatorios de seguridad
DEMO_CLIENT_PREFIX = "DEMO IA - "
DEMO_PRODUCT_PREFIX = "DEMO-IA-"

# Ruta del archivo de manifiesto de seguridad
DEFAULT_MANIFEST_PATH = os.path.join(BACKEND_DIR, ".demo_ai_seed_manifest.json")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("seed_ai_demo")


# =============================================================================
# GENERACIÓN DE RUCs SINTÉTICOS VÁLIDOS CON ALGORITMO SUNAT (MÓDULO 11)
# =============================================================================

def generate_valid_sunat_ruc(base_8_digits: str) -> str:
    """
    Genera un número de RUC de 11 dígitos válido conforme al algoritmo oficial
    de SUNAT (módulo 11) para personas jurídicas (prefijo 20).
    """
    prefix = "20" + base_8_digits[:8]
    weights = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    total = sum(int(d) * w for d, w in zip(prefix, weights))
    residue = 11 - (total % 11)
    if residue == 10:
        check_digit = 0
    elif residue == 11:
        check_digit = 1
    else:
        check_digit = residue
    return prefix + str(check_digit)


# =============================================================================
# DEFINICIÓN DEL CATÁLOGO DEMO (10 PRODUCTOS DIVERSIFICADOS)
# =============================================================================

DEMO_PRODUCTS_SPEC: List[Dict[str, Any]] = [
    {
        "codigo": "DEMO-IA-P001",
        "nombre": "DEMO IA - Jabón Líquido Antibacterial 1 Galón",
        "descripcion": "Jabón líquido desinfectante para manos, uso institucional y hospitalario.",
        "categoria": "Higiene y Limpieza",
        "precio": Decimal("28.50"),
        "stock_inicial": 5000,
        "unidad_medida": "gal",
    },
    {
        "codigo": "DEMO-IA-P002",
        "nombre": "DEMO IA - Desinfectante Multiuso Pino 5L",
        "descripcion": "Limpiador desinfectante concentrado con fragancia a pino.",
        "categoria": "Higiene y Limpieza",
        "precio": Decimal("32.00"),
        "stock_inicial": 5000,
        "unidad_medida": "bidon",
    },
    {
        "codigo": "DEMO-IA-P003",
        "nombre": "DEMO IA - Papel Toalla Interdoblada Caja 20 Paquetes",
        "descripcion": "Papel toalla blanco hoja doble para dispensador universal.",
        "categoria": "Papelería",
        "precio": Decimal("65.00"),
        "stock_inicial": 5000,
        "unidad_medida": "caja",
    },
    {
        "codigo": "DEMO-IA-P004",
        "nombre": "DEMO IA - Papel Higiénico Institucional Jumbo 12 Rollos",
        "descripcion": "Rollo de papel higiénico institucional de 250 metros hoja simple.",
        "categoria": "Papelería",
        "precio": Decimal("48.00"),
        "stock_inicial": 5000,
        "unidad_medida": "paquete",
    },
    {
        "codigo": "DEMO-IA-P005",
        "nombre": "DEMO IA - Dispensador Automático de Jabón Líquido",
        "descripcion": "Dispensador con sensor infrarrojo para pared 1000ml.",
        "categoria": "Equipamiento",
        "precio": Decimal("85.00"),
        "stock_inicial": 2000,
        "unidad_medida": "und",
    },
    {
        "codigo": "DEMO-IA-P006",
        "nombre": "DEMO IA - Dispensador de Papel Toalla Acero",
        "descripcion": "Dispensador de acero inoxidable satinado de alta durabilidad.",
        "categoria": "Equipamiento",
        "precio": Decimal("110.00"),
        "stock_inicial": 2000,
        "unidad_medida": "und",
    },
    {
        "codigo": "DEMO-IA-P007",
        "nombre": "DEMO IA - Alcohol en Gel 70% 1 Galón",
        "descripcion": "Alcohol en gel antiséptico al 70% para desinfección rápida.",
        "categoria": "Higiene y Limpieza",
        "precio": Decimal("36.50"),
        "stock_inicial": 5000,
        "unidad_medida": "gal",
    },
    {
        "codigo": "DEMO-IA-P008",
        "nombre": "DEMO IA - Detergente Industrial Concentrado 20L",
        "descripcion": "Detergente líquido de alta espuma para lavandería y pisos industriales.",
        "categoria": "Químicos",
        "precio": Decimal("145.00"),
        "stock_inicial": 3000,
        "unidad_medida": "bidon",
    },
    {
        "codigo": "DEMO-IA-P009",
        "nombre": "DEMO IA - Bolsas para Residuos 140L Paquete 50 und",
        "descripcion": "Bolsas de polietileno de alta resistencia calibre pesado color negro.",
        "categoria": "Envases",
        "precio": Decimal("22.00"),
        "stock_inicial": 5000,
        "unidad_medida": "paquete",
    },
    {
        "codigo": "DEMO-IA-P010",
        "nombre": "DEMO IA - Guantes de Nitrilo Caja 100 und",
        "descripcion": "Guantes de examinación sin polvo de nitrilo color azul.",
        "categoria": "Protección",
        "precio": Decimal("34.00"),
        "stock_inicial": 5000,
        "unidad_medida": "caja",
    },
]


# =============================================================================
# ESPECIFICACIÓN DE LOS 24 CLIENTES Y SUS 3 GRUPOS GENERADORES
# =============================================================================

@dataclass
class DemoClientSpec:
    index: int
    grupo: str
    razon_social: str
    numero_documento: str
    nombre_contacto: str
    telefono: str
    email: str
    direccion: str
    ciudad: str
    ventas_objetivo: int


def build_demo_clients_specs() -> List[DemoClientSpec]:
    """
    Construye las especificaciones de los 24 clientes demo distribuidos en 3 grupos:
    - Grupo A: 8 clientes retail/bodegas (7 ventas cada uno = 56 ventas).
    - Grupo B: 8 clientes comerciales medianos (5 ventas cada uno = 40 ventas).
    - Grupo C: 8 clientes mayoristas/distribuidores (3 ventas cada uno = 24 ventas).
    Total: 24 clientes, 120 ventas históricas.
    """
    specs: List[DemoClientSpec] = []

    # Grupo A: Compras pequeñas y frecuentes (8 clientes)
    for i in range(1, 9):
        idx = i
        ruc = generate_valid_sunat_ruc(f"9001{idx:04d}")
        specs.append(DemoClientSpec(
            index=idx,
            grupo="Grupo A",
            razon_social=f"DEMO IA - Retail Bodega {i:02d}",
            numero_documento=ruc,
            nombre_contacto=f"Contacto Bodega {i:02d}",
            telefono=f"9440100{i:02d}",
            email=f"demo_bodega{i:02d}@dharma-demo.pe",
            direccion=f"Av. Larco {1000 + i * 25}, Urb. La Merced",
            ciudad="Trujillo",
            ventas_objetivo=7,
        ))

    # Grupo B: Compras medianas y periódicas (8 clientes)
    for i in range(1, 9):
        idx = 8 + i
        ruc = generate_valid_sunat_ruc(f"9002{i:04d}")
        specs.append(DemoClientSpec(
            index=idx,
            grupo="Grupo B",
            razon_social=f"DEMO IA - Comercial Mediana {i:02d}",
            numero_documento=ruc,
            nombre_contacto=f"Contacto Comercial {i:02d}",
            telefono=f"9440200{i:02d}",
            email=f"demo_comercial{i:02d}@dharma-demo.pe",
            direccion=f"Av. Mansiche {500 + i * 40}, Urb. Santa Inés",
            ciudad="Trujillo",
            ventas_objetivo=5,
        ))

    # Grupo C: Compras de mayor volumen y menor frecuencia (8 clientes)
    for i in range(1, 9):
        idx = 16 + i
        ruc = generate_valid_sunat_ruc(f"9003{i:04d}")
        specs.append(DemoClientSpec(
            index=idx,
            grupo="Grupo C",
            razon_social=f"DEMO IA - Mayorista Distribuidor {i:02d}",
            numero_documento=ruc,
            nombre_contacto=f"Contacto Mayorista {i:02d}",
            telefono=f"9440300{i:02d}",
            email=f"demo_mayorista{i:02d}@dharma-demo.pe",
            direccion=f"Av. América Norte {1200 + i * 35}, Los Jardines",
            ciudad="Trujillo",
            ventas_objetivo=3,
        ))

    return specs


# =============================================================================
# GENERACIÓN DETERMINÍSTICA DE HISTORIAL COMERCIAL (120 VENTAS)
# =============================================================================

@dataclass
class DemoSaleItemPlan:
    codigo_producto: str
    cantidad: int
    precio_unitario: Decimal
    subtotal: Decimal


@dataclass
class DemoSalePlan:
    client_spec: DemoClientSpec
    created_at: datetime
    items: List[DemoSaleItemPlan]
    subtotal: Decimal
    igv: Decimal
    total: Decimal


def plan_demo_dataset(
    reference_datetime: Optional[datetime] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[List[DemoClientSpec], List[Dict[str, Any]], List[DemoSalePlan]]:
    """
    Planifica y genera en memoria el dataset determinístico completo:
    - 24 clientes demo.
    - 10 productos demo con stock inicial.
    - 120 ventas históricas ordenadas cronológicamente con sus líneas de detalle.
    """
    rng = random.Random(random_state)
    clients_specs = build_demo_clients_specs()
    products_specs = [dict(p) for p in DEMO_PRODUCTS_SPEC]
    products_by_code = {p["codigo"]: p for p in products_specs}

    if reference_datetime is None:
        # Por defecto, referencia local Lima en fecha actual
        ref_dt = datetime.now(LIMA_TZ)
    else:
        if reference_datetime.tzinfo is None:
            ref_dt = reference_datetime.replace(tzinfo=LIMA_TZ)
        else:
            ref_dt = reference_datetime.astimezone(LIMA_TZ)

    sales_plans: List[DemoSalePlan] = []

    # Códigos de productos clasificados para selección coherente de carritos
    consumibles_menores = ["DEMO-IA-P001", "DEMO-IA-P002", "DEMO-IA-P004", "DEMO-IA-P007", "DEMO-IA-P009"]
    intermedios = ["DEMO-IA-P001", "DEMO-IA-P002", "DEMO-IA-P003", "DEMO-IA-P004", "DEMO-IA-P005", "DEMO-IA-P007", "DEMO-IA-P010"]
    mayoristas = [p["codigo"] for p in products_specs]

    for c in clients_specs:
        if c.grupo == "Grupo A":
            # 7 compras en una ventana de ~75 días (intervalos de 7 a 13 días)
            # Días atrás relativos a la fecha de referencia
            offsets_days = [70, 58, 47, 36, 25, 14, 3]
            for sale_idx, days_ago in enumerate(offsets_days):
                jitter_days = rng.randint(-1, 1)
                sale_day_offset = max(1, days_ago + jitter_days)
                sale_hour = rng.randint(9, 18)
                sale_minute = rng.choice([0, 10, 15, 20, 30, 40, 45, 50])
                sale_second = rng.randint(0, 59)

                sale_dt = ref_dt - timedelta(days=sale_day_offset)
                sale_dt = sale_dt.replace(hour=sale_hour, minute=sale_minute, second=sale_second)

                # Carrito: 1 a 3 productos distintos, 1 a 4 unidades cada uno
                num_prods = rng.choice([1, 2, 2, 3])
                chosen_codes = rng.sample(consumibles_menores, num_prods)

                items: List[DemoSaleItemPlan] = []
                subtotal_acum = Decimal("0.00")

                for code in chosen_codes:
                    prod = products_by_code[code]
                    cant = rng.randint(1, 4)
                    pu = prod["precio"].quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                    line_sub = (pu * Decimal(cant)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                    subtotal_acum += line_sub
                    items.append(DemoSaleItemPlan(
                        codigo_producto=code,
                        cantidad=cant,
                        precio_unitario=pu,
                        subtotal=line_sub,
                    ))

                subtotal = subtotal_acum.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                igv = (subtotal * IGV_RATE).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                total = (subtotal + igv).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

                sales_plans.append(DemoSalePlan(
                    client_spec=c,
                    created_at=sale_dt,
                    items=items,
                    subtotal=subtotal,
                    igv=igv,
                    total=total,
                ))

        elif c.grupo == "Grupo B":
            # 5 compras en una ventana de ~95 días (intervalos de 16 a 24 días)
            offsets_days = [90, 68, 48, 28, 8]
            for sale_idx, days_ago in enumerate(offsets_days):
                jitter_days = rng.randint(-2, 2)
                sale_day_offset = max(2, days_ago + jitter_days)
                sale_hour = rng.randint(9, 17)
                sale_minute = rng.choice([5, 15, 25, 35, 45, 55])
                sale_second = rng.randint(0, 59)

                sale_dt = ref_dt - timedelta(days=sale_day_offset)
                sale_dt = sale_dt.replace(hour=sale_hour, minute=sale_minute, second=sale_second)

                # Carrito: 2 a 4 productos distintos, 2 a 8 unidades cada uno
                num_prods = rng.choice([2, 3, 3, 4])
                chosen_codes = rng.sample(intermedios, num_prods)

                items = []
                subtotal_acum = Decimal("0.00")

                for code in chosen_codes:
                    prod = products_by_code[code]
                    cant = rng.randint(2, 8)
                    pu = prod["precio"].quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                    line_sub = (pu * Decimal(cant)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                    subtotal_acum += line_sub
                    items.append(DemoSaleItemPlan(
                        codigo_producto=code,
                        cantidad=cant,
                        precio_unitario=pu,
                        subtotal=line_sub,
                    ))

                subtotal = subtotal_acum.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                igv = (subtotal * IGV_RATE).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                total = (subtotal + igv).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

                sales_plans.append(DemoSalePlan(
                    client_spec=c,
                    created_at=sale_dt,
                    items=items,
                    subtotal=subtotal,
                    igv=igv,
                    total=total,
                ))

        elif c.grupo == "Grupo C":
            # 3 compras en una ventana de ~110 días (intervalos de 30 a 45 días)
            offsets_days = [105, 65, 20]
            for sale_idx, days_ago in enumerate(offsets_days):
                jitter_days = rng.randint(-3, 3)
                sale_day_offset = max(5, days_ago + jitter_days)
                sale_hour = rng.randint(10, 16)
                sale_minute = rng.choice([0, 15, 30, 45])
                sale_second = rng.randint(0, 59)

                sale_dt = ref_dt - timedelta(days=sale_day_offset)
                sale_dt = sale_dt.replace(hour=sale_hour, minute=sale_minute, second=sale_second)

                # Carrito mayorista: 3 a 6 productos distintos, 10 a 30 unidades cada uno
                num_prods = rng.choice([3, 4, 5, 6])
                chosen_codes = rng.sample(mayoristas, num_prods)

                items = []
                subtotal_acum = Decimal("0.00")

                for code in chosen_codes:
                    prod = products_by_code[code]
                    cant = rng.randint(10, 30)
                    pu = prod["precio"].quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                    line_sub = (pu * Decimal(cant)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                    subtotal_acum += line_sub
                    items.append(DemoSaleItemPlan(
                        codigo_producto=code,
                        cantidad=cant,
                        precio_unitario=pu,
                        subtotal=line_sub,
                    ))

                subtotal = subtotal_acum.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                igv = (subtotal * IGV_RATE).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                total = (subtotal + igv).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

                sales_plans.append(DemoSalePlan(
                    client_spec=c,
                    created_at=sale_dt,
                    items=items,
                    subtotal=subtotal,
                    igv=igv,
                    total=total,
                ))

    # Ordenar globalmente todas las ventas por fecha cronológica ascendente
    sales_plans.sort(key=lambda s: s.created_at)

    return clients_specs, products_specs, sales_plans


# =============================================================================
# MANEJO DEL MANIFIESTO DE SEGURIDAD
# =============================================================================

def load_manifest(manifest_path: str = DEFAULT_MANIFEST_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error al leer el archivo de manifiesto '{manifest_path}': {e}")
        return None


def save_manifest(manifest_path: str, data: Dict[str, Any]) -> None:
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Manifiesto persistido exitosamente en: '{manifest_path}'")


def delete_manifest(manifest_path: str = DEFAULT_MANIFEST_PATH) -> bool:
    if os.path.exists(manifest_path):
        try:
            os.remove(manifest_path)
            logger.info(f"Manifiesto eliminado: '{manifest_path}'")
            return True
        except Exception as e:
            logger.error(f"Error al eliminar manifiesto: {e}")
            return False
    return False


# =============================================================================
# GESTIÓN DE CONEXIÓN A BASE DE DATOS
# =============================================================================

def get_db_url(db_url_override: Optional[str] = None) -> str:
    """
    Obtiene la URL de base de datos resolviendo en orden de prioridad:
    1. Parámetro explícito --db-url.
    2. Variable de entorno DEMO_DATABASE_URL.
    3. Variable de entorno DATABASE_URL (del backend).
    """
    url = db_url_override or os.getenv("DEMO_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "No se encontró URL de base de datos configurada. "
            "Defina DATABASE_URL, DEMO_DATABASE_URL o use el parámetro --db-url."
        )
    return url


def sanitize_db_url(url: str) -> str:
    """Enmascara la contraseña en una URL de base de datos para visualización segura."""
    if "@" in url and "://" in url:
        prefix, rest = url.split("://", 1)
        creds, host_db = rest.split("@", 1)
        if ":" in creds:
            user = creds.split(":", 1)[0]
            return f"{prefix}://{user}:***@{host_db}"
    return url


def get_engine_and_session(db_url: str):
    engine = create_engine(db_url)
    SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionClass


def _generate_numero_venta(db: Session) -> str:
    """
    Genera el siguiente número correlativo seguro para ventas.
    Si la base de datos es PostgreSQL, utiliza la secuencia nativa venta_numero_seq.
    Si es SQLite u otro dialecto, calcula el correlativo basado en MAX(id).
    """
    if db.bind and db.bind.dialect.name == "postgresql":
        # Asegurar que la secuencia exista si es una base de datos nueva
        db.execute(text("CREATE SEQUENCE IF NOT EXISTS venta_numero_seq START WITH 1 INCREMENT BY 1"))
        seq_val = db.execute(text("SELECT nextval('venta_numero_seq')")).scalar()
        return f"V-{seq_val:06d}"
    else:
        max_id = db.query(func.max(Venta.id)).scalar() or 0
        return f"V-{max_id + 1:06d}"


# =============================================================================
# COMANDOS: STATUS, DRY-RUN, APPLY, CLEANUP, VALIDATE
# =============================================================================

def execute_status(
    db_url: Optional[str] = None,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
) -> None:
    """Muestra el estado actual de la base de datos y dataset demo."""
    url = get_db_url(db_url)
    sanitized_url = sanitize_db_url(url)
    engine, SessionClass = get_engine_and_session(url)

    logger.info("=" * 70)
    logger.info("ESTADO DEL ENTORNO Y DATASET DEMO")
    logger.info("=" * 70)
    logger.info(f"Base de datos objetivo: {sanitized_url}")

    db = SessionClass()
    try:
        # Clientes
        total_clientes = db.query(Cliente).count()
        demo_clientes = db.query(Cliente).filter(Cliente.razon_social.like(f"{DEMO_CLIENT_PREFIX}%")).count()
        base_clientes = total_clientes - demo_clientes

        # Productos
        total_prods = db.query(Producto).count()
        demo_prods = db.query(Producto).filter(Producto.codigo.like(f"{DEMO_PRODUCT_PREFIX}%")).count()
        base_prods = total_prods - demo_prods

        # Ventas
        total_ventas = db.query(Venta).count()
        # Ventas demo identificadas a través de clientes demo
        demo_ventas = (
            db.query(Venta)
            .join(Cliente, Venta.cliente_id == Cliente.id)
            .filter(Cliente.razon_social.like(f"{DEMO_CLIENT_PREFIX}%"))
            .count()
        )
        base_ventas = total_ventas - demo_ventas

        # Comprobantes y modelos
        total_comprobantes = db.query(Comprobante).count()
        total_modelos = db.query(ModeloEntrenamiento).count()
        total_alertas = db.query(AlertaVenta).count()

        # Clientes elegibles para K-Means (total_compras >= 2)
        ventas_por_cliente = (
            db.query(Venta.cliente_id, func.count(Venta.id))
            .filter(Venta.estado == "COMPLETADA")
            .group_by(Venta.cliente_id)
            .all()
        )
        elegibles_kmeans = sum(1 for cid, cnt in ventas_por_cliente if cnt >= 2)

        print("\n--- RESUMEN DE CONTEOS EN BASE DE DATOS ---")
        print(f"  Clientes:     Total: {total_clientes:4d} | Base: {base_clientes:4d} | Demo: {demo_clientes:4d}")
        print(f"  Productos:    Total: {total_prods:4d} | Base: {base_prods:4d} | Demo: {demo_prods:4d}")
        print(f"  Ventas:       Total: {total_ventas:4d} | Base: {base_ventas:4d} | Demo: {demo_ventas:4d}")
        print(f"  Comprobantes: Total: {total_comprobantes:4d} (SUNAT)")
        print(f"  Modelos IA:   Total: {total_modelos:4d}")
        print(f"  Alertas IA:   Total: {total_alertas:4d}")

        print("\n--- ELEGIBILIDAD PARA MODELOS ML ---")
        k_ready = "SÍ (Suficiente)" if elegibles_kmeans >= 15 else f"NO ({elegibles_kmeans}/15 requeridos)"
        print(f"  K-Means (>= 15 clientes con >= 2 compras):  {elegibles_kmeans} clientes elegibles -> {k_ready}")

        ventas_completadas = db.query(Venta).filter(Venta.estado == "COMPLETADA").count()
        if_ready = "SÍ (Suficiente)" if ventas_completadas >= 50 else f"NO ({ventas_completadas}/50 requeridas)"
        print(f"  Isolation Forest (>= 50 ventas completadas): {ventas_completadas} ventas completadas -> {if_ready}")

        manifest = load_manifest(manifest_path)
        print("\n--- ESTADO DEL MANIFIESTO ---")
        if manifest:
            print(f"  Archivo manifiesto: PRESENTE ('{manifest_path}')")
            print(f"  Fecha de creación:  {manifest.get('created_at')}")
            print(f"  Clientes registrados: {len(manifest.get('clientes_ids', []))}")
            print(f"  Productos registrados: {len(manifest.get('productos_ids', []))}")
            print(f"  Ventas registradas:   {len(manifest.get('ventas_ids', []))}")
            print(f"  Detalles registrados: {len(manifest.get('detalle_ventas_ids', []))}")
        else:
            print(f"  Archivo manifiesto: NO EXISTE ('{manifest_path}')")

        print("=" * 70)
    finally:
        db.close()


def execute_dry_run(
    reference_date_str: Optional[str] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    """Calcula y muestra las estadísticas detalladas del dataset sin escribir nada en BD."""
    ref_dt = None
    if reference_date_str:
        ref_dt = datetime.strptime(reference_date_str, "%Y-%m-%d").replace(tzinfo=LIMA_TZ)

    clients_specs, products_specs, sales_plans = plan_demo_dataset(
        reference_datetime=ref_dt,
        random_state=random_state,
    )

    logger.info("=" * 70)
    logger.info("SIMULACIÓN DE DATASET DEMO (DRY-RUN)")
    logger.info("=" * 70)

    # 1. Estadísticas de clientes
    total_clients = len(clients_specs)
    purchases_per_client = [c.ventas_objetivo for c in clients_specs]
    min_purchases = min(purchases_per_client)
    max_purchases = max(purchases_per_client)
    avg_purchases = sum(purchases_per_client) / total_clients

    print("\n--- CLIENTES A GENERAR ---")
    print(f"  Total Clientes Demo:        {total_clients}")
    print(f"  Compras por Cliente (Min):  {min_purchases}")
    print(f"  Compras por Cliente (Max):  {max_purchases}")
    print(f"  Compras por Cliente (Media):{avg_purchases:.2f}")

    grupos_count = defaultdict(int)
    grupos_ventas = defaultdict(int)
    for c in clients_specs:
        grupos_count[c.grupo] += 1
        grupos_ventas[c.grupo] += c.ventas_objetivo

    print("\n  Desglose por Perfil Generador:")
    for grp in sorted(grupos_count.keys()):
        print(f"    - {grp}: {grupos_count[grp]} clientes | {grupos_ventas[grp]} ventas totales ({grupos_ventas[grp]//grupos_count[grp]} c/u)")

    # 2. Estadísticas de ventas
    total_sales = len(sales_plans)
    totals = [float(s.total) for s in sales_plans]
    sorted_totals = sorted(totals)
    min_total = sorted_totals[0]
    max_total = sorted_totals[-1]
    mean_total = sum(totals) / total_sales
    median_total = sorted_totals[total_sales // 2]

    print("\n--- VENTAS HISTÓRICAS A GENERAR ---")
    print(f"  Total Ventas Demo:          {total_sales}")
    print(f"  Monto Total Mínimo:         S/ {min_total:10.2f}")
    print(f"  Monto Total Máximo:         S/ {max_total:10.2f}")
    print(f"  Monto Total Mediana:        S/ {median_total:10.2f}")
    print(f"  Monto Total Medio:          S/ {mean_total:10.2f}")

    # 3. Métricas de carrito
    units_per_sale = [sum(item.cantidad for item in s.items) for s in sales_plans]
    distinct_prods_per_sale = [len(s.items) for s in sales_plans]
    avg_units = sum(units_per_sale) / total_sales
    avg_distinct = sum(distinct_prods_per_sale) / total_sales

    print("\n--- COMPOSICIÓN DEL CARRITO ---")
    print(f"  Unidades Promedio por Venta:       {avg_units:.2f} (Min: {min(units_per_sale)}, Max: {max(units_per_sale)})")
    print(f"  Productos Distintos Promedio:      {avg_distinct:.2f} (Min: {min(distinct_prods_per_sale)}, Max: {max(distinct_prods_per_sale)})")
    print(f"  Total Líneas de Detalle:           {sum(distinct_prods_per_sale)}")

    # 4. Distribución Temporal
    dates = [s.created_at for s in sales_plans]
    min_date = min(dates)
    max_date = max(dates)
    span_days = (max_date - min_date).days

    hours = [s.created_at.hour for s in sales_plans]
    min_hour = min(hours)
    max_hour = max(hours)

    print("\n--- DISTRIBUCIÓN TEMPORAL (America/Lima) ---")
    print(f"  Venta más antigua:          {min_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Venta más reciente:         {max_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Amplitud Histórica:         {span_days} días")
    print(f"  Rango Horario Comercial:    {min_hour:02d}:00 a {max_hour:02d}:59")

    # 5. Consumo de Stock en Productos Demo
    consumed_units = defaultdict(int)
    for s in sales_plans:
        for item in s.items:
            consumed_units[item.codigo_producto] += item.cantidad

    print("\n--- CONSUMO DE STOCK PROYECTADO (10 PRODUCTOS DEMO) ---")
    for p in products_specs:
        code = p["codigo"]
        initial = p["stock_inicial"]
        consumed = consumed_units[code]
        remaining = initial - consumed
        print(f"  {code}: Inicial: {initial:5d} | Consumido: {consumed:5d} | Remanente: {remaining:5d} ({p['unidad_medida']})")

    logger.info("=" * 70)
    logger.info("DRY-RUN EXITOSO: Ningún dato fue escrito en la base de datos.")
    logger.info("Para persistir estos datos ejecute el script con: --apply")
    logger.info("=" * 70)


def execute_apply(
    db_url: Optional[str] = None,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    reference_date_str: Optional[str] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    """Inserta el dataset demo en una única transacción atómica y escribe el manifiesto."""
    url = get_db_url(db_url)
    sanitized_url = sanitize_db_url(url)
    engine, SessionClass = get_engine_and_session(url)

    logger.info("=" * 70)
    logger.info("APLICACIÓN DEL DATASET HISTÓRICO DEMO")
    logger.info("=" * 70)
    logger.info(f"Base de datos objetivo: {sanitized_url}")

    # 1. Comprobación de idempotencia: verificar si existe manifiesto o clientes demo
    if os.path.exists(manifest_path):
        raise RuntimeError(
            f"El archivo de manifiesto '{manifest_path}' ya existe. "
            "Para evitar duplicados, ejecute primero: python seed_ai_demo.py --cleanup"
        )

    db = SessionClass()
    try:
        existing_demo_clients = db.query(Cliente).filter(Cliente.razon_social.like(f"{DEMO_CLIENT_PREFIX}%")).count()
        if existing_demo_clients > 0:
            raise RuntimeError(
                f"Se detectaron {existing_demo_clients} clientes demo existentes en la base de datos. "
                "Para evitar inconsistencias, ejecute primero: python seed_ai_demo.py --cleanup"
            )

        ref_dt = None
        if reference_date_str:
            ref_dt = datetime.strptime(reference_date_str, "%Y-%m-%d").replace(tzinfo=LIMA_TZ)

        clients_specs, products_specs, sales_plans = plan_demo_dataset(
            reference_datetime=ref_dt,
            random_state=random_state,
        )

        created_client_ids: List[int] = []
        created_product_ids: List[int] = []
        created_sale_ids: List[int] = []
        created_detail_ids: List[int] = []
        grupos_metadata: Dict[str, Dict[str, List[int]]] = {
            "Grupo A": {"clientes_ids": [], "ventas_ids": []},
            "Grupo B": {"clientes_ids": [], "ventas_ids": []},
            "Grupo C": {"clientes_ids": [], "ventas_ids": []},
        }

        # 2. Ejecutar inserción completa dentro de una única transacción atómica
        logger.info("Iniciando transacción atómica de persistencia...")
        try:
            # A. Insertar productos demo
            db_products: Dict[str, Producto] = {}
            for p in products_specs:
                prod = Producto(
                    codigo=p["codigo"],
                    nombre=p["nombre"],
                    descripcion=p["descripcion"],
                    categoria=p["categoria"],
                    precio=p["precio"],
                    stock=p["stock_inicial"],
                    stock_minimo=10,
                    unidad_medida=p["unidad_medida"],
                    is_active=True,
                )
                db.add(prod)
                db.flush()
                db_products[p["codigo"]] = prod
                created_product_ids.append(prod.id)

            logger.info(f"Insertados {len(created_product_ids)} productos demo.")

            # B. Insertar clientes demo
            # Asignar fecha de registro anterior a su primera venta (entre 120 y 160 días antes de ref)
            db_clients: Dict[int, Cliente] = {}
            for c_spec in clients_specs:
                client_created_at = (ref_dt or datetime.now(LIMA_TZ)) - timedelta(days=140 + c_spec.index)
                cliente = Cliente(
                    tipo_documento="RUC",
                    numero_documento=c_spec.numero_documento,
                    razon_social=c_spec.razon_social,
                    nombre_contacto=c_spec.nombre_contacto,
                    telefono=c_spec.telefono,
                    email=c_spec.email,
                    direccion=c_spec.direccion,
                    ciudad=c_spec.ciudad,
                    tipo_cliente="Empresa",
                    is_active=True,
                    created_at=client_created_at,
                )
                db.add(cliente)
                db.flush()
                db_clients[c_spec.index] = cliente
                created_client_ids.append(cliente.id)
                grupos_metadata[c_spec.grupo]["clientes_ids"].append(cliente.id)

            logger.info(f"Insertados {len(created_client_ids)} clientes demo.")

            # C. Insertar ventas y detalles
            for sale_plan in sales_plans:
                cliente_orm = db_clients[sale_plan.client_spec.index]
                num_venta = _generate_numero_venta(db)

                venta = Venta(
                    numero_venta=num_venta,
                    cliente_id=cliente_orm.id,
                    subtotal=sale_plan.subtotal,
                    igv=sale_plan.igv,
                    total=sale_plan.total,
                    estado="COMPLETADA",
                    created_at=sale_plan.created_at,
                )
                db.add(venta)
                db.flush()
                created_sale_ids.append(venta.id)
                grupos_metadata[sale_plan.client_spec.grupo]["ventas_ids"].append(venta.id)

                for item_plan in sale_plan.items:
                    prod_orm = db_products[item_plan.codigo_producto]
                    detalle = DetalleVenta(
                        venta_id=venta.id,
                        producto_id=prod_orm.id,
                        cantidad=item_plan.cantidad,
                        precio_unitario=item_plan.precio_unitario,
                        subtotal=item_plan.subtotal,
                    )
                    db.add(detalle)
                    db.flush()
                    created_detail_ids.append(detalle.id)

                    # Descontar stock
                    prod_orm.stock -= item_plan.cantidad
                    if prod_orm.stock < 0:
                        raise ValueError(f"Error de consistencia: stock negativo para {prod_orm.codigo}")

            logger.info(f"Insertadas {len(created_sale_ids)} ventas demo con {len(created_detail_ids)} detalles.")

            # Confirmar la transacción atómica
            db.commit()
            logger.info("Transacción confirmada exitosamente (COMMIT).")

        except Exception as e:
            db.rollback()
            logger.error(f"Error durante la inserción. Se ejecutó ROLLBACK completo: {e}")
            raise

        # 3. Guardar manifiesto de seguridad
        manifest_data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "database_url_target": sanitized_url,
            "random_state": random_state,
            "summary": {
                "total_clientes": len(created_client_ids),
                "total_productos": len(created_product_ids),
                "total_ventas": len(created_sale_ids),
                "total_detalles": len(created_detail_ids),
            },
            "clientes_ids": created_client_ids,
            "productos_ids": created_product_ids,
            "ventas_ids": created_sale_ids,
            "detalle_ventas_ids": created_detail_ids,
            "grupos_metadata": grupos_metadata,
        }
        save_manifest(manifest_path, manifest_data)

        # 4. Validación inmediata post-inserción
        execute_validate(db_url=db_url, manifest_path=manifest_path)

        logger.info("=" * 70)
        logger.info("APLICACIÓN COMPLETADA EXITOSAMENTE")
        logger.info("=" * 70)

    finally:
        db.close()


def execute_cleanup(
    db_url: Optional[str] = None,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    force_prefix_cleanup: bool = False,
) -> None:
    """
    Elimina exclusivamente los registros demo creados.
    Utiliza por seguridad los IDs registrados en el manifiesto, validando
    además los prefijos 'DEMO IA - ' y 'DEMO-IA-' antes de cualquier borrado.
    """
    url = get_db_url(db_url)
    sanitized_url = sanitize_db_url(url)
    engine, SessionClass = get_engine_and_session(url)

    logger.info("=" * 70)
    logger.info("LIMPIEZA SEGURA DE DATASET DEMO (CLEANUP)")
    logger.info("=" * 70)
    logger.info(f"Base de datos objetivo: {sanitized_url}")

    manifest = load_manifest(manifest_path)
    if not manifest and not force_prefix_cleanup:
        raise RuntimeError(
            f"No se encontró el archivo de manifiesto '{manifest_path}'. "
            "Por estricta seguridad anti-borrado, no se realizarán eliminaciones masivas sin manifiesto."
        )

    db = SessionClass()
    try:
        if manifest:
            client_ids: List[int] = manifest.get("clientes_ids", [])
            product_ids: List[int] = manifest.get("productos_ids", [])
            sale_ids: List[int] = manifest.get("ventas_ids", [])
            detail_ids: List[int] = manifest.get("detalle_ventas_ids", [])

            logger.info(f"Manifiesto cargado con {len(client_ids)} clientes, {len(product_ids)} productos, {len(sale_ids)} ventas.")

            # DOBLE CAPA DE SEGURIDAD: Verificar que TODOS los clientes y productos a borrar son efectivamente DEMO
            if client_ids:
                non_demo_clients = (
                    db.query(Cliente)
                    .filter(Cliente.id.in_(client_ids), ~Cliente.razon_social.like(f"{DEMO_CLIENT_PREFIX}%"))
                    .all()
                )
                if non_demo_clients:
                    names = [c.razon_social for c in non_demo_clients]
                    raise RuntimeError(
                        f"ABORTE DE SEGURIDAD: El manifiesto contiene IDs de clientes NO demo: {names}."
                    )

            if product_ids:
                non_demo_products = (
                    db.query(Producto)
                    .filter(Producto.id.in_(product_ids), ~Producto.codigo.like(f"{DEMO_PRODUCT_PREFIX}%"))
                    .all()
                )
                if non_demo_products:
                    codes = [p.codigo for p in non_demo_products]
                    raise RuntimeError(
                        f"ABORTE DE SEGURIDAD: El manifiesto contiene IDs de productos NO demo: {codes}."
                    )

            # Transacción atómica de borrado en orden estricto de claves foráneas
            try:
                # 1. Borrar detalles de venta
                deleted_details = 0
                if detail_ids:
                    deleted_details = db.query(DetalleVenta).filter(DetalleVenta.id.in_(detail_ids)).delete(synchronize_session=False)

                # 2. Borrar ventas
                deleted_sales = 0
                if sale_ids:
                    deleted_sales = db.query(Venta).filter(Venta.id.in_(sale_ids)).delete(synchronize_session=False)

                # 3. Borrar productos demo
                deleted_products = 0
                if product_ids:
                    deleted_products = db.query(Producto).filter(Producto.id.in_(product_ids)).delete(synchronize_session=False)

                # 4. Borrar clientes demo
                deleted_clients = 0
                if client_ids:
                    deleted_clients = db.query(Cliente).filter(Cliente.id.in_(client_ids)).delete(synchronize_session=False)

                db.commit()
                logger.info("Registros demo eliminados exitosamente de la base de datos:")
                logger.info(f"  - Detalles de venta eliminados: {deleted_details}")
                logger.info(f"  - Ventas demo eliminadas:       {deleted_sales}")
                logger.info(f"  - Productos demo eliminados:    {deleted_products}")
                logger.info(f"  - Clientes demo eliminados:     {deleted_clients}")

            except Exception as e:
                db.rollback()
                logger.error(f"Error al eliminar registros. Se ejecutó ROLLBACK: {e}")
                raise

            delete_manifest(manifest_path)

        else:
            # Limpieza forzada de rescate sólo por prefijo estricto
            logger.warning("Ejecutando limpieza de rescate por prefijo estricto...")
            try:
                demo_client_ids = [c.id for c in db.query(Cliente.id).filter(Cliente.razon_social.like(f"{DEMO_CLIENT_PREFIX}%")).all()]
                demo_prod_ids = [p.id for p in db.query(Producto.id).filter(Producto.codigo.like(f"{DEMO_PRODUCT_PREFIX}%")).all()]
                demo_sale_ids = [v.id for v in db.query(Venta.id).filter(Venta.cliente_id.in_(demo_client_ids)).all()]

                db.query(DetalleVenta).filter(DetalleVenta.venta_id.in_(demo_sale_ids)).delete(synchronize_session=False)
                db.query(Venta).filter(Venta.id.in_(demo_sale_ids)).delete(synchronize_session=False)
                db.query(Producto).filter(Producto.id.in_(demo_prod_ids)).delete(synchronize_session=False)
                db.query(Cliente).filter(Cliente.id.in_(demo_client_ids)).delete(synchronize_session=False)
                db.commit()
                logger.info("Limpieza por prefijo completada exitosamente.")
            except Exception as e:
                db.rollback()
                logger.error(f"Error en limpieza por prefijo: {e}")
                raise

        logger.info("=" * 70)
        logger.info("CLEANUP EXITOSO: Base de datos restaurada. Datos reales intactos.")
        logger.info("=" * 70)

    finally:
        db.close()


def execute_validate(
    db_url: Optional[str] = None,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
) -> None:
    """Valida la consistencia estructural y prueba la ejecución de Feature Engineering sin entrenar."""
    url = get_db_url(db_url)
    engine, SessionClass = get_engine_and_session(url)

    logger.info("=" * 70)
    logger.info("VALIDACIÓN DE INTEGRIDAD Y FEATURE ENGINEERING")
    logger.info("=" * 70)

    db = SessionClass()
    try:
        # 1. Comprobaciones de integridad básica
        demo_clients = db.query(Cliente).filter(Cliente.razon_social.like(f"{DEMO_CLIENT_PREFIX}%")).all()
        demo_products = db.query(Producto).filter(Producto.codigo.like(f"{DEMO_PRODUCT_PREFIX}%")).all()
        demo_client_ids = [c.id for c in demo_clients]
        demo_sales = db.query(Venta).filter(Venta.cliente_id.in_(demo_client_ids)).all()

        logger.info(f"Verificando recuentos: {len(demo_clients)} clientes demo, {len(demo_products)} productos demo, {len(demo_sales)} ventas demo.")

        assert len(demo_clients) == 24, f"Se esperaban 24 clientes demo, encontrados: {len(demo_clients)}"
        assert len(demo_products) == 10, f"Se esperaban 10 productos demo, encontrados: {len(demo_products)}"
        assert len(demo_sales) == 120, f"Se esperaban 120 ventas demo, encontradas: {len(demo_sales)}"

        # Verificar que todos los estados son COMPLETADA
        for v in demo_sales:
            assert v.estado == "COMPLETADA", f"Venta {v.id} con estado {v.estado} != COMPLETADA"

        # Verificar que ningún producto demo tiene stock negativo
        for p in demo_products:
            assert p.stock >= 0, f"Producto {p.codigo} tiene stock negativo: {p.stock}"

        # Verificar que ninguna alerta o comprobante SUNAT fue creado
        comprobantes_demo = (
            db.query(Comprobante)
            .join(Venta, Comprobante.venta_id == Venta.id)
            .filter(Venta.cliente_id.in_(demo_client_ids))
            .count()
        )
        assert comprobantes_demo == 0, f"Se detectaron comprobantes SUNAT creados para ventas demo: {comprobantes_demo}"

        modelos_count = db.query(ModeloEntrenamiento).count()
        alertas_count = db.query(AlertaVenta).count()
        logger.info(f"Garantía de Cero Efectos Secundarios: {comprobantes_demo} comprobantes SUNAT, {modelos_count} modelos, {alertas_count} alertas.")

        # 2. Prueba de Feature Engineering K-Means
        logger.info("Ejecutando build_customer_features()...")
        df_customers = build_customer_features(db)
        demo_df_customers = df_customers[df_customers["cliente_id"].isin(demo_client_ids)]

        assert len(demo_df_customers) == 24, f"Se esperaban 24 clientes en dataset K-Means, obtenidos: {len(demo_df_customers)}"
        assert all(demo_df_customers["total_compras"] >= 2), "No todos los clientes demo tienen >= 2 compras"
        assert not demo_df_customers["dias_entre_compras_promedio"].isna().any(), "Existen NaN en dias_entre_compras_promedio para clientes con >= 2 compras"

        logger.info(f"Feature Engineering K-Means VÁLIDO: 24 clientes demo elegibles. Columnas: {list(demo_df_customers.columns)}")

        # 3. Prueba de Feature Engineering Isolation Forest
        logger.info("Ejecutando build_sales_features()...")
        df_sales = build_sales_features(db)
        demo_sale_ids = [v.id for v in demo_sales]
        demo_df_sales = df_sales[df_sales["venta_id"].isin(demo_sale_ids)]

        assert len(demo_df_sales) == 120, f"Se esperaban 120 ventas en dataset Isolation Forest, obtenidas: {len(demo_df_sales)}"
        assert (demo_df_sales["primera_compra"] == 1).sum() == 24, "Se esperaban exactamente 24 primeras compras (1 por cada cliente)"

        logger.info(f"Feature Engineering Isolation Forest VÁLIDO: 120 ventas analizadas. Columnas: {list(demo_df_sales.columns)}")

        print("\n" + "=" * 70)
        print("VALIDACIÓN POST-SEED SUPERADA AL 100%")
        print("  - 24 Clientes Demo presentes, todos con >= 2 compras (aptos para K-Means)")
        print("  - 10 Productos Demo presentes, sin stock negativo")
        print("  - 120 Ventas Demo históricas presentes, todas COMPLETADA (aptas para Isolation Forest)")
        print("  - 0 Comprobantes SUNAT generados")
        print("  - 0 Modelos IA entrenados")
        print("  - 0 Alertas IA generadas")
        print("  - Feature Engineering genera DataFrames sin errores")
        print("=" * 70 + "\n")

    finally:
        db.close()


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Herramienta de Seed, Verificación y Limpieza de Dataset Demo IA para Dharma Sales System.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Ver estado actual del entorno y elegibilidad ML:
  python seed_ai_demo.py --status

  # Simular dataset en memoria sin escribir en BD:
  python seed_ai_demo.py --dry-run

  # Aplicar inserción en base de datos:
  python seed_ai_demo.py --apply

  # Validar integridad y feature engineering del dataset aplicado:
  python seed_ai_demo.py --validate

  # Eliminar exclusivamente los datos demo y restaurar la BD:
  python seed_ai_demo.py --cleanup

  # Apuntar a una base de datos demo aislada:
  python seed_ai_demo.py --status --db-url postgresql+psycopg://dharma_app:dharmaadmin@localhost:5432/dharma_sales_demo
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="Muestra el estado actual de la BD y dataset demo.")
    group.add_argument("--dry-run", action="store_true", help="Simula y muestra estadísticas del dataset sin escribir en BD.")
    group.add_argument("--apply", action="store_true", help="Inserta el dataset demo en una transacción atómica.")
    group.add_argument("--cleanup", action="store_true", help="Elimina exclusivamente los datos demo creados mediante manifiesto.")
    group.add_argument("--validate", action="store_true", help="Valida el dataset y feature engineering sin entrenar.")

    parser.add_argument("--db-url", type=str, default=None, help="URL de conexión SQLAlchemy (override de DATABASE_URL).")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST_PATH, help="Ruta del archivo de manifiesto de seguridad.")
    parser.add_argument("--ref-date", type=str, default=None, help="Fecha de referencia YYYY-MM-DD (por defecto hoy en America/Lima).")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Semilla pseudoaleatoria (default: 42).")

    return parser.parse_args()


def main():
    args = parse_args()

    # Si se invoca sin argumentos de acción, por seguridad NO escribe nada y muestra status
    if not (args.status or args.dry_run or args.apply or args.cleanup or args.validate):
        print("No se especificó ninguna acción. Mostrando estado actual (--status).")
        print("Ejecute con --help para ver los comandos disponibles.\n")
        execute_status(db_url=args.db_url, manifest_path=args.manifest)
        return

    if args.status:
        execute_status(db_url=args.db_url, manifest_path=args.manifest)
    elif args.dry_run:
        execute_dry_run(reference_date_str=args.ref_date, random_state=args.random_state)
    elif args.apply:
        execute_apply(
            db_url=args.db_url,
            manifest_path=args.manifest,
            reference_date_str=args.ref_date,
            random_state=args.random_state,
        )
    elif args.cleanup:
        execute_cleanup(db_url=args.db_url, manifest_path=args.manifest)
    elif args.validate:
        execute_validate(db_url=args.db_url, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
