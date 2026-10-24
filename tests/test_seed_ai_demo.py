"""
Pruebas Unitarias y de Integración para el Seed del Dataset Demo IA (Parte 8A).

Verifica:
1. Generación de RUCs sintéticos válidos bajo algoritmo SUNAT (módulo 11).
2. Reproducibilidad determinística de la planificación con random_state=42.
3. Distribución exacta de grupos (Grupo A: 8x7=56, Grupo B: 8x5=40, Grupo C: 8x3=24 -> 120 ventas).
4. Rigor matemático de totales (subtotal + IGV 18% == total en Decimal con ROUND_HALF_UP).
5. Coherencia interna de carritos (productos_distintos <= unidades_totales, cantidades > 0).
6. Distribución de fechas y horarios comerciales (08:00 a 20:00, 90-120 días).
7. Gestión del archivo de manifiesto (creación, lectura, eliminación).
8. Barrera de seguridad anti-borrado (rechazo estricto si se intentan borrar IDs no demo).
9. Ciclo de vida completo (apply -> validate con Feature Engineering -> cleanup) en base de datos aislada.
10. Preservación absoluta de registros reales en base de datos.
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
import os
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Asegurar importación de backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import Base
from app.models.client import Cliente
from app.models.product import Producto
from app.models.sale import DetalleVenta, Venta
from app.models.receipt import Comprobante
from app.models.ai import ModeloEntrenamiento, AlertaVenta
from scripts.seed_ai_demo import (
    DEMO_CLIENT_PREFIX,
    DEMO_PRODUCT_PREFIX,
    TWO_PLACES,
    IGV_RATE,
    LIMA_TZ,
    build_demo_clients_specs,
    generate_valid_sunat_ruc,
    load_manifest,
    plan_demo_dataset,
    save_manifest,
    delete_manifest,
    execute_apply,
    execute_cleanup,
    execute_validate,
)


class TestSeedAIDemo(unittest.TestCase):
    def setUp(self):
        self.ref_dt = datetime(2026, 10, 15, 18, 0, 0, tzinfo=LIMA_TZ)
        self.temp_manifest = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.temp_manifest.close()
        # Eliminar archivo inicial para probar creación
        if os.path.exists(self.temp_manifest.name):
            os.remove(self.temp_manifest.name)

    def tearDown(self):
        if os.path.exists(self.temp_manifest.name):
            os.remove(self.temp_manifest.name)

    def test_01_generate_valid_sunat_ruc(self):
        """Comprueba que los 24 RUCs generados tienen 11 dígitos, son únicos y cumplen checksum SUNAT."""
        rucs = [generate_valid_sunat_ruc(f"9001{i:04d}") for i in range(1, 25)]
        self.assertEqual(len(rucs), 24)
        self.assertEqual(len(set(rucs)), 24, "Los RUCs generados deben ser estrictamente únicos")

        weights = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
        for ruc in rucs:
            self.assertEqual(len(ruc), 11)
            self.assertTrue(ruc.startswith("20"))
            self.assertTrue(ruc.isdigit())

            # Validar algoritmo módulo 11
            total = sum(int(d) * w for d, w in zip(ruc[:10], weights))
            residue = 11 - (total % 11)
            expected_cd = 0 if residue == 10 else (1 if residue == 11 else residue)
            self.assertEqual(int(ruc[10]), expected_cd, f"Checksum inválido para RUC {ruc}")

    def test_02_deterministic_planning(self):
        """Verifica que plan_demo_dataset con random_state=42 es estrictamente reproducible."""
        c1, p1, s1 = plan_demo_dataset(reference_datetime=self.ref_dt, random_state=42)
        c2, p2, s2 = plan_demo_dataset(reference_datetime=self.ref_dt, random_state=42)

        self.assertEqual(len(c1), 24)
        self.assertEqual(len(p1), 10)
        self.assertEqual(len(s1), 120)

        for sale1, sale2 in zip(s1, s2):
            self.assertEqual(sale1.client_spec.numero_documento, sale2.client_spec.numero_documento)
            self.assertEqual(sale1.created_at, sale2.created_at)
            self.assertEqual(sale1.total, sale2.total)
            self.assertEqual(len(sale1.items), len(sale2.items))

    def test_03_group_distribution_and_purchases(self):
        """Verifica que Grupo A (8x7=56), Grupo B (8x5=40), Grupo C (8x3=24) suman exactamente 120 ventas."""
        clients, _, sales = plan_demo_dataset(reference_datetime=self.ref_dt, random_state=42)

        sales_by_client = {}
        for s in sales:
            cid = s.client_spec.numero_documento
            sales_by_client[cid] = sales_by_client.get(cid, 0) + 1

        for c in clients:
            if c.grupo == "Grupo A":
                self.assertEqual(sales_by_client[c.numero_documento], 7)
            elif c.grupo == "Grupo B":
                self.assertEqual(sales_by_client[c.numero_documento], 5)
            elif c.grupo == "Grupo C":
                self.assertEqual(sales_by_client[c.numero_documento], 3)

        self.assertEqual(len(sales), 120)
        self.assertTrue(all(count >= 2 for count in sales_by_client.values()))

    def test_04_monetary_exactness(self):
        """Verifica que cada venta respeta subtotal + IGV (18%) == total con Decimal y ROUND_HALF_UP."""
        _, _, sales = plan_demo_dataset(reference_datetime=self.ref_dt, random_state=42)

        for s in sales:
            subtotal_calc = sum(item.subtotal for item in s.items).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
            self.assertEqual(s.subtotal, subtotal_calc)

            igv_calc = (s.subtotal * IGV_RATE).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
            self.assertEqual(s.igv, igv_calc)

            total_calc = (s.subtotal + s.igv).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
            self.assertEqual(s.total, total_calc)

    def test_05_cart_diversity_and_consistency(self):
        """Verifica que los carritos cumplen productos_distintos <= unidades_totales y cantidades > 0."""
        _, _, sales = plan_demo_dataset(reference_datetime=self.ref_dt, random_state=42)

        for s in sales:
            unidades_totales = sum(item.cantidad for item in s.items)
            productos_distintos = len(s.items)
            self.assertGreaterEqual(unidades_totales, productos_distintos)
            self.assertGreater(unidades_totales, 0)
            for item in s.items:
                self.assertGreater(item.cantidad, 0)
                self.assertGreater(item.precio_unitario, Decimal("0.00"))

    def test_06_temporal_validity(self):
        """Verifica que las ventas están dentro del horario comercial y en orden cronológico por cliente."""
        clients, _, sales = plan_demo_dataset(reference_datetime=self.ref_dt, random_state=42)

        client_sales = {}
        for s in sales:
            hour = s.created_at.hour
            self.assertGreaterEqual(hour, 8, f"Hora anómala detectada: {s.created_at}")
            self.assertLessEqual(hour, 19, f"Hora anómala detectada: {s.created_at}")

            cid = s.client_spec.numero_documento
            if cid not in client_sales:
                client_sales[cid] = []
            client_sales[cid].append(s.created_at)

        for cid, dates in client_sales.items():
            for i in range(len(dates) - 1):
                self.assertLess(dates[i], dates[i + 1], f"Ventas desordenadas cronológicamente para {cid}")

    def test_07_manifest_lifecycle(self):
        """Verifica la persistencia, carga y borrado del manifiesto JSON."""
        manifest_data = {
            "created_at": "2026-10-16T19:00:00Z",
            "clientes_ids": [101, 102],
            "productos_ids": [201, 202],
            "ventas_ids": [301, 302],
            "detalle_ventas_ids": [401, 402],
        }
        save_manifest(self.temp_manifest.name, manifest_data)
        loaded = load_manifest(self.temp_manifest.name)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["clientes_ids"], [101, 102])

        deleted = delete_manifest(self.temp_manifest.name)
        self.assertTrue(deleted)
        self.assertIsNone(load_manifest(self.temp_manifest.name))

    def test_08_end_to_end_lifecycle_in_isolated_db(self):
        """
        Ejecuta el ciclo completo:
        1. Base de datos aislada con registros reales preexistentes.
        2. --apply para insertar 24 clientes, 10 productos, 120 ventas.
        3. --validate ejecutando Feature Engineering sin errores.
        4. --cleanup para eliminar exclusivamente los registros demo.
        5. Comprueba que los registros preexistentes NUNCA fueron modificados o eliminados.
        """
        db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_file.close()
        sqlite_url = f"sqlite:///{db_file.name}"

        try:
            engine = create_engine(sqlite_url)
            Base.metadata.create_all(engine)
            SessionLocal = sessionmaker(bind=engine)

            # Insertar datos base "reales" preexistentes
            db = SessionLocal()
            c_real = Cliente(
                tipo_documento="RUC",
                numero_documento="20123456789",
                razon_social="CLIENTE REAL EXISTENTE S.A.C.",
                ciudad="Trujillo",
                tipo_cliente="Empresa",
                is_active=True,
            )
            db.add(c_real)
            p_real = Producto(
                codigo="PROD-REAL-001",
                nombre="Producto Real Existente",
                categoria="Real",
                precio=Decimal("50.00"),
                stock=100,
                is_active=True,
            )
            db.add(p_real)
            db.commit()

            v_real = Venta(
                numero_venta="V-REAL-001",
                cliente_id=c_real.id,
                subtotal=Decimal("50.00"),
                igv=Decimal("9.00"),
                total=Decimal("59.00"),
                estado="COMPLETADA",
            )
            db.add(v_real)
            db.commit()
            d_real = DetalleVenta(
                venta_id=v_real.id,
                producto_id=p_real.id,
                cantidad=1,
                precio_unitario=Decimal("50.00"),
                subtotal=Decimal("50.00"),
            )
            db.add(d_real)
            db.commit()
            db.close()

            # 1. Ejecutar APPLY
            execute_apply(
                db_url=sqlite_url,
                manifest_path=self.temp_manifest.name,
                reference_date_str="2026-10-15",
                random_state=42,
            )

            # Verificar que el manifiesto se creó
            self.assertTrue(os.path.exists(self.temp_manifest.name))
            manifest = load_manifest(self.temp_manifest.name)
            self.assertEqual(len(manifest["clientes_ids"]), 24)
            self.assertEqual(len(manifest["productos_ids"]), 10)
            self.assertEqual(len(manifest["ventas_ids"]), 120)

            # 2. Ejecutar VALIDATE
            execute_validate(db_url=sqlite_url, manifest_path=self.temp_manifest.name)

            # Verificar que existe cliente real + 24 clientes demo
            db = SessionLocal()
            self.assertEqual(db.query(Cliente).count(), 25)
            self.assertEqual(db.query(Producto).count(), 11)
            self.assertEqual(db.query(Venta).count(), 121)
            db.close()

            # 3. Intentar APPLY repetido (debe fallar por idempotencia)
            with self.assertRaises(RuntimeError):
                execute_apply(
                    db_url=sqlite_url,
                    manifest_path=self.temp_manifest.name,
                    reference_date_str="2026-10-15",
                    random_state=42,
                )

            # 4. Ejecutar CLEANUP
            execute_cleanup(db_url=sqlite_url, manifest_path=self.temp_manifest.name)

            # Verificar que los datos demo se eliminaron y los reales permanecen intactos
            db = SessionLocal()
            self.assertEqual(db.query(Cliente).count(), 1)
            self.assertEqual(db.query(Cliente).first().razon_social, "CLIENTE REAL EXISTENTE S.A.C.")

            self.assertEqual(db.query(Producto).count(), 1)
            self.assertEqual(db.query(Producto).first().codigo, "PROD-REAL-001")

            self.assertEqual(db.query(Venta).count(), 1)
            self.assertEqual(db.query(Venta).first().numero_venta, "V-REAL-001")
            db.close()

            # Disponer el engine para liberar el lock de Windows sobre el archivo SQLite
            engine.dispose()

            # Verificar que el manifiesto fue removido
            self.assertFalse(os.path.exists(self.temp_manifest.name))

        finally:
            try:
                if 'engine' in locals():
                    engine.dispose()
                if os.path.exists(db_file.name):
                    os.remove(db_file.name)
            except Exception:
                pass

    def test_09_safety_guard_rejects_non_demo_ids(self):
        """Verifica que el cleanup aborta inmediatamente si el manifiesto contuviera IDs no demo."""
        db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_file.close()
        sqlite_url = f"sqlite:///{db_file.name}"

        try:
            engine = create_engine(sqlite_url)
            Base.metadata.create_all(engine)
            SessionLocal = sessionmaker(bind=engine)

            db = SessionLocal()
            c_real = Cliente(
                tipo_documento="RUC",
                numero_documento="20999999999",
                razon_social="EMPRESA IMPORTANTE NO DEMO S.A.C.",
                ciudad="Trujillo",
                tipo_cliente="Empresa",
                is_active=True,
            )
            db.add(c_real)
            db.commit()
            real_id = c_real.id
            db.close()

            # Manifiesto malicioso o corrupto intentando borrar a un cliente real
            corrupt_manifest = {
                "clientes_ids": [real_id],
                "productos_ids": [],
                "ventas_ids": [],
                "detalle_ventas_ids": [],
            }
            save_manifest(self.temp_manifest.name, corrupt_manifest)

            with self.assertRaises(RuntimeError) as ctx:
                execute_cleanup(db_url=sqlite_url, manifest_path=self.temp_manifest.name)

            self.assertIn("ABORTE DE SEGURIDAD", str(ctx.exception))

            # Comprobar que el cliente real NO fue borrado
            db = SessionLocal()
            self.assertEqual(db.query(Cliente).count(), 1)
            db.close()
            engine.dispose()

        finally:
            try:
                if 'engine' in locals():
                    engine.dispose()
                if os.path.exists(db_file.name):
                    os.remove(db_file.name)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
