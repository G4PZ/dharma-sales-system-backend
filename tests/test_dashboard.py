"""
Pruebas Unitarias e Integración para el Dashboard Dharma (Business Intelligence).
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import os
import sys
import unittest
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

# Asegurar backend en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.routes.auth import get_current_user

LIMA_TZ = ZoneInfo("America/Lima")


class TestDashboardRoute(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_dashboard_unauthorized_without_session(self):
        """Verifica que /api/dashboard/resumen rechace peticiones sin autenticación (HTTP 401)."""
        response = self.client.get("/api/dashboard/resumen")
        self.assertEqual(response.status_code, 401)

    def test_dashboard_default_range_last_7_days(self):
        """Verifica que sin parámetros, el dashboard use por defecto los últimos 7 días."""
        app.dependency_overrides[get_current_user] = lambda: {
            "email": "test@dharma.com",
            "name": "Usuario Test",
        }

        try:
            response = self.client.get("/api/dashboard/resumen")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Verificación de rango por defecto
            self.assertIn("rango_fechas", data)
            self.assertEqual(data["rango_fechas"]["dias_totales"], 7)

            today_lima = datetime.now(LIMA_TZ).date()
            expected_desde = (today_lima - timedelta(days=6)).isoformat()
            expected_hasta = today_lima.isoformat()
            self.assertEqual(data["rango_fechas"]["fecha_desde"], expected_desde)
            self.assertEqual(data["rango_fechas"]["fecha_hasta"], expected_hasta)

            # Verificación de KPIs
            self.assertIn("ventas", data)
            self.assertIn("cantidad", data["ventas"])
            self.assertIn("ingresos", data["ventas"])
            self.assertIn("ventas_hoy", data)

            self.assertIn("comprobantes", data)
            self.assertIn("alertas_ia", data)

            # Verificación de evolución de 7 días
            self.assertIn("evolucion_ventas", data)
            self.assertEqual(len(data["evolucion_ventas"]), 7)

            # Verificación de tablas y widgets recientes
            self.assertIn("alertas_recientes", data)
            self.assertIn("ventas_recientes", data)
            self.assertIsInstance(data["alertas_recientes"], list)
            self.assertIsInstance(data["ventas_recientes"], list)

        finally:
            app.dependency_overrides.clear()

    def test_dashboard_custom_range_success(self):
        """Verifica que con un rango personalizado válido, el dashboard calcule los días exactos y filtre."""
        app.dependency_overrides[get_current_user] = lambda: {
            "email": "test@dharma.com",
            "name": "Usuario Test",
        }

        try:
            # Rango de 10 días: 2026-10-01 al 2026-10-10
            response = self.client.get(
                "/api/dashboard/resumen?fecha_desde=2026-10-01&fecha_hasta=2026-10-10"
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertEqual(data["rango_fechas"]["fecha_desde"], "2026-10-01")
            self.assertEqual(data["rango_fechas"]["fecha_hasta"], "2026-10-10")
            self.assertEqual(data["rango_fechas"]["dias_totales"], 10)
            self.assertEqual(len(data["evolucion_ventas"]), 10)

        finally:
            app.dependency_overrides.clear()

    def test_dashboard_invalid_range_raises_400(self):
        """Verifica que fecha_desde > fecha_hasta devuelva error controlado HTTP 400 Bad Request."""
        app.dependency_overrides[get_current_user] = lambda: {
            "email": "test@dharma.com",
            "name": "Usuario Test",
        }

        try:
            response = self.client.get(
                "/api/dashboard/resumen?fecha_desde=2026-10-25&fecha_hasta=2026-10-20"
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn("detail", data)
            self.assertIn("no puede ser posterior", data["detail"])

        finally:
            app.dependency_overrides.clear()

    def test_dashboard_single_day_range(self):
        """Verifica que un rango de un solo día (fecha_desde == fecha_hasta) funcione y contenga 1 día."""
        app.dependency_overrides[get_current_user] = lambda: {
            "email": "test@dharma.com",
            "name": "Usuario Test",
        }

        try:
            response = self.client.get(
                "/api/dashboard/resumen?fecha_desde=2026-10-15&fecha_hasta=2026-10-15"
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertEqual(data["rango_fechas"]["fecha_desde"], "2026-10-15")
            self.assertEqual(data["rango_fechas"]["fecha_hasta"], "2026-10-15")
            self.assertEqual(data["rango_fechas"]["dias_totales"], 1)
            self.assertEqual(len(data["evolucion_ventas"]), 1)

        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
