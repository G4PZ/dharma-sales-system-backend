"""
Pruebas Unitarias e Integración para los Endpoints Administrativos del Motor IA (Fase 7).

Verifica:
1. GET /api/alertas/modelos/preparacion:
   - Reporte en tiempo real de volumen actual vs requisitos mínimos:
     * K-Means: clientes_totales, clientes_elegibles, compras_minimas_por_cliente (2), clientes_minimos_requeridos (15).
     * Isolation Forest: ventas_completadas, ventas_minimas_requeridas (50).
   - puede_entrenar = False en el estado real actual.
2. POST /api/alertas/modelos/entrenar/kmeans:
   - Responde HTTP 409 Conflict ante InsufficientDataError en la base de datos real.
3. POST /api/alertas/modelos/entrenar/isolation-forest:
   - Responde HTTP 409 Conflict ante InsufficientDataError en la base de datos real.
4. Autenticación estricta:
   - Peticiones sin sesión válida retornan HTTP 401 Unauthorized.
5. Invocación y ejecución controlada mediante mocks:
   - Los endpoints invocan exactamente a train_kmeans_customers y train_isolation_forest_sales.
   - Limpian la memoria caché del detector tras entrenar exitosamente.
6. Inmunidad y desacoplamiento con ventas y SUNAT:
   - No se alteran ventas ni comprobantes en base de datos.
"""

from datetime import datetime, timezone
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Asegurar backend en el path y base de datos base para pruebas unitarias
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DATABASE_URL"] = "postgresql+psycopg://dharma_app:dharmaadmin@localhost:5432/dharma_sales_db"

from app.database import SessionLocal
from app.main import app
from app.models.ai import AlertaVenta, ModeloEntrenamiento
from app.models.receipt import Comprobante
from app.models.sale import Venta
from app.routes.auth import get_current_user
from app.services.ai.detector import ModelNotAvailableError
from app.services.ai.trainer import (
    InsufficientDataError,
    TrainingInProgressError,
)


class TestAIAdminRoutes(unittest.TestCase):

    def setUp(self):
        self.mock_user = {"email": "admin@dharma.pe", "name": "Administrador General"}
        app.dependency_overrides[get_current_user] = lambda: self.mock_user
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    # -------------------------------------------------------------------------
    # 1. GET /api/alertas/modelos/preparacion
    # -------------------------------------------------------------------------

    def test_01_get_modelos_preparacion_datos_actuales(self):
        """
        Verifica que GET /api/alertas/modelos/preparacion calcule en tiempo real los contadores:
        - K-Means: clientes_totales=2, clientes_elegibles=1 (>= 2 compras), puede_entrenar=False.
        - Isolation Forest: ventas_completadas=5, ventas_minimas_requeridas=50, puede_entrenar=False.
        """
        res = self.client.get("/api/alertas/modelos/preparacion")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Validación estructura K-Means
        self.assertIn("kmeans", data)
        km = data["kmeans"]
        self.assertEqual(km["compras_minimas_por_cliente"], 2)
        self.assertEqual(km["clientes_minimos_requeridos"], 15)
        self.assertFalse(km["puede_entrenar"])
        self.assertEqual(km["clientes_elegibles"], 1)
        self.assertGreaterEqual(km["clientes_totales"], 2)
        self.assertAlmostEqual(km["porcentaje_progreso"], round((1 / 15) * 100.0, 1), places=1)

        # Validación estructura Isolation Forest
        self.assertIn("isolation_forest", data)
        if_data = data["isolation_forest"]
        self.assertEqual(if_data["ventas_minimas_requeridas"], 50)
        self.assertFalse(if_data["puede_entrenar"])
        self.assertEqual(if_data["ventas_completadas"], 5)
        self.assertAlmostEqual(if_data["porcentaje_progreso"], round((5 / 50) * 100.0, 1), places=1)

    # -------------------------------------------------------------------------
    # 2. POST /api/alertas/modelos/entrenar/kmeans devuelve 409 si insuficiente
    # -------------------------------------------------------------------------

    def test_02_post_entrenar_kmeans_datos_insuficientes_409(self):
        """
        Verifica que intentar entrenar K-Means con los datos reales actuales (1 cliente elegible)
        sea rechazado controladamente con HTTP 409 Conflict y mensaje claro.
        """
        res = self.client.post("/api/alertas/modelos/entrenar/kmeans")
        self.assertEqual(res.status_code, 409)
        data = res.json()
        self.assertIn("Entrenamiento cancelado por datos insuficientes", data["detail"])
        self.assertIn("mínimo de 15", data["detail"])

    # -------------------------------------------------------------------------
    # 3. POST /api/alertas/modelos/entrenar/isolation-forest devuelve 409 si insuficiente
    # -------------------------------------------------------------------------

    def test_03_post_entrenar_isolation_forest_datos_insuficientes_409(self):
        """
        Verifica que intentar entrenar Isolation Forest con los datos reales actuales (5 ventas)
        sea rechazado controladamente con HTTP 409 Conflict y mensaje claro.
        """
        res = self.client.post("/api/alertas/modelos/entrenar/isolation-forest")
        self.assertEqual(res.status_code, 409)
        data = res.json()
        self.assertIn("Entrenamiento cancelado por datos insuficientes", data["detail"])
        self.assertIn("mínimo de 50", data["detail"])

    # -------------------------------------------------------------------------
    # 4. Autenticación requerida en endpoints administrativos
    # -------------------------------------------------------------------------

    def test_04_autenticacion_requerida_endpoints_administrativos(self):
        """
        Verifica que los endpoints administrativos requieran sesión válida de usuario (HTTP 401).
        """
        # Limpiar overrides para probar sin sesión autenticada
        app.dependency_overrides.clear()
        unauth_client = TestClient(app)

        res_prep = unauth_client.get("/api/alertas/modelos/preparacion")
        self.assertEqual(res_prep.status_code, 401)

        res_km = unauth_client.post("/api/alertas/modelos/entrenar/kmeans")
        self.assertEqual(res_km.status_code, 401)

        res_if = unauth_client.post("/api/alertas/modelos/entrenar/isolation-forest")
        self.assertEqual(res_if.status_code, 401)

    # -------------------------------------------------------------------------
    # 5. Entrenamiento exitoso mock K-Means
    # -------------------------------------------------------------------------

    def test_05_entrenamiento_exitoso_mock_kmeans(self):
        """
        Verifica que al invocarse el endpoint de K-Means y completarse el algoritmo,
        retorne HTTP 200 con el schema de TrainingExecutionResponse y limpie la caché.
        """
        fake_km_result = {
            "status": "ENTRENADO_Y_ACTIVO",
            "modelo_id": 101,
            "version": "kmeans-test-mock-v1",
            "total_muestras": 20,
            "ruta_artefacto": "C:\\fake\\path\\kmeans.joblib",
            "metricas": {"best_silhouette": 0.65, "selected_k": 3},
            "resumen_modelo": {"clusters": {"Cluster 0": {"tamano": 10}}},
        }

        with patch("app.routes.alertas.train_kmeans_customers", return_value=fake_km_result) as mock_train:
            with patch("app.routes.alertas.detector.clear_model_cache") as mock_clear_cache:
                res = self.client.post("/api/alertas/modelos/entrenar/kmeans")

                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "ENTRENADO_Y_ACTIVO")
                self.assertEqual(data["modelo_id"], 101)
                self.assertEqual(data["version"], "kmeans-test-mock-v1")

                mock_train.assert_called_once()
                mock_clear_cache.assert_called_once()

    # -------------------------------------------------------------------------
    # 6. Entrenamiento exitoso mock Isolation Forest
    # -------------------------------------------------------------------------

    def test_06_entrenamiento_exitoso_mock_isolation_forest(self):
        """
        Verifica que al invocarse el endpoint de Isolation Forest y completarse el algoritmo,
        retorne HTTP 200 con el schema de TrainingExecutionResponse y limpie la caché.
        """
        fake_if_result = {
            "status": "ENTRENADO_Y_ACTIVO",
            "modelo_id": 102,
            "version": "if-test-mock-v1",
            "total_muestras": 60,
            "ruta_artefacto": "C:\\fake\\path\\if.joblib",
            "metricas": {"distribucion_anomaly_score": {"median": 0.05}},
            "resumen_modelo": {"total_ventas_entrenamiento": 60},
        }

        with patch("app.routes.alertas.train_isolation_forest_sales", return_value=fake_if_result) as mock_train:
            with patch("app.routes.alertas.detector.clear_model_cache") as mock_clear_cache:
                res = self.client.post("/api/alertas/modelos/entrenar/isolation-forest")

                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "ENTRENADO_Y_ACTIVO")
                self.assertEqual(data["modelo_id"], 102)
                self.assertEqual(data["version"], "if-test-mock-v1")

                mock_train.assert_called_once()
                mock_clear_cache.assert_called_once()

    # -------------------------------------------------------------------------
    # 7. Manejo de entrenamiento en progreso (concurrencia)
    # -------------------------------------------------------------------------

    def test_07_entrenamiento_en_progreso_devuelve_409(self):
        """
        Verifica que si existe un entrenamiento en ejecución concurrente (_LOCK activo),
        el endpoint capture TrainingInProgressError y devuelva HTTP 409 Conflict.
        """
        with patch("app.routes.alertas.train_kmeans_customers", side_effect=TrainingInProgressError("En ejecución")):
            res = self.client.post("/api/alertas/modelos/entrenar/kmeans")
            self.assertEqual(res.status_code, 409)
            self.assertIn("En ejecución", res.json()["detail"])

        with patch("app.routes.alertas.train_isolation_forest_sales", side_effect=TrainingInProgressError("En ejecución")):
            res = self.client.post("/api/alertas/modelos/entrenar/isolation-forest")
            self.assertEqual(res.status_code, 409)
            self.assertIn("En ejecución", res.json()["detail"])

    # -------------------------------------------------------------------------
    # 8. Desacoplamiento total con ventas y comprobantes
    # -------------------------------------------------------------------------

    def test_08_no_interaccion_con_ventas_ni_comprobantes(self):
        """
        Verifica que la consulta de preparación y el rechazo de entrenamiento no alteren
        el conteo ni el estado de ventas ni comprobantes.
        """
        db = SessionLocal()
        try:
            ventas_antes = db.query(Venta).count()
            comprobantes_antes = db.query(Comprobante).count()

            # Consultar preparación e intentar entrenar
            self.client.get("/api/alertas/modelos/preparacion")
            self.client.post("/api/alertas/modelos/entrenar/kmeans")
            self.client.post("/api/alertas/modelos/entrenar/isolation-forest")

            ventas_despues = db.query(Venta).count()
            comprobantes_despues = db.query(Comprobante).count()

            self.assertEqual(ventas_antes, ventas_despues)
            self.assertEqual(comprobantes_antes, comprobantes_despues)
        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 9. Analizar venta existente por numero_venta
    # -------------------------------------------------------------------------

    def test_09_analizar_por_numero_venta_resuelve_id_correctamente(self):
        """
        Verifica que POST /api/alertas/analizar/numero/{numero_venta} reciba un código de venta
        (incluso con minúsculas o espacios), lo resuelva internamente a su venta_id y reutilice
        detector.analyze_sale(db, venta.id, persist_alert=True).
        """
        db = SessionLocal()
        try:
            venta = db.query(Venta).first()
            self.assertIsNotNone(venta, "Se requiere al menos una venta para la prueba")
            target_id = venta.id
            target_numero = venta.numero_venta
        finally:
            db.close()

        fake_analysis = {
            "venta_id": target_id,
            "cliente_id": 1,
            "nivel_alerta": "NORMAL",
            "puntaje_anomalia": 0.15,
            "raw_anomaly_score": 0.05,
            "alerta_generada": False,
            "alerta_id": None,
            "ya_analizada": False,
            "modelo_isolation_forest_id": 1,
            "modelo_isolation_forest_version": "if-test-v1",
            "modelo_kmeans_id": 1,
            "modelo_kmeans_version": "km-test-v1",
            "cluster_perfil_id": 0,
            "perfil_nombre": "Perfil Frecuente",
            "descripcion": "Operación normal dentro del comportamiento esperado.",
            "factores_contribuyentes": [],
        }

        with patch("app.routes.alertas.detector.analyze_sale", return_value=fake_analysis) as mock_analyze:
            # Enviamos con minúsculas para validar normalización
            res = self.client.post(f"/api/alertas/analizar/numero/{target_numero.lower()}")
            self.assertEqual(res.status_code, 200)
            data = res.json()

            # Verificamos que se resolvió al ID correcto
            self.assertEqual(data["venta_id"], target_id)
            self.assertEqual(data["numero_venta"], target_numero)
            self.assertEqual(data["nivel_alerta"], "NORMAL")

            # Verificamos reutilización del detector con el venta_id resuelto
            mock_analyze.assert_called_once()
            call_kwargs = mock_analyze.call_args.kwargs
            self.assertEqual(call_kwargs.get("venta_id"), target_id)
            self.assertEqual(call_kwargs.get("persist_alert"), True)

    # -------------------------------------------------------------------------
    # 10. Venta inexistente por numero_venta -> 404
    # -------------------------------------------------------------------------

    def test_10_analizar_por_numero_venta_inexistente_404(self):
        """
        Verifica que consultar un número de venta que no existe responda HTTP 404
        con mensaje informativo claro.
        """
        res = self.client.post("/api/alertas/analizar/numero/V-999999")
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertIn("No se encontró la venta 'V-999999'", data["detail"])

    # -------------------------------------------------------------------------
    # 11. Modelo no disponible -> 409
    # -------------------------------------------------------------------------

    def test_11_analizar_por_numero_modelo_no_disponible_409(self):
        """
        Verifica que si el detector lanza ModelNotAvailableError (modelo no entrenado),
        se retorne HTTP 409 Conflict con mensaje informativo, no un error 500 no controlado.
        """
        db = SessionLocal()
        try:
            venta = db.query(Venta).first()
            target_numero = venta.numero_venta
        finally:
            db.close()

        with patch(
            "app.routes.alertas.detector.analyze_sale",
            side_effect=ModelNotAvailableError("El modelo de Isolation Forest no está disponible")
        ):
            res = self.client.post(f"/api/alertas/analizar/numero/{target_numero}")
            self.assertEqual(res.status_code, 409)
            data = res.json()
            self.assertIn("no está disponible", data["detail"])

    # -------------------------------------------------------------------------
    # 12. Compatibilidad del endpoint legacy por ID
    # -------------------------------------------------------------------------

    def test_12_analizar_por_id_legacy_continua_funcionando(self):
        """
        Verifica que POST /api/alertas/analizar/{venta_id} continúe funcionando intacto
        por compatibilidad técnica hacia atrás y reporte tanto venta_id como numero_venta.
        """
        db = SessionLocal()
        try:
            venta = db.query(Venta).first()
            target_id = venta.id
            target_numero = venta.numero_venta
        finally:
            db.close()

        fake_analysis = {
            "venta_id": target_id,
            "cliente_id": 1,
            "nivel_alerta": "ALTO",
            "puntaje_anomalia": 0.85,
            "raw_anomaly_score": -0.15,
            "alerta_generada": True,
            "alerta_id": 99,
            "ya_analizada": False,
            "modelo_isolation_forest_id": 1,
            "modelo_isolation_forest_version": "if-test-v1",
            "modelo_kmeans_id": 1,
            "modelo_kmeans_version": "km-test-v1",
            "cluster_perfil_id": 0,
            "perfil_nombre": "Perfil Frecuente",
            "descripcion": "Operación inusual detectada",
            "factores_contribuyentes": ["Monto elevado"],
        }

        with patch("app.routes.alertas.detector.analyze_sale", return_value=fake_analysis) as mock_analyze:
            res = self.client.post(f"/api/alertas/analizar/{target_id}")
            self.assertEqual(res.status_code, 200)
            data = res.json()

            self.assertEqual(data["venta_id"], target_id)
            self.assertEqual(data["numero_venta"], target_numero)
            self.assertEqual(data["nivel_alerta"], "ALTO")
            mock_analyze.assert_called_once()

    def test_13_analizar_por_id_legacy_inexistente_404(self):
        """
        Verifica que POST /api/alertas/analizar/{venta_id} con ID inexistente retorne 404.
        """
        res = self.client.post("/api/alertas/analizar/999999")
        self.assertEqual(res.status_code, 404)
        self.assertIn("No se encontró la venta con ID 999999", res.json()["detail"])

    # -------------------------------------------------------------------------
    # 14. Enriquecimiento de alertas con numero_venta y cliente_razon_social
    # -------------------------------------------------------------------------

    def test_14_listado_y_detalle_alertas_con_numero_venta_y_razon_social(self):
        """
        Verifica:
        1. Listado de alertas incluye 'numero_venta' y 'cliente_razon_social'.
        2. Detalle de alerta incluye ambos campos de negocio.
        3. Los IDs internos ('venta_id', 'cliente_id') siguen presentes.
        4. La paginación ('skip', 'limit', 'total') se preserva correctamente.
        5. La retroalimentación (feedback) se registra y devuelve los datos enriquecidos.
        """
        db = SessionLocal()
        alerta_creada = None
        modelo_creado = None
        try:
            venta = db.query(Venta).first()
            self.assertIsNotNone(venta, "Se requiere al menos una venta para la prueba")
            cliente_id = venta.cliente_id
            target_numero_venta = venta.numero_venta
            target_razon_social = venta.cliente.razon_social

            # Crear modelo IF temporal para asociar la alerta
            modelo_creado = ModeloEntrenamiento(
                tipo_modelo="ISOLATION_FOREST_VENTAS",
                version="if-test-enriquecido-v1",
                parametros={"n_estimators": 50},
                metricas={"median_score": 0.05},
                total_muestras=50,
                ruta_artefacto="C:\\fake\\if_test.joblib",
                estado="ACTIVO",
            )
            db.add(modelo_creado)
            db.commit()
            db.refresh(modelo_creado)

            # Crear alerta temporal
            alerta_creada = AlertaVenta(
                venta_id=venta.id,
                cliente_id=cliente_id,
                modelo_isolation_forest_id=modelo_creado.id,
                modelo_kmeans_id=None,
                puntaje_anomalia=1.0000,
                nivel_alerta="ALTO",
                descripcion="Operación inusual detectada en prueba de enriquecimiento.",
                factores_contribuyentes=[
                    {"factor": "monto_total", "valor_actual": float(venta.total), "descripcion": "Monto atípico"}
                ],
                estado_revision="PENDIENTE",
            )
            db.add(alerta_creada)
            db.commit()
            db.refresh(alerta_creada)

            # 1. Probar GET /api/alertas (listado y paginación)
            res_list = self.client.get("/api/alertas?skip=0&limit=10")
            self.assertEqual(res_list.status_code, 200)
            data_list = res_list.json()

            self.assertGreaterEqual(data_list["total"], 1)
            self.assertEqual(data_list["skip"], 0)
            self.assertEqual(data_list["limit"], 10)
            self.assertIsInstance(data_list["items"], list)

            # Buscar la alerta creada en el listado
            item_match = next((item for item in data_list["items"] if item["id"] == alerta_creada.id), None)
            self.assertIsNotNone(item_match)
            # Validar campos de negocio
            self.assertEqual(item_match["numero_venta"], target_numero_venta)
            self.assertEqual(item_match["cliente_razon_social"], target_razon_social)
            # Validar que los IDs internos siguen presentes
            self.assertEqual(item_match["venta_id"], venta.id)
            self.assertEqual(item_match["cliente_id"], cliente_id)

            # 2. Probar GET /api/alertas/{alerta_id} (detalle)
            res_detail = self.client.get(f"/api/alertas/{alerta_creada.id}")
            self.assertEqual(res_detail.status_code, 200)
            data_detail = res_detail.json()

            self.assertEqual(data_detail["numero_venta"], target_numero_venta)
            self.assertEqual(data_detail["cliente_razon_social"], target_razon_social)
            self.assertEqual(data_detail["venta_id"], venta.id)
            self.assertEqual(data_detail["cliente_id"], cliente_id)
            self.assertEqual(data_detail["nivel_alerta"], "ALTO")

            # 3. Probar PATCH /api/alertas/{alerta_id}/feedback
            res_feedback = self.client.patch(
                f"/api/alertas/{alerta_creada.id}/feedback",
                json={
                    "estado_revision": "CONFIRMADA",
                    "retroalimentacion_notas": "Revisado por auditoría interna",
                },
            )
            self.assertEqual(res_feedback.status_code, 200)
            data_feedback = res_feedback.json()
            self.assertEqual(data_feedback["estado_revision"], "CONFIRMADA")
            self.assertEqual(data_feedback["numero_venta"], target_numero_venta)
            self.assertEqual(data_feedback["cliente_razon_social"], target_razon_social)
            self.assertEqual(data_feedback["venta_id"], venta.id)

        finally:
            # Limpieza estricta de la BD
            if alerta_creada and alerta_creada.id:
                db.query(AlertaVenta).filter(AlertaVenta.id == alerta_creada.id).delete()
            if modelo_creado and modelo_creado.id:
                db.query(ModeloEntrenamiento).filter(ModeloEntrenamiento.id == modelo_creado.id).delete()
            db.commit()
            db.close()


if __name__ == "__main__":
    unittest.main()
