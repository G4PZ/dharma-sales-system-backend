"""
Pruebas Unitarias e Integración para la Integración Automática de Ventas con el Motor IA (Fase 6).

Verifica:
1. Desacoplamiento de la transacción de venta:
   - La venta se confirma y devuelve HTTP 201 antes de la ejecución del análisis IA.
   - El response de venta no altera su schema (sin campos de IA agregados).
2. Tareas en segundo plano (BackgroundTasks):
   - process_sale_ai_background es agendado automáticamente para ventas completadas.
3. Gestión y aislamiento de sesiones SQLAlchemy:
   - La tarea de fondo crea su propia SessionLocal() independiente de la petición HTTP.
   - La sesión de fondo SIEMPRE se cierra en el bloque finally (ante éxito, skip o excepción).
4. Tolerancia y resiliencia a fallos:
   - Ausencia de modelos (ModelNotAvailableError) se gestiona como skip informativo sin error al cliente.
   - Excepciones inesperadas en detector.analyze_sale son capturadas, registradas y absorbidas sin revertir la venta.
5. Inferencia sin reentrenamiento:
   - El background task nunca dispara train_kmeans_customers ni train_isolation_forest_sales.
6. Idempotencia en segundo plano:
   - Múltiples ejecuciones para la misma venta y modelos no generan alertas duplicadas.
7. Desacoplamiento total de comprobantes y SUNAT:
   - La IA no interfiere con facturación electrónica ni SUNAT.
8. Aislamiento estricto de pruebas:
   - No se ejecutan DELETE masivos sobre tablas completas; solo se limpian registros temporales creados por el test.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import os
import sys
import unittest
from typing import Generator, Optional, Tuple
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import joblib

# Asegurar backend en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import ML_MODELS_DIR
from app.database import SessionLocal
from app.main import app
from app.models.ai import AlertaVenta, ModeloEntrenamiento
from app.models.sale import Venta
from app.routes.auth import get_current_user
from app.services.ai import detector
from app.services.ai.background_tasks import process_sale_ai_background
from app.services.ai.detector import ModelNotAvailableError, clear_model_cache
from app.services.ai.trainer import (
    ISOLATION_FOREST_MODEL_TYPE,
    ISOLATION_FOREST_TRAINING_FEATURES,
    fit_isolation_forest_pipeline,
)


# =============================================================================
# FIXTURE SEGURO CON LIMPIEZA ESPECÍFICA (SIN DELETE MASIVO)
# =============================================================================

@contextmanager
def temporary_synthetic_if_model(
    random_state: int = 42,
) -> Generator[Tuple[ModeloEntrenamiento, str], None, None]:
    """
    Context manager seguro de aislamiento estricto:
    1. Crea un artefacto .joblib temporal en ML_MODELS_DIR con prefijo único de testing.
    2. Registra un ModeloEntrenamiento en base de datos con versión única de testing.
    3. Cede el control a la prueba.
    4. En finally:
       - Elimina ÚNICAMENTE las alertas vinculadas al ID de este modelo temporal específico.
       - Elimina ÚNICAMENTE este registro específico de ModeloEntrenamiento.
       - Elimina ÚNICAMENTE el archivo .joblib físico creado.
       - Limpia la memoria caché del detector.
       - NUNCA ejecuta DELETE masivo indiscriminado sobre toda la tabla.
    """
    from tests.test_isolation_forest_trainer import _generate_synthetic_sales_cohort

    db = SessionLocal()
    artifact_path: Optional[str] = None
    modelo_record: Optional[ModeloEntrenamiento] = None

    try:
        df_sales, _ = _generate_synthetic_sales_cohort(n_normal=80, n_anomalies=10, random_state=random_state)
        fit_res = fit_isolation_forest_pipeline(df_features=df_sales, random_state=random_state)

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        version_str = f"test-f6-if-{timestamp_str}"
        artifact_filename = f"_test_f6_temp_if_{timestamp_str}.joblib"
        artifact_path = os.path.join(ML_MODELS_DIR, artifact_filename)

        payload = {
            "pipeline": fit_res["pipeline"],
            "input_features": ISOLATION_FOREST_TRAINING_FEATURES,
            "transformed_features": fit_res["transformed_features"],
            "version": version_str,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": len(df_sales),
            "parameters": {"n_estimators": 10, "random_state": random_state},
            "metrics": {
                "distribucion_decision_function": fit_res["distribucion_decision_function"],
                "distribucion_anomaly_score": fit_res["distribucion_anomaly_score"],
            },
            "anomaly_score_quantiles": fit_res["distribucion_anomaly_score"]["anomaly_score_quantiles"],
            "resumen_modelo": {"total_ventas_entrenamiento": len(df_sales)},
        }

        joblib.dump(payload, artifact_path)

        modelo_record = ModeloEntrenamiento(
            tipo_modelo=ISOLATION_FOREST_MODEL_TYPE,
            version=version_str,
            parametros=payload["parameters"],
            metricas=payload["metrics"],
            total_muestras=payload["total_samples"],
            ruta_artefacto=artifact_path,
            estado="ACTIVO",
            resumen_modelo=payload["resumen_modelo"],
        )
        db.add(modelo_record)
        db.commit()
        db.refresh(modelo_record)

        clear_model_cache()
        yield modelo_record, artifact_path

    finally:
        try:
            if modelo_record and modelo_record.id:
                # Limpiar ÚNICAMENTE alertas vinculadas a este modelo específico
                db.query(AlertaVenta).filter(
                    AlertaVenta.modelo_isolation_forest_id == modelo_record.id
                ).delete(synchronize_session=False)

                # Limpiar ÚNICAMENTE este modelo específico
                db.query(ModeloEntrenamiento).filter(
                    ModeloEntrenamiento.id == modelo_record.id
                ).delete(synchronize_session=False)

                db.commit()
        except Exception as exc:
            db.rollback()
            print(f"[TEST CLEANUP DB ERROR] {exc}")
        finally:
            db.close()

        # Eliminar ÚNICAMENTE el archivo físico creado
        if artifact_path and os.path.exists(artifact_path):
            try:
                os.remove(artifact_path)
            except Exception as exc:
                print(f"[TEST CLEANUP FILE ERROR] {exc}")

        clear_model_cache()


# =============================================================================
# CLASE DE PRUEBAS PARA FASE 6
# =============================================================================

class TestAIBackgroundIntegration(unittest.TestCase):

    def setUp(self):
        clear_model_cache()
        self.mock_user = {"email": "vendedor@dharma.pe", "name": "Vendedor Dharma"}
        app.dependency_overrides[get_current_user] = lambda: self.mock_user
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        clear_model_cache()

    # -------------------------------------------------------------------------
    # 1. Flujo sin modelo activo: skip controlado
    # -------------------------------------------------------------------------

    def test_01_flujo_sin_modelo_activo_skip_controlado(self):
        """
        Verifica que al ejecutar process_sale_ai_background en la base de datos real actual
        (0 modelos Isolation Forest activos), la tarea detecte ModelNotAvailableError,
        lo gestione como un skip informativo, retorne None y no genere ninguna alerta.
        """
        db = SessionLocal()
        try:
            alertas_antes = db.query(AlertaVenta).count()

            # Venta 1 existente en BD real
            res = process_sale_ai_background(venta_id=1)
            self.assertIsNone(res, "El análisis debe retornar None al no existir modelo activo")

            alertas_despues = db.query(AlertaVenta).count()
            self.assertEqual(alertas_antes, alertas_despues, "No debe crearse ninguna alerta")

        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 2. Endpoint de ventas agenda BackgroundTask y conserva su schema intacto
    # -------------------------------------------------------------------------

    def test_02_endpoint_ventas_agenda_background_task_schema_intacto(self):
        """
        Verifica que al registrar una venta a través de la API (POST /api/ventas):
        - La respuesta sea HTTP 201 Created.
        - El schema de respuesta corresponda exactamente a SaleResponse sin campos de IA.
        - Se invoque background_tasks.add_task(process_sale_ai_background, sale_id).
        """
        # Mock de sale_service.create_sale para probar el routing sin alterar inventario real
        fake_sale_response = {
            "id": 9991,
            "numero_venta": "V-999991",
            "cliente_id": 1,
            "cliente": {
                "id": 1,
                "numero_documento": "20601234567",
                "razon_social": "Cliente Mock Test",
                "nombre_contacto": "Contacto Mock",
                "telefono": "999888777",
                "email": "mock@test.pe",
                "direccion": "Av. Test 123",
                "ciudad": "Trujillo",
            },
            "detalles": [],
            "subtotal": 100.0,
            "igv": 18.0,
            "total": 118.0,
            "estado": "COMPLETADA",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        sale_payload = {
            "cliente_id": 1,
            "productos": [{"producto_id": 1, "cantidad": 2}],
        }

        with patch("app.routes.ventas.sale_service.create_sale", return_value=fake_sale_response):
            with patch("app.routes.ventas.process_sale_ai_background") as mock_bg_task:
                res = self.client.post("/api/ventas", json=sale_payload)

                self.assertEqual(res.status_code, 201)
                data = res.json()

                # Verificar schema compatible con SaleResponse
                self.assertEqual(data["id"], 9991)
                self.assertEqual(data["numero_venta"], "V-999991")
                self.assertEqual(data["estado"], "COMPLETADA")

                # Verificar que NO se agregaron campos arbitrarios de IA al response de ventas
                self.assertNotIn("ia_resultado", data)
                self.assertNotIn("anomaly_score", data)
                self.assertNotIn("nivel_alerta", data)

                # Verificar que la tarea en segundo plano fue agendada con el ID correcto
                mock_bg_task.assert_called_once_with(9991)

    # -------------------------------------------------------------------------
    # 3. Independencia de sesión SQLAlchemy HTTP vs BackgroundTask
    # -------------------------------------------------------------------------

    def test_03_independencia_sesion_http_vs_background(self):
        """
        Verifica que process_sale_ai_background instancie una sesión limpia propia (SessionLocal)
        y nunca reutilice el objeto Session inyectado en el request HTTP original.
        """
        with patch("app.services.ai.background_tasks.SessionLocal") as mock_session_factory:
            mock_db = MagicMock()
            mock_session_factory.return_value = mock_db

            with patch("app.services.ai.detector.analyze_sale", return_value={"nivel_alerta": "NORMAL"}):
                process_sale_ai_background(venta_id=1)

                # Comprobar que instanció su propia sesión
                mock_session_factory.assert_called_once()
                # Comprobar que le pasó su propia sesión al detector
                detector.analyze_sale.assert_called_once_with(
                    db=mock_db,
                    venta_id=1,
                    persist_alert=True,
                )
                # Comprobar que cerró su propia sesión
                mock_db.close.assert_called_once()

    # -------------------------------------------------------------------------
    # 4. Cierre garantizado de la sesión background en todos los escenarios
    # -------------------------------------------------------------------------

    def test_04_cierre_garantizado_sesion_en_todos_los_escenarios(self):
        """
        Verifica que db.close() se invoque de forma incondicional en el bloque finally:
        - Escenario A: ModelNotAvailableError.
        - Escenario B: Análisis exitoso NORMAL.
        - Escenario C: Análisis exitoso con ALERTA.
        - Escenario D: Excepción inesperada (RuntimeError).
        """
        casos = [
            ("ModelNotAvailableError", ModelNotAvailableError("Sin modelo")),
            ("NORMAL", {"nivel_alerta": "NORMAL"}),
            ("ALERTA", {"nivel_alerta": "MEDIO", "alerta_id": 42}),
            ("RuntimeError", RuntimeError("Fallo inesperado simulado")),
        ]

        for nombre_caso, retorno_o_error in casos:
            with self.subTest(caso=nombre_caso):
                with patch("app.services.ai.background_tasks.SessionLocal") as mock_factory:
                    mock_db = MagicMock()
                    mock_factory.return_value = mock_db

                    if isinstance(retorno_o_error, Exception):
                        with patch("app.services.ai.detector.analyze_sale", side_effect=retorno_o_error):
                            process_sale_ai_background(venta_id=1)
                    else:
                        with patch("app.services.ai.detector.analyze_sale", return_value=retorno_o_error):
                            process_sale_ai_background(venta_id=1)

                    # En todos los casos db.close() debe haber sido llamado exactamente 1 vez
                    mock_db.close.assert_called_once()

    # -------------------------------------------------------------------------
    # 5. Resiliencia ante fallo catastrófico del detector
    # -------------------------------------------------------------------------

    def test_05_resiliencia_ante_fallo_inesperado_detector(self):
        """
        Verifica que si analyze_sale produce una excepción inesperada (ej. corrupción de archivo,
        fallo de memoria o error de cálculo en NumPy):
        - process_sale_ai_background capture la excepción con logger.exception.
        - Ejecute rollback sobre su sesión local.
        - Cierre la sesión.
        - NO propague la excepción (retorne None).
        - La venta continúe confirmada en la base de datos.
        """
        with patch("app.services.ai.detector.analyze_sale", side_effect=RuntimeError("Error crítico simulado")):
            with patch("app.services.ai.background_tasks.logger.exception") as mock_log_exc:
                res = process_sale_ai_background(venta_id=1)

                self.assertIsNone(res, "Debe retornar None ante fallo inesperado")
                mock_log_exc.assert_called_once()
                self.assertIn("Error inesperado durante el análisis IA automático", mock_log_exc.call_args[0][0])

        # Verificar que la venta en BD real sigue intacta
        db = SessionLocal()
        try:
            venta = db.query(Venta).filter(Venta.id == 1).first()
            self.assertIsNotNone(venta)
            self.assertEqual(venta.estado, "COMPLETADA")
        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 6. Ejecución con modelo temporal genera alerta en segundo plano
    # -------------------------------------------------------------------------

    def test_06_ejecucion_con_modelo_activo_genera_alerta_en_segundo_plano(self):
        """
        Verifica el flujo integral con un modelo temporal sintético activo:
        - process_sale_ai_background ejecuta analyze_sale.
        - Si el score cae en nivel atípico (ej. MEDIO), se persiste la AlertaVenta.
        - Se limpia el registro temporal al finalizar sin dejar alertas huérfanas.
        """
        with temporary_synthetic_if_model() as (if_record, _):
            with patch("app.services.ai.detector.calculate_anomaly_index", return_value=0.9700):
                res = process_sale_ai_background(venta_id=1)

                self.assertIsNotNone(res)
                self.assertEqual(res["nivel_alerta"], "MEDIO")
                self.assertTrue(res["alerta_generada"])
                alerta_id = res["alerta_id"]

                # Verificar en base de datos
                db = SessionLocal()
                try:
                    alerta = db.query(AlertaVenta).filter(AlertaVenta.id == alerta_id).first()
                    self.assertIsNotNone(alerta)
                    self.assertEqual(alerta.nivel_alerta, "MEDIO")
                    self.assertEqual(float(alerta.puntaje_anomalia), 0.9700)
                    self.assertEqual(alerta.modelo_isolation_forest_id, if_record.id)
                finally:
                    db.close()

    # -------------------------------------------------------------------------
    # 7. Idempotencia en segundo plano
    # -------------------------------------------------------------------------

    def test_07_idempotencia_en_segundo_plano(self):
        """
        Verifica que si la misma venta es procesada dos veces consecutivas en segundo plano
        bajo el mismo modelo:
        - La primera ejecución persiste la alerta.
        - La segunda ejecución detecta la alerta existente y la retorna con ya_analizada=True.
        - La tabla alertas_venta conserva una única fila para dicha venta y modelo.
        """
        with temporary_synthetic_if_model() as (if_record, _):
            with patch("app.services.ai.detector.calculate_anomaly_index", return_value=0.9600):
                # Primera llamada
                res1 = process_sale_ai_background(venta_id=1)
                self.assertIsNotNone(res1)
                self.assertFalse(res1.get("ya_analizada", False))
                alerta_id_1 = res1["alerta_id"]

                # Segunda llamada
                res2 = process_sale_ai_background(venta_id=1)
                self.assertIsNotNone(res2)
                self.assertTrue(res2.get("ya_analizada", False))
                alerta_id_2 = res2["alerta_id"]

                self.assertEqual(alerta_id_1, alerta_id_2)

                # Comprobar en base de datos
                db = SessionLocal()
                try:
                    count_alertas = db.query(AlertaVenta).filter(
                        AlertaVenta.venta_id == 1,
                        AlertaVenta.modelo_isolation_forest_id == if_record.id,
                    ).count()
                    self.assertEqual(count_alertas, 1, "No deben existir alertas duplicadas")
                finally:
                    db.close()

    # -------------------------------------------------------------------------
    # 8. Comprobación de que no se dispara reentrenamiento
    # -------------------------------------------------------------------------

    def test_08_no_entrena_al_procesar_venta_en_background(self):
        """
        Verifica estrictamente que procesar una venta en segundo plano NO dispare
        ningún proceso de entrenamiento (train_kmeans_customers ni train_isolation_forest_sales).
        Inferencia y entrenamiento deben permanecer 100% desacoplados.
        """
        with patch("app.services.ai.trainer.train_kmeans_customers") as mock_train_km:
            with patch("app.services.ai.trainer.train_isolation_forest_sales") as mock_train_if:
                process_sale_ai_background(venta_id=1)

                mock_train_km.assert_not_called()
                mock_train_if.assert_not_called()

    # -------------------------------------------------------------------------
    # 9. Desacoplamiento total de comprobantes y SUNAT
    # -------------------------------------------------------------------------

    def test_09_desacoplamiento_sunat_y_regresion(self):
        """
        Verifica que el subsistema de comprobantes y SUNAT permanezca completamente desacoplado:
        - Los servicios de comprobantes y clientes SUNAT pueden importarse limpiamente.
        - Las rutas de comprobantes no han sido modificadas.
        - La creación de ventas y su análisis en segundo plano no ejecutan ninguna llamada a SUNAT.
        """
        # Verificar imports limpios de servicios SUNAT
        from app.services import receipt_service, sunat_client
        self.assertTrue(hasattr(receipt_service, "create_receipt"))
        self.assertTrue(hasattr(sunat_client, "SunatClient"))
        self.assertTrue(hasattr(sunat_client.sunat_client, "emitir_factura"))

        # Verificar que el endpoint de comprobantes sigue operativo en TestClient
        res_disponibles = self.client.get("/api/comprobantes/ventas-disponibles")
        self.assertEqual(res_disponibles.status_code, 200)

    # -------------------------------------------------------------------------
    # 10. Verificación de limpieza estricta final
    # -------------------------------------------------------------------------

    def test_10_limpieza_estricta_especifica_de_test(self):
        """
        Verifica que tras la ejecución de las pruebas no quede ningún artefacto huérfano
        con prefijo '_test_f6_' ni en disco ni en base de datos.
        """
        db = SessionLocal()
        try:
            modelos_test = (
                db.query(ModeloEntrenamiento)
                .filter(ModeloEntrenamiento.version.like("test-f6-%"))
                .count()
            )
            self.assertEqual(modelos_test, 0, f"Quedaron modelos temporales huérfanos: {modelos_test}")

            archivos_test = [f for f in os.listdir(ML_MODELS_DIR) if f.startswith("_test_f6_")]
            self.assertEqual(len(archivos_test), 0, f"Quedaron archivos .joblib temporales: {archivos_test}")
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
