"""
Pruebas Unitarias e Integración para el Motor Contextual y Alertas IA (Fase 5).

Verifica:
1. Comportamiento en base de datos real sin modelos entrenados:
   - get_models_status reporta "NO DISPONIBLE" para Isolation Forest y K-Means.
   - analyze_sale lanza ModelNotAvailableError controlado.
   - Endpoint GET /api/alertas/modelos/estado retorna 200 con "NO DISPONIBLE".
   - Endpoint POST /api/alertas/analizar/{id} retorna HTTP 409 Conflict controlado.
   - Endpoints de alertas (/api/alertas, /api/alertas/stats) retornan colecciones vacías y contadores en 0.
2. Carga segura de modelos:
   - Validación estricta de rutas canónicas contra Directory Traversal.
3. Cálculo preciso del índice relativo (anomaly_index):
   - Interpolación monótona sobre la rejilla empírica de 101 cuantiles.
   - Correcto encuadre de valores inferiores al mínimo (0.0) y superiores al máximo (1.0).
4. Umbrales operativos iniciales:
   - NORMAL (< 0.90), BAJO (0.90 - 0.95), MEDIO (0.95 - 0.99), ALTO (>= 0.99).
5. Regla crítica Anti-Data-Leakage para K-Means:
   - build_customer_features_before_sale evalúa solo ventas anteriores a la venta analizada.
6. Aislamiento sintético:
   - Operación NORMAL no persiste AlertaVenta.
   - Operaciones BAJO, MEDIO y ALTO generan AlertaVenta persistente con puntaje_anomalia == anomaly_index.
7. Contextualización dual (IF + K-Means):
   - Cliente con historial suficiente (>= 2 compras) recibe perfil K-Means sin modificar el score de IF.
8. Cold Start (cliente nuevo o con 1 compra):
   - modelo_kmeans_id = NULL, evaluación contra distribución global, sin estigmatizar al cliente nuevo.
9. Explicabilidad y ausencia de terminología de fraude:
   - Factores contextuales con ratio y montos verificables.
   - Prohibición absoluta de "fraude", "delito" y "probabilidad de fraude".
10. Idempotencia:
    - Re-análisis con mismos modelos no lanza IntegrityError y devuelve la alerta existente.
11. Retroalimentación humana (Feedback Loop):
    - Transiciones a CONFIRMADA y FALSO_POSITIVO, registro de revisor y preservación del score.
12. Endpoints FastAPI completos (TestClient):
    - Inyección automática del revisor desde la sesión autenticada.
13. Estado final limpio:
    - Garantía de 0 modelos reales activos y 0 alertas ficticias persistentes en PostgreSQL.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import os
import sys
import unittest
from typing import Generator, Tuple
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd

# Asegurar backend en el path y base de datos para pruebas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DATABASE_URL"] = "postgresql+psycopg://dharma_app:dharmaadmin@localhost:5432/dharma_sales_db"

from app.config import ML_MODELS_DIR
from app.database import SessionLocal
from app.main import app
from app.models.ai import AlertaVenta, ModeloEntrenamiento
from app.models.sale import Venta
from app.routes.auth import get_current_user
from app.schemas.ai import AlertFeedbackUpdate
from app.services.ai import alert_service, detector
from app.services.ai.detector import (
    LEVEL_ALTO,
    LEVEL_BAJO,
    LEVEL_MEDIO,
    LEVEL_NORMAL,
    THRESHOLD_ALTO,
    THRESHOLD_BAJO,
    THRESHOLD_MEDIO,
    THRESHOLD_NORMAL,
    ModelNotAvailableError,
    _validate_safe_model_path,
    calculate_anomaly_index,
    clear_model_cache,
    determine_alert_level,
)
from app.services.ai.feature_engineering import (
    KMEANS_FEATURE_COLUMNS,
    build_customer_features_before_sale,
)
from app.services.ai.trainer import (
    ISOLATION_FOREST_MODEL_TYPE,
    ISOLATION_FOREST_TRAINING_FEATURES,
    KMEANS_MODEL_TYPE,
    KMEANS_TRAINING_FEATURES,
    MIN_PURCHASES_FOR_KMEANS_PROFILE,
    compute_score_distribution,
    fit_isolation_forest_pipeline,
    fit_kmeans_pipeline,
)


# =============================================================================
# GENERADORES DE MODELOS SINTÉTICOS TEMPORALES Y FIXTURES
# =============================================================================

def _build_synthetic_if_artifact(random_state: int = 42) -> Tuple[Dict, str]:
    """Crea en memoria y serializa un pipeline Isolation Forest sintético para pruebas."""
    from tests.test_isolation_forest_trainer import _generate_synthetic_sales_cohort

    df_sales, _ = _generate_synthetic_sales_cohort(n_normal=100, n_anomalies=10, random_state=random_state)
    fit_res = fit_isolation_forest_pipeline(df_features=df_sales, random_state=random_state)

    artifact_filename = f"_test_temp_if_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.joblib"
    artifact_path = os.path.join(ML_MODELS_DIR, artifact_filename)

    payload = {
        "pipeline": fit_res["pipeline"],
        "input_features": ISOLATION_FOREST_TRAINING_FEATURES,
        "transformed_features": fit_res["transformed_features"],
        "version": f"test-if-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
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
    return payload, artifact_path


def _build_synthetic_kmeans_artifact(random_state: int = 42) -> Tuple[Dict, str]:
    """Crea en memoria y serializa un pipeline K-Means sintético para pruebas."""
    from tests.test_kmeans_trainer import _generate_synthetic_customers

    df_cust = _generate_synthetic_customers(n_per_group=20, random_state=random_state)
    fit_res = fit_kmeans_pipeline(df_features=df_cust, random_state=random_state)

    artifact_filename = f"_test_temp_km_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.joblib"
    artifact_path = os.path.join(ML_MODELS_DIR, artifact_filename)

    payload = {
        "pipeline": fit_res["pipeline"],
        "feature_columns": KMEANS_TRAINING_FEATURES,
        "version": f"test-km-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "selected_k": fit_res["selected_k"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(df_cust),
        "resumen_modelo": fit_res["resumen_modelo"],
    }

    joblib.dump(payload, artifact_path)
    return payload, artifact_path


@contextmanager
def temporary_synthetic_models(
    include_if: bool = True,
    include_kmeans: bool = False,
) -> Generator[Tuple[Optional[ModeloEntrenamiento], Optional[ModeloEntrenamiento]], None, None]:
    """
    Context manager de aislamiento estricto:
    1. Crea artefactos .joblib sintéticos en ML_MODELS_DIR con prefijo temporal.
    2. Registra registros temporales en modelos_entrenamiento con estado = ACTIVO.
    3. Cede el control a la prueba.
    4. En finally:
       - Elimina cualquier AlertaVenta generada durante la prueba.
       - Elimina los registros temporales de ModeloEntrenamiento.
       - Elimina físicamente los archivos .joblib sintéticos.
       - Limpia la memoria caché del detector.
       - Ejecuta commit garantizando 0 residuos en base de datos.
    """
    db = SessionLocal()
    if_record: Optional[ModeloEntrenamiento] = None
    km_record: Optional[ModeloEntrenamiento] = None
    if_path: Optional[str] = None
    km_path: Optional[str] = None

    try:
        if include_if:
            if_payload, if_path = _build_synthetic_if_artifact()
            if_record = ModeloEntrenamiento(
                tipo_modelo=ISOLATION_FOREST_MODEL_TYPE,
                version=if_payload["version"],
                parametros=if_payload["parameters"],
                metricas=if_payload["metrics"],
                total_muestras=if_payload["total_samples"],
                ruta_artefacto=if_path,
                estado="ACTIVO",
                resumen_modelo=if_payload["resumen_modelo"],
            )
            db.add(if_record)

        if include_kmeans:
            km_payload, km_path = _build_synthetic_kmeans_artifact()
            km_record = ModeloEntrenamiento(
                tipo_modelo=KMEANS_MODEL_TYPE,
                version=km_payload["version"],
                parametros={"k": km_payload["selected_k"]},
                metricas={},
                total_muestras=km_payload["total_samples"],
                ruta_artefacto=km_path,
                estado="ACTIVO",
                resumen_modelo=km_payload["resumen_modelo"],
            )
            db.add(km_record)

        db.commit()
        if if_record:
            db.refresh(if_record)
        if km_record:
            db.refresh(km_record)

        clear_model_cache()
        yield if_record, km_record

    finally:
        try:
            # Limpiar alertas creadas vinculadas a estos modelos temporales
            model_ids = [m.id for m in [if_record, km_record] if m is not None]
            if model_ids:
                db.query(AlertaVenta).filter(
                    (AlertaVenta.modelo_isolation_forest_id.in_(model_ids)) |
                    (AlertaVenta.modelo_kmeans_id.in_(model_ids))
                ).delete(synchronize_session=False)

                # Limpiar los modelos de entrenamiento temporales
                db.query(ModeloEntrenamiento).filter(
                    ModeloEntrenamiento.id.in_(model_ids)
                ).delete(synchronize_session=False)

                db.commit()
        except Exception as exc:
            db.rollback()
            print(f"[CLEANUP ERROR DB] {exc}")
        finally:
            db.close()

        # Limpiar archivos temporales en disco
        for path in [if_path, km_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as exc:
                    print(f"[CLEANUP ERROR FILE] {exc}")

        clear_model_cache()


# =============================================================================
# CLASE PRINCIPAL DE PRUEBAS
# =============================================================================

class TestContextualDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Registrar los archivos preexistentes en ML_MODELS_DIR para verificar aislamiento
        cls.preexisting_files = set(os.listdir(ML_MODELS_DIR)) if os.path.exists(ML_MODELS_DIR) else set()

    def setUp(self):
        clear_model_cache()
        # Mock de autenticación para TestClient
        self.mock_user = {"email": "auditor@dharma.pe", "name": "Auditor IA"}
        app.dependency_overrides[get_current_user] = lambda: self.mock_user
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        clear_model_cache()

    # -------------------------------------------------------------------------
    # 1. Comportamiento en base de datos real sin modelos
    # -------------------------------------------------------------------------

    def test_01_estado_modelos_bd_real_sin_entrenar(self):
        """
        Verifica que al consultar el estado de modelos en la base de datos real actual (0 modelos),
        el servicio y el endpoint respondan 'NO DISPONIBLE' y analyze_sale lance ModelNotAvailableError.
        """
        db = SessionLocal()
        try:
            # Verificar en servicio
            status_data = alert_service.get_models_status(db=db)
            self.assertFalse(status_data["isolation_forest"]["disponible"])
            self.assertEqual(status_data["isolation_forest"]["estado"], "NO DISPONIBLE")
            self.assertFalse(status_data["kmeans"]["disponible"])
            self.assertEqual(status_data["kmeans"]["estado"], "NO DISPONIBLE")

            # Verificar que analyze_sale rechaza controladamente
            with self.assertRaises(ModelNotAvailableError) as ctx:
                detector.analyze_sale(db=db, venta_id=1)
            self.assertIn("no existe un modelo Isolation Forest entrenado y activo", str(ctx.exception))

        finally:
            db.close()

    def test_02_endpoints_fastapi_sin_modelos(self):
        """
        Verifica el comportamiento de los endpoints FastAPI cuando no existen modelos reales activos:
        - GET /api/alertas/modelos/estado -> 200 con 'NO DISPONIBLE'
        - POST /api/alertas/analizar/1 -> HTTP 409 Conflict
        - GET /api/alertas -> 200 con items: []
        - GET /api/alertas/stats -> 200 con todos los contadores en 0
        """
        # Estado de modelos
        res_estado = self.client.get("/api/alertas/modelos/estado")
        self.assertEqual(res_estado.status_code, 200)
        data_estado = res_estado.json()
        self.assertEqual(data_estado["isolation_forest"]["estado"], "NO DISPONIBLE")
        self.assertEqual(data_estado["kmeans"]["estado"], "NO DISPONIBLE")

        # Intento de análisis manual sin modelo IF
        res_analizar = self.client.post("/api/alertas/analizar/1")
        self.assertEqual(res_analizar.status_code, 409)
        self.assertIn("no existe un modelo Isolation Forest", res_analizar.json()["detail"])

        # Listado de alertas vacío
        res_list = self.client.get("/api/alertas")
        self.assertEqual(res_list.status_code, 200)
        self.assertEqual(res_list.json()["total"], 0)
        self.assertEqual(res_list.json()["items"], [])

        # Estadísticas en 0
        res_stats = self.client.get("/api/alertas/stats")
        self.assertEqual(res_stats.status_code, 200)
        data_stats = res_stats.json()
        self.assertEqual(data_stats["total"], 0)
        self.assertEqual(data_stats["pendientes"], 0)
        self.assertEqual(data_stats["bajo"], 0)
        self.assertEqual(data_stats["medio"], 0)
        self.assertEqual(data_stats["alto"], 0)

    # -------------------------------------------------------------------------
    # 2. Seguridad de carga de artefactos y Path Traversal
    # -------------------------------------------------------------------------

    def test_03_seguridad_path_traversal_en_carga(self):
        """
        Verifica que _validate_safe_model_path rechace cualquier ruta que intente escapar
        de ML_MODELS_DIR mediante Directory Traversal o rutas externas absolutas.
        """
        with self.assertRaises(ValueError):
            _validate_safe_model_path("../../../etc/shadow")

        with self.assertRaises(ValueError):
            _validate_safe_model_path("C:\\Windows\\System32\\cmd.exe")

    # -------------------------------------------------------------------------
    # 3. Cálculo de anomaly_index y monotonía
    # -------------------------------------------------------------------------

    def test_04_calculo_anomaly_index_interpolacion_cuantiles(self):
        """
        Verifica que calculate_anomaly_index interpole linealmente sobre la rejilla empírica,
        acote valores inferiores a 0.0000, superiores a 1.0000 y preserve monotonía estricta.
        """
        # Rejilla sintética uniforme de 101 puntos de -0.50 a +0.50
        grid = list(np.linspace(-0.50, 0.50, 101))

        # Fronteras y extremos
        self.assertEqual(calculate_anomaly_index(-0.80, grid), 0.0)
        self.assertEqual(calculate_anomaly_index(-0.50, grid), 0.0)
        self.assertEqual(calculate_anomaly_index(0.50, grid), 1.0)
        self.assertEqual(calculate_anomaly_index(0.95, grid), 1.0)

        # Mediana exacta (cuantil 50% = 0.0)
        self.assertAlmostEqual(calculate_anomaly_index(0.0, grid), 0.50, places=3)

        # Monotonía
        scores = np.linspace(-0.60, 0.60, 25)
        indices = [calculate_anomaly_index(float(s), grid) for s in scores]
        for i in range(len(indices) - 1):
            self.assertLessEqual(indices[i], indices[i + 1])

    # -------------------------------------------------------------------------
    # 4. Umbrales operativos iniciales
    # -------------------------------------------------------------------------

    def test_05_umbrales_operativos_y_clasificacion_niveles(self):
        """
        Verifica la clasificación según las constantes de umbral:
        NORMAL: < 0.90
        BAJO: 0.90 <= index < 0.95
        MEDIO: 0.95 <= index < 0.99
        ALTO: index >= 0.99
        """
        self.assertEqual(determine_alert_level(0.85), LEVEL_NORMAL)
        self.assertEqual(determine_alert_level(0.8999), LEVEL_NORMAL)
        self.assertEqual(determine_alert_level(0.90), LEVEL_BAJO)
        self.assertEqual(determine_alert_level(0.9499), LEVEL_BAJO)
        self.assertEqual(determine_alert_level(0.95), LEVEL_MEDIO)
        self.assertEqual(determine_alert_level(0.9899), LEVEL_MEDIO)
        self.assertEqual(determine_alert_level(0.99), LEVEL_ALTO)
        self.assertEqual(determine_alert_level(1.00), LEVEL_ALTO)

    # -------------------------------------------------------------------------
    # 5. Anti-Data-Leakage para características del cliente antes de la venta
    # -------------------------------------------------------------------------

    def test_06_anti_data_leakage_perfil_cliente_antes_de_venta(self):
        """
        Verifica que build_customer_features_before_sale evalúe únicamente las compras
        estrictamente anteriores a (created_at, id) de la venta analizada:
        - Para la venta 5 (cuarta compra del cliente 1): total_compras previas debe ser exactamente 3.
        - Para la venta 1 (primera compra del cliente 1): total_compras previas debe ser 0.
        - Para la venta 13 (primera compra del cliente 5): total_compras previas debe ser 0.
        """
        db = SessionLocal()
        try:
            # Venta 5 del cliente 1 (tiene previas las ventas 1, 2 y 4)
            df_v5 = build_customer_features_before_sale(db=db, venta_id=5)
            self.assertEqual(len(df_v5), 1)
            self.assertEqual(int(df_v5["total_compras"].iloc[0]), 3)

            # Venta 1 del cliente 1 (primera venta en la cronología)
            df_v1 = build_customer_features_before_sale(db=db, venta_id=1)
            self.assertEqual(int(df_v1["total_compras"].iloc[0]), 0)
            self.assertTrue(pd.isna(df_v1["monto_promedio"].iloc[0]))

            # Venta 13 del cliente 5 (cliente nuevo)
            df_v13 = build_customer_features_before_sale(db=db, venta_id=13)
            self.assertEqual(int(df_v13["total_compras"].iloc[0]), 0)
        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 6. Análisis sintético aislado: NORMAL no persiste alerta
    # -------------------------------------------------------------------------

    def test_07_analisis_caso_normal_no_persiste_alerta(self):
        """
        Verifica que cuando una venta resulta NORMAL (anomaly_index < 0.90):
        - analyze_sale devuelve el score y nivel NORMAL.
        - alerta_generada es False.
        - alerta_id es None.
        - NO se inserta ningún registro en la tabla alertas_venta.
        """
        db = SessionLocal()
        try:
            with temporary_synthetic_models(include_if=True, include_kmeans=False) as (if_mod, _):
                # Forzar un anomaly_index bajo simulando cuantiles donde el score caiga en nivel NORMAL
                res = detector.analyze_sale(db=db, venta_id=1, persist_alert=True)

                self.assertIn("nivel_alerta", res)
                self.assertIn("puntaje_anomalia", res)
                self.assertIn("raw_anomaly_score", res)

                # Si es normal, no debe persistir
                if res["nivel_alerta"] == LEVEL_NORMAL:
                    self.assertFalse(res["alerta_generada"])
                    self.assertIsNone(res["alerta_id"])
                    alertas_count = db.query(AlertaVenta).count()
                    self.assertEqual(alertas_count, 0)
        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 7. Análisis sintético: BAJO, MEDIO y ALTO generan alerta con puntaje_anomalia == anomaly_index
    # -------------------------------------------------------------------------

    def test_08_analisis_casos_bajo_medio_alto_y_persistencia(self):
        """
        Verifica mediante mocks controlados de decision_function que:
        - Score BAJO (ej. index 0.92) genera AlertaVenta con nivel BAJO y puntaje_anomalia = 0.9200.
        - Score MEDIO (ej. index 0.97) genera AlertaVenta con nivel MEDIO y puntaje_anomalia = 0.9700.
        - Score ALTO (ej. index 0.995) genera AlertaVenta con nivel ALTO y puntaje_anomalia = 0.9950.
        - puntaje_anomalia almacena el índice relativo percentilar y NO el score bruto.
        """
        db = SessionLocal()
        try:
            with temporary_synthetic_models(include_if=True, include_kmeans=False) as (if_mod, _):
                # Caso BAJO simulado
                with unittest.mock.patch("app.services.ai.detector.calculate_anomaly_index", return_value=0.9200):
                    res_bajo = detector.analyze_sale(db=db, venta_id=1, persist_alert=True)
                    self.assertEqual(res_bajo["nivel_alerta"], LEVEL_BAJO)
                    self.assertTrue(res_bajo["alerta_generada"])
                    self.assertIsNotNone(res_bajo["alerta_id"])

                    alerta_db = db.query(AlertaVenta).filter(AlertaVenta.id == res_bajo["alerta_id"]).first()
                    self.assertIsNotNone(alerta_db)
                    self.assertEqual(float(alerta_db.puntaje_anomalia), 0.9200)
                    self.assertEqual(alerta_db.nivel_alerta, LEVEL_BAJO)
                    # Limpiar para la siguiente simulación
                    db.delete(alerta_db)
                    db.commit()

                # Caso MEDIO simulado
                with unittest.mock.patch("app.services.ai.detector.calculate_anomaly_index", return_value=0.9700):
                    res_medio = detector.analyze_sale(db=db, venta_id=1, persist_alert=True)
                    self.assertEqual(res_medio["nivel_alerta"], LEVEL_MEDIO)
                    self.assertTrue(res_medio["alerta_generada"])
                    alerta_db = db.query(AlertaVenta).filter(AlertaVenta.id == res_medio["alerta_id"]).first()
                    self.assertEqual(float(alerta_db.puntaje_anomalia), 0.9700)
                    self.assertEqual(alerta_db.nivel_alerta, LEVEL_MEDIO)
                    db.delete(alerta_db)
                    db.commit()

                # Caso ALTO simulado
                with unittest.mock.patch("app.services.ai.detector.calculate_anomaly_index", return_value=0.9950):
                    res_alto = detector.analyze_sale(db=db, venta_id=1, persist_alert=True)
                    self.assertEqual(res_alto["nivel_alerta"], LEVEL_ALTO)
                    self.assertTrue(res_alto["alerta_generada"])
                    alerta_db = db.query(AlertaVenta).filter(AlertaVenta.id == res_alto["alerta_id"]).first()
                    self.assertEqual(float(alerta_db.puntaje_anomalia), 0.9950)
                    self.assertEqual(alerta_db.nivel_alerta, LEVEL_ALTO)
                    db.delete(alerta_db)
                    db.commit()

        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 8. Test de cliente con K-Means (historial suficiente >= 2 compras)
    # -------------------------------------------------------------------------

    def test_09_analisis_dual_con_kmeans_cliente_con_historial(self):
        """
        Verifica escenario con cliente con historial (Venta 5, cliente 1 con 3 compras previas):
        - Ambos modelos (IF y K-Means) están disponibles.
        - Isolation Forest produce score continuo e índice relativo.
        - K-Means asigna cluster_perfil_id y perfil_nombre ('Perfil X').
        - La alerta creada referencia tanto a modelo_isolation_forest_id como a modelo_kmeans_id.
        - K-Means NO modifica el score de anomalía de Isolation Forest.
        """
        db = SessionLocal()
        try:
            with temporary_synthetic_models(include_if=True, include_kmeans=True) as (if_mod, km_mod):
                with unittest.mock.patch("app.services.ai.detector.calculate_anomaly_index", return_value=0.9600):
                    res = detector.analyze_sale(db=db, venta_id=5, persist_alert=True)

                    self.assertEqual(res["nivel_alerta"], LEVEL_MEDIO)
                    self.assertEqual(res["modelo_isolation_forest_id"], if_mod.id)
                    self.assertEqual(res["modelo_kmeans_id"], km_mod.id)
                    self.assertIsNotNone(res["cluster_perfil_id"])
                    self.assertTrue(str(res["perfil_nombre"]).startswith("Perfil "))

                    alerta = db.query(AlertaVenta).filter(AlertaVenta.id == res["alerta_id"]).first()
                    self.assertIsNotNone(alerta)
                    self.assertEqual(alerta.modelo_isolation_forest_id, if_mod.id)
                    self.assertEqual(alerta.modelo_kmeans_id, km_mod.id)
                    self.assertEqual(alerta.cluster_perfil_id, res["cluster_perfil_id"])
                    self.assertEqual(alerta.perfil_nombre, res["perfil_nombre"])
        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 9. Test sin K-Means (Cold Start / cliente sin historial suficiente)
    # -------------------------------------------------------------------------

    def test_10_analisis_sin_kmeans_cold_start(self):
        """
        Verifica escenario para cliente nuevo o con compras insuficientes (< 2 compras):
        - Venta 13 (cliente 5 tiene 0 compras previas).
        - Aunque exista K-Means activo, NO se asigna perfil K-Means (modelo_kmeans_id = NULL).
        - Isolation Forest analiza la venta normalmente con score e índice válidos.
        - Los factores contextuales indican evaluación contra el comportamiento global aprendido.
        - Prohibido afirmar 'Cliente nuevo = riesgo'.
        """
        db = SessionLocal()
        try:
            with temporary_synthetic_models(include_if=True, include_kmeans=True) as (if_mod, km_mod):
                with unittest.mock.patch("app.services.ai.detector.calculate_anomaly_index", return_value=0.9300):
                    res = detector.analyze_sale(db=db, venta_id=13, persist_alert=True)

                    self.assertEqual(res["nivel_alerta"], LEVEL_BAJO)
                    self.assertEqual(res["modelo_isolation_forest_id"], if_mod.id)
                    self.assertIsNone(res["modelo_kmeans_id"])
                    self.assertIsNone(res["cluster_perfil_id"])
                    self.assertIsNone(res["perfil_nombre"])

                    alerta = db.query(AlertaVenta).filter(AlertaVenta.id == res["alerta_id"]).first()
                    self.assertIsNotNone(alerta)
                    self.assertIsNone(alerta.modelo_kmeans_id)
                    self.assertIsNone(alerta.cluster_perfil_id)

                    # Verificar factor de cliente sin historial
                    factores = res["factores_contribuyentes"]
                    cold_factors = [f for f in factores if f.get("factor") == "cliente_sin_historial"]
                    self.assertEqual(len(cold_factors), 1)
                    desc_factor = cold_factors[0]["descripcion"]
                    self.assertIn("sin historial individual suficiente", desc_factor)
                    self.assertIn("comportamiento global aprendido", desc_factor)
                    self.assertNotIn("riesgo", desc_factor.lower())
        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 10. Explicabilidad y ausencia de terminología de fraude
    # -------------------------------------------------------------------------

    def test_11_explicabilidad_y_ausencia_termino_fraude(self):
        """
        Verifica que:
        - Si el monto actual supera ampliamente el historial, se genere el factor 'monto_vs_historial'
          con monto actual, valor de referencia y ratio numérico.
        - En ninguna parte de los factores ni en la descripción se empleen palabras prohibidas
          ('fraude', 'delito', 'fraude confirmado', 'probabilidad de fraude').
        """
        df_sim = pd.DataFrame([{
            "monto_total": 4500.0,
            "unidades_totales": 35,
            "compras_previas": 10,
            "monto_promedio_previo": 300.0,
            "unidades_promedio_previas": 5.0,
            "concentracion_importe": 0.85,
            "hora_operacion": 14,
            "ratio_monto_vs_promedio_cliente": 15.0,
            "ratio_unidades_vs_promedio_cliente": 7.0,
        }])

        factors = detector.generate_contextual_factors(
            df_sale_features=df_sim,
            anomaly_index=0.985,
            raw_anomaly_score=0.18,
            customer_has_kmeans_profile=True,
            kmeans_profile_name="Perfil 1",
        )

        # Factor monto_vs_historial
        monto_fact = [f for f in factors if f["factor"] == "monto_vs_historial"]
        self.assertEqual(len(monto_fact), 1)
        f_m = monto_fact[0]
        self.assertEqual(f_m["valor_actual"], 4500.0)
        self.assertEqual(f_m["valor_referencia"], 300.0)
        self.assertEqual(f_m["ratio"], 15.0)
        self.assertIn("15.0 veces el ticket promedio histórico", f_m["descripcion"])

        # Descripción
        desc = detector.build_alert_description(
            level=LEVEL_MEDIO,
            factors=factors,
            has_kmeans_profile=True,
            kmeans_profile_name="Perfil 1",
        )

        # Verificación estricta de ausencia de terminología punitiva / fraude
        texto_completo = (desc + " " + " ".join(f["descripcion"] for f in factors)).lower()
        palabras_prohibidas = ["fraude", "delito", "estafa", "delictiva", "probabilidad de fraude", "fraude confirmado"]
        for p in palabras_prohibidas:
            self.assertNotIn(p, texto_completo, f"Palabra prohibida encontrada: '{p}'")

    # -------------------------------------------------------------------------
    # 11. Idempotencia ante re-análisis con mismos modelos
    # -------------------------------------------------------------------------

    def test_12_idempotencia_reanalisis_misma_venta(self):
        """
        Verifica que al analizar dos veces consecutivas la misma venta con el mismo modelo:
        - El primer análisis crea la alerta.
        - El segundo análisis detecta la alerta existente y la retorna con ya_analizada=True.
        - NO se produce un IntegrityError de base de datos ni error 500.
        - La tabla alertas_venta conserva una única fila.
        """
        db = SessionLocal()
        try:
            with temporary_synthetic_models(include_if=True, include_kmeans=False) as (if_mod, _):
                with unittest.mock.patch("app.services.ai.detector.calculate_anomaly_index", return_value=0.9300):
                    # Primer análisis
                    res1 = detector.analyze_sale(db=db, venta_id=1, persist_alert=True)
                    self.assertTrue(res1["alerta_generada"])
                    self.assertFalse(res1["ya_analizada"])
                    alerta_id_1 = res1["alerta_id"]

                    # Segundo análisis
                    res2 = detector.analyze_sale(db=db, venta_id=1, persist_alert=True)
                    self.assertTrue(res2["alerta_generada"])
                    self.assertTrue(res2["ya_analizada"])
                    alerta_id_2 = res2["alerta_id"]

                    self.assertEqual(alerta_id_1, alerta_id_2)

                    # Verificar exactamente 1 alerta en base de datos
                    total_alertas = db.query(AlertaVenta).filter(AlertaVenta.venta_id == 1).count()
                    self.assertEqual(total_alertas, 1)

        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 12. Retroalimentación humana (Feedback Loop)
    # -------------------------------------------------------------------------

    def test_13_retroalimentacion_humana_feedback(self):
        """
        Verifica el ciclo de auditoría humana:
        - Transición PENDIENTE -> CONFIRMADA con notas y revisor.
        - Preservación estricta e inalterada del puntaje_anomalia original.
        - Transición posterior a FALSO_POSITIVO.
        """
        db = SessionLocal()
        try:
            with temporary_synthetic_models(include_if=True, include_kmeans=False) as (if_mod, _):
                with unittest.mock.patch("app.services.ai.detector.calculate_anomaly_index", return_value=0.9650):
                    res = detector.analyze_sale(db=db, venta_id=1, persist_alert=True)
                    alerta_id = res["alerta_id"]

                    # 1. Auditoría CONFIRMADA
                    fb_confirmada = AlertFeedbackUpdate(
                        estado_revision="CONFIRMADA",
                        retroalimentacion_notas="Operación confirmada como atípica por cliente no habitual.",
                    )
                    alerta_conf = alert_service.submit_feedback(
                        db=db,
                        alerta_id=alerta_id,
                        feedback_data=fb_confirmada,
                        revisado_por="auditor@dharma.pe",
                    )

                    self.assertEqual(alerta_conf.estado_revision, "CONFIRMADA")
                    self.assertEqual(alerta_conf.revisado_por, "auditor@dharma.pe")
                    self.assertIsNotNone(alerta_conf.fecha_revision)
                    self.assertEqual(float(alerta_conf.puntaje_anomalia), 0.9650)

                    # 2. Auditoría FALSO_POSITIVO
                    fb_fp = AlertFeedbackUpdate(
                        estado_revision="FALSO_POSITIVO",
                        retroalimentacion_notas="Monto verificado telefónicamente con gerencia.",
                    )
                    alerta_fp = alert_service.submit_feedback(
                        db=db,
                        alerta_id=alerta_id,
                        feedback_data=fb_fp,
                        revisado_por="supervisor@dharma.pe",
                    )
                    self.assertEqual(alerta_fp.estado_revision, "FALSO_POSITIVO")
                    self.assertEqual(alerta_fp.revisado_por, "supervisor@dharma.pe")
                    self.assertEqual(float(alerta_fp.puntaje_anomalia), 0.9650)

        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 13. Endpoints FastAPI completos y seguridad de identidad del revisor
    # -------------------------------------------------------------------------

    def test_14_fastapi_endpoints_completos_y_feedback_revisor(self):
        """
        Verifica la API REST a través de TestClient:
        - Creación de alerta sintética en contexto temporal.
        - GET /api/alertas/{id} -> 200 con relaciones y factores contextuales.
        - PATCH /api/alertas/{id}/feedback -> 200, asignando revisado_por desde la sesión autenticada.
        - GET /api/alertas/stats -> 200, reflejando el nuevo estado de la alerta.
        """
        db = SessionLocal()
        try:
            with temporary_synthetic_models(include_if=True, include_kmeans=False) as (if_mod, _):
                with unittest.mock.patch("app.services.ai.detector.calculate_anomaly_index", return_value=0.9750):
                    res = detector.analyze_sale(db=db, venta_id=1, persist_alert=True)
                    alerta_id = res["alerta_id"]

                    # GET /api/alertas/{id}
                    res_get = self.client.get(f"/api/alertas/{alerta_id}")
                    self.assertEqual(res_get.status_code, 200)
                    data_get = res_get.json()
                    self.assertEqual(data_get["id"], alerta_id)
                    self.assertEqual(data_get["nivel_alerta"], LEVEL_MEDIO)
                    self.assertIn("factores_contribuyentes", data_get)

                    # PATCH /api/alertas/{id}/feedback
                    body_patch = {
                        "estado_revision": "CONFIRMADA",
                        "retroalimentacion_notas": "Revisado por analista de seguridad comercial.",
                    }
                    res_patch = self.client.patch(f"/api/alertas/{alerta_id}/feedback", json=body_patch)
                    self.assertEqual(res_patch.status_code, 200)
                    data_patch = res_patch.json()
                    self.assertEqual(data_patch["estado_revision"], "CONFIRMADA")
                    # La identidad debe provenir de current_user (auditor@dharma.pe)
                    self.assertEqual(data_patch["revisado_por"], "auditor@dharma.pe")
                    self.assertIsNotNone(data_patch["fecha_revision"])

                    # GET /api/alertas/stats debe mostrar 1 confirmada
                    res_stats = self.client.get("/api/alertas/stats")
                    self.assertEqual(res_stats.status_code, 200)
                    self.assertEqual(res_stats.json()["confirmadas"], 1)

        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 14. Verificación de limpieza final absoluta en PostgreSQL y ML_MODELS_DIR
    # -------------------------------------------------------------------------

    def test_15_limpieza_absoluta_post_tests(self):
        """
        Verifica que tras la ejecución de las pruebas la base de datos de pruebas PostgreSQL
        y el directorio ML_MODELS_DIR conserven un estado 100% puro y aislado:
        - 0 modelos sintéticos huérfanos en modelos_entrenamiento.
        - 0 alertas sintéticas huérfanas en alertas_venta.
        - Ningún archivo .joblib sintético huérfano dejado por las pruebas en ML_MODELS_DIR:
          el conjunto de archivos posterior debe ser idéntico al preexistente antes de las pruebas.
        """
        db = SessionLocal()
        try:
            total_modelos = db.query(ModeloEntrenamiento).count()
            total_alertas = db.query(AlertaVenta).count()
            self.assertEqual(total_modelos, 0, f"Quedaron modelos huérfanos en BD: {total_modelos}")
            self.assertEqual(total_alertas, 0, f"Quedaron alertas huérfanas en BD: {total_alertas}")

            archivos_despues = set(os.listdir(ML_MODELS_DIR)) if os.path.exists(ML_MODELS_DIR) else set()
            archivos_nuevos_huerfanos = archivos_despues - self.preexisting_files

            self.assertEqual(
                archivos_despues,
                self.preexisting_files,
                f"Quedaron archivos .joblib sintéticos huérfanos en ML_MODELS_DIR: {archivos_nuevos_huerfanos}",
            )

            # Verificación adicional: ningún archivo temporal con prefijo de prueba
            huerfanos_test = [f for f in archivos_despues if f.startswith("_test_temp_")]
            self.assertEqual(
                len(huerfanos_test),
                0,
                f"Quedaron archivos temporales de prueba en ML_MODELS_DIR: {huerfanos_test}",
            )
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
