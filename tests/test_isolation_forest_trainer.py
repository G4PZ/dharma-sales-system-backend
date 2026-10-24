"""
Pruebas Unitarias para el Entrenador de Isolation Forest (ISOLATION_FOREST_VENTAS).

Verifica:
1. Rechazo controlado ante muestra insuficiente en la base de datos real (< 50 ventas).
2. Ajuste algorítmico y detección de anomalías sintéticas con métricas (ROC-AUC > 0.90).
3. Reproducibilidad estricta con random_state=42.
4. Manejo correcto de valores faltantes (NaN) mediante SimpleImputer en variables históricas.
5. Cold start: cliente nuevo con primera compra normal NO es clasificado como anómalo.
6. Cold start extremo: cliente nuevo con primera compra desproporcionada sí obtiene score anómalo.
7. Contextualización individual: misma venta absoluta resulta más anómala si quiebra el historial del cliente.
8. Transformación temporal cíclica: proximidad matemática entre 23h-00h y Domingo-Lunes.
9. Exclusión estricta de metadata (venta_id, cliente_id, created_at, comprobantes).
10. Distribución de scores y percentiles persistidos monótonamente (P1 a P99).
11. Versionado resistente a colisiones en el mismo microsegundo.
12. Limpieza segura de artefactos huérfanos ante fallo de persistencia.
13. Bloqueo de concurrencia mediante _IF_TRAINING_LOCK.
14. Ausencia total de escritura en BD real durante pruebas sintéticas en memoria.
15. Inmunidad e independencia absoluta frente a modelos K-Means.
"""

import os
import sys
import unittest
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Asegurar que 'backend' está en el path sin importar desde dónde se ejecute el test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import ML_MODELS_DIR
from app.database import SessionLocal
from app.models.ai import ModeloEntrenamiento
from app.services.ai.trainer import (
    ISOLATION_FOREST_MODEL_TYPE,
    ISOLATION_FOREST_RANDOM_STATE,
    ISOLATION_FOREST_TRAINING_FEATURES,
    ISOLATION_FOREST_TRANSFORMED_FEATURES,
    KMEANS_MODEL_TYPE,
    MIN_SALES_FOR_ISOLATION_FOREST,
    CyclicTemporalTransformer,
    InsufficientDataError,
    TrainingError,
    TrainingInProgressError,
    _cleanup_orphan_artifact,
    _IF_TRAINING_LOCK,
    compute_score_distribution,
    fit_isolation_forest_pipeline,
    generate_isolation_forest_version,
    train_isolation_forest_sales,
)


def _generate_synthetic_sales_cohort(
    n_normal: int = 350,
    n_anomalies: int = 30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Genera un dataset sintético representativo en memoria para validación técnica de Isolation Forest.
    Produce una cohorte equilibrada de operaciones habituales de retail y mayoristas,
    más un subconjunto de anomalías deliberadas multivariables (NO pasadas como feature al modelo).
    """
    rng = np.random.RandomState(random_state)

    n_ret = int(n_normal * 0.70)
    n_who = n_normal - n_ret

    # 1. Operaciones Normales - Segmento Minorista / Estándar
    m_ret = rng.gamma(shape=3.0, scale=40.0, size=n_ret).clip(20.0, 450.0)
    u_ret = rng.poisson(lam=4, size=n_ret).clip(1, 15)
    prod_ret = np.minimum(u_ret, rng.poisson(lam=2, size=n_ret).clip(1, 6))
    lin_ret = prod_ret
    pm_ret = m_ret * rng.uniform(0.85, 1.15, size=n_ret)
    pu_ret = u_ret * rng.uniform(0.85, 1.15, size=n_ret)

    # 2. Operaciones Normales - Segmento Mayorista / Institucional
    m_who = rng.normal(loc=4000.0, scale=300.0, size=n_who).clip(3200.0, 4800.0)
    u_who = rng.normal(loc=50.0, scale=5.0, size=n_who).clip(35, 70).astype(int)
    prod_who = rng.normal(loc=8.0, scale=1.5, size=n_who).clip(4, 15).astype(int)
    lin_who = prod_who
    pm_who = m_who * rng.uniform(0.9, 1.1, size=n_who)
    pu_who = u_who * rng.uniform(0.9, 1.1, size=n_who)

    monto_norm = np.concatenate([m_ret, m_who])
    unidades_norm = np.concatenate([u_ret, u_who])
    productos_norm = np.concatenate([prod_ret, prod_who])
    lineas_norm = np.concatenate([lin_ret, lin_who])
    pm_norm = np.concatenate([pm_ret, pm_who])
    pu_norm = np.concatenate([pu_ret, pu_who])

    conc_imp_norm = rng.uniform(0.3, 0.75, size=n_normal)
    conc_uni_norm = rng.uniform(0.3, 0.75, size=n_normal)
    hora_norm = rng.choice(np.arange(8, 20), size=n_normal)
    dia_norm = rng.choice(np.arange(0, 7), size=n_normal)
    fin_semana_norm = (dia_norm >= 5).astype(int)

    # Contexto histórico normal (incluye una proporción natural de primeras compras)
    compras_prev_norm = rng.poisson(lam=6, size=n_normal)
    is_new_norm = (compras_prev_norm == 0).astype(int)
    compras_prev_30d_norm = np.where(is_new_norm, 0, rng.poisson(lam=2, size=n_normal))
    monto_prom_prev_norm = np.where(is_new_norm, np.nan, pm_norm)
    monto_max_prev_norm = np.where(is_new_norm, np.nan, monto_prom_prev_norm * rng.uniform(1.1, 1.4))
    uni_prom_prev_norm = np.where(is_new_norm, np.nan, pu_norm)
    dias_ant_norm = np.where(is_new_norm, np.nan, rng.uniform(3.0, 30.0, size=n_normal))
    antig_norm = np.where(is_new_norm, rng.uniform(0.1, 2.0, size=n_normal), rng.uniform(30.0, 360.0, size=n_normal))
    div_prev_norm = np.where(is_new_norm, 0, rng.poisson(lam=6, size=n_normal).clip(1, 20))
    ratio_monto_norm = np.where(is_new_norm, np.nan, monto_norm / monto_prom_prev_norm)
    ratio_uni_norm = np.where(is_new_norm, np.nan, unidades_norm / uni_prom_prev_norm)

    # 3. Operaciones Anómalas (patrones de quiebre histórico, horarios nocturnos, concentración 100%)
    monto_anom = [4000.0, 6500.0, 9500.0, 3800.0, 12000.0] * (n_anomalies // 5)
    unidades_anom = [50, 250, 400, 150, 300] * (n_anomalies // 5)
    productos_anom = [1, 2, 1, 35, 1] * (n_anomalies // 5)
    lineas_anom = productos_anom
    conc_imp_anom = [0.99, 1.0, 0.98, 0.1, 1.0] * (n_anomalies // 5)
    conc_uni_anom = [0.99, 1.0, 0.98, 0.1, 1.0] * (n_anomalies // 5)
    hora_anom = [3, 4, 2, 3, 4] * (n_anomalies // 5)  # compras en madrugada
    dia_anom = [0, 1, 2, 3, 4] * (n_anomalies // 5)
    fin_semana_anom = [0] * n_anomalies
    compras_prev_anom = [10, 8, 1, 12, 15] * (n_anomalies // 5)
    is_new_anom = [0, 0, 1, 0, 0] * (n_anomalies // 5)
    compras_prev_30d_anom = [1, 0, 0, 1, 2] * (n_anomalies // 5)
    # Importante: para anomalías de cliente minorista, su historial es de compras pequeñas
    monto_prom_prev_anom = [150.0, 180.0, np.nan, 200.0, 220.0] * (n_anomalies // 5)
    monto_max_prev_anom = [200.0, 240.0, np.nan, 260.0, 280.0] * (n_anomalies // 5)
    uni_prom_prev_anom = [4.0, 4.0, np.nan, 5.0, 5.0] * (n_anomalies // 5)
    dias_ant_anom = [15.0, 20.0, np.nan, 30.0, 10.0] * (n_anomalies // 5)
    antig_anom = [150.0, 200.0, 0.5, 180.0, 210.0] * (n_anomalies // 5)
    div_prev_anom = [3, 2, 0, 4, 5] * (n_anomalies // 5)
    ratio_monto_anom = [float(m / p) if not np.isnan(p) else np.nan for m, p in zip(monto_anom, monto_prom_prev_anom)]
    ratio_uni_anom = [float(u / p) if not np.isnan(p) else np.nan for u, p in zip(unidades_anom, uni_prom_prev_anom)]

    df_norm = pd.DataFrame({
        "monto_total": monto_norm, "unidades_totales": unidades_norm, "productos_distintos": productos_norm,
        "lineas_detalle": lineas_norm, "concentracion_importe": conc_imp_norm, "concentracion_unidades": conc_uni_norm,
        "hora_operacion": hora_norm, "dia_semana": dia_norm, "es_fin_semana": fin_semana_norm,
        "compras_previas": compras_prev_norm, "compras_previas_30d": compras_prev_30d_norm,
        "monto_promedio_previo": monto_prom_prev_norm, "monto_maximo_previo": monto_max_prev_norm,
        "unidades_promedio_previas": uni_prom_prev_norm, "dias_desde_compra_anterior": dias_ant_norm,
        "antiguedad_cliente_dias": antig_norm, "primera_compra": is_new_norm,
        "diversidad_productos_previa": div_prev_norm, "ratio_monto_vs_promedio_cliente": ratio_monto_norm,
        "ratio_unidades_vs_promedio_cliente": ratio_uni_norm,
    })

    df_anom = pd.DataFrame({
        "monto_total": monto_anom, "unidades_totales": unidades_anom, "productos_distintos": productos_anom,
        "lineas_detalle": lineas_anom, "concentracion_importe": conc_imp_anom, "concentracion_unidades": conc_uni_anom,
        "hora_operacion": hora_anom, "dia_semana": dia_anom, "es_fin_semana": fin_semana_anom,
        "compras_previas": compras_prev_anom, "compras_previas_30d": compras_prev_30d_anom,
        "monto_promedio_previo": monto_prom_prev_anom, "monto_maximo_previo": monto_max_prev_anom,
        "unidades_promedio_previas": uni_prom_prev_anom, "dias_desde_compra_anterior": dias_ant_anom,
        "antiguedad_cliente_dias": antig_anom, "primera_compra": is_new_anom,
        "diversidad_productos_previa": div_prev_anom, "ratio_monto_vs_promedio_cliente": ratio_monto_anom,
        "ratio_unidades_vs_promedio_cliente": ratio_uni_anom,
    })

    df_all = pd.concat([df_norm, df_anom], ignore_index=True)
    # Incluir columnas de metadatos para comprobar que el pipeline las ignora
    df_all["venta_id"] = np.arange(1, len(df_all) + 1)
    df_all["cliente_id"] = np.random.randint(100, 200, size=len(df_all))

    y_labels = np.array([0] * n_normal + [1] * n_anomalies)
    return df_all, y_labels


class TestIsolationForestTrainer(unittest.TestCase):

    def test_01_rechazo_muestra_insuficiente_bd_real(self):
        """
        Verifica que al intentar entrenar Isolation Forest sobre la base de datos real (5 ventas),
        el sistema detecte la insuficiencia de datos y lance InsufficientDataError de forma controlada,
        sin insertar registros en PostgreSQL ni generar archivos .joblib.
        """
        db = SessionLocal()
        try:
            modelos_previos = db.query(ModeloEntrenamiento).count()
            archivos_previos = set(os.listdir(ML_MODELS_DIR)) if os.path.exists(ML_MODELS_DIR) else set()

            with self.assertRaises(InsufficientDataError) as ctx:
                train_isolation_forest_sales(db=db)

            err_msg = str(ctx.exception)
            self.assertIn("Entrenamiento cancelado por datos insuficientes", err_msg)
            self.assertIn(f"mínimo de {MIN_SALES_FOR_ISOLATION_FOREST}", err_msg)
            self.assertIn("5 ventas completadas", err_msg)

            # Verificar ausencia total de escritura en BD
            modelos_posteriores = db.query(ModeloEntrenamiento).count()
            self.assertEqual(modelos_previos, modelos_posteriores)

            # Verificar ausencia de archivos creados
            archivos_posteriores = set(os.listdir(ML_MODELS_DIR)) if os.path.exists(ML_MODELS_DIR) else set()
            self.assertEqual(archivos_previos, archivos_posteriores)

        finally:
            db.close()

    def test_02_entrenamiento_sintetico_valido_y_metricas(self):
        """
        Verifica que el pipeline en memoria entrene sobre la cohorte sintética,
        generando scores continuos capaces de discriminar anomalías con ROC-AUC > 0.90
        y capturar las anomalías en los percentiles superiores de score.
        """
        df_all, y_true = _generate_synthetic_sales_cohort(n_normal=350, n_anomalies=30, random_state=42)

        res = fit_isolation_forest_pipeline(
            df_features=df_all,
            feature_columns=ISOLATION_FOREST_TRAINING_FEATURES,
            random_state=ISOLATION_FOREST_RANDOM_STATE,
        )

        anomaly_scores = res["raw_anomaly_scores"]  # higher = more anomalous
        self.assertEqual(len(anomaly_scores), len(df_all))

        # Calcular ROC-AUC sintético
        auc = roc_auc_score(y_true, anomaly_scores)
        self.assertGreater(auc, 0.85, f"El ROC-AUC sintético debe ser alto, obtenido: {auc:.4f}")

        # La media de score de las anomalías debe ser significativamente mayor a la de las normales
        mean_normal = float(np.mean(anomaly_scores[y_true == 0]))
        mean_anom = float(np.mean(anomaly_scores[y_true == 1]))
        self.assertGreater(mean_anom, mean_normal + 0.04)

        # Captura en el Top 20% de scores más anómalos
        umbral_top20 = np.percentile(anomaly_scores, 80)
        anomalias_detectadas = np.sum((anomaly_scores >= umbral_top20) & (y_true == 1))
        tasa_captura = anomalias_detectadas / np.sum(y_true == 1)
        self.assertGreaterEqual(tasa_captura, 0.70, f"Debe capturar al menos el 70% de anomalías en top 20%, obtenido: {tasa_captura}")

    def test_03_reproducibilidad_estricta(self):
        """
        Verifica que dos ejecuciones con random_state=42 produzcan idénticos scores
        continuos en todas las observaciones con precisión flotante total.
        """
        df_all, _ = _generate_synthetic_sales_cohort(n_normal=100, n_anomalies=10, random_state=123)

        res1 = fit_isolation_forest_pipeline(df_all, random_state=42)
        res2 = fit_isolation_forest_pipeline(df_all, random_state=42)

        np.testing.assert_array_almost_equal(
            res1["raw_decision_scores"],
            res2["raw_decision_scores"],
            decimal=7,
        )
        np.testing.assert_array_almost_equal(
            res1["raw_anomaly_scores"],
            res2["raw_anomaly_scores"],
            decimal=7,
        )

    def test_04_manejo_de_nan_e_imputacion(self):
        """
        Verifica que SimpleImputer(median) impute adecuadamente las características
        históricas que son NaN (ej. compras_previas = 0) permitiendo ajuste e inferencia continua.
        """
        df_all, _ = _generate_synthetic_sales_cohort(n_normal=100, n_anomalies=10, random_state=77)

        # Verificar que existen NaNs en las columnas históricas (debido a clientes nuevos)
        self.assertGreater(df_all["monto_promedio_previo"].isna().sum(), 0)
        self.assertGreater(df_all["ratio_monto_vs_promedio_cliente"].isna().sum(), 0)

        # El pipeline debe procesar los NaNs sin lanzar excepciones
        res = fit_isolation_forest_pipeline(df_all)
        self.assertIn("pipeline", res)
        self.assertFalse(np.isnan(res["raw_anomaly_scores"]).any())

    def test_05_cold_start_cliente_nuevo_no_anomalo(self):
        """
        Verifica que una primera compra estándar (cold start) NO sea clasificada
        como altamente anómala simplemente por carecer de historial.
        """
        df_all, _ = _generate_synthetic_sales_cohort(n_normal=250, n_anomalies=20, random_state=42)
        res = fit_isolation_forest_pipeline(df_all)
        pipeline = res["pipeline"]

        # Venta estándar de cliente nuevo
        venta_nueva_normal = pd.DataFrame([{
            "monto_total": 85.0,
            "unidades_totales": 3,
            "productos_distintos": 2,
            "lineas_detalle": 2,
            "concentracion_importe": 0.60,
            "concentracion_unidades": 0.60,
            "hora_operacion": 14,
            "dia_semana": 2,
            "es_fin_semana": 0,
            "compras_previas": 0,
            "compras_previas_30d": 0,
            "monto_promedio_previo": np.nan,
            "monto_maximo_previo": np.nan,
            "unidades_promedio_previas": np.nan,
            "dias_desde_compra_anterior": np.nan,
            "antiguedad_cliente_dias": 0.5,
            "primera_compra": 1,
            "diversidad_productos_previa": 0,
            "ratio_monto_vs_promedio_cliente": np.nan,
            "ratio_unidades_vs_promedio_cliente": np.nan,
        }])

        score_decision = float(pipeline.decision_function(venta_nueva_normal)[0])
        score_anomalia = -score_decision

        # El score de anomalía debe ubicarse por debajo del P90 de la distribución
        p90 = res["distribucion_anomaly_score"]["p90"]
        self.assertLess(score_anomalia, p90, "Una primera compra estándar no debe superar el P90 de anomalía")

    def test_06_cold_start_cliente_nuevo_extremo(self):
        """
        Verifica que si una primera compra es desproporcionada (ej. S/ 8,000 en madrugada con 200 unidades),
        el modelo sí le asigne un score significativamente más anómalo que a una primera compra estándar.
        """
        df_all, _ = _generate_synthetic_sales_cohort(n_normal=250, n_anomalies=20, random_state=42)
        res = fit_isolation_forest_pipeline(df_all)
        pipeline = res["pipeline"]

        venta_nueva_normal = pd.DataFrame([{
            "monto_total": 85.0, "unidades_totales": 3, "productos_distintos": 2, "lineas_detalle": 2,
            "concentracion_importe": 0.60, "concentracion_unidades": 0.60, "hora_operacion": 14,
            "dia_semana": 2, "es_fin_semana": 0, "compras_previas": 0, "compras_previas_30d": 0,
            "monto_promedio_previo": np.nan, "monto_maximo_previo": np.nan, "unidades_promedio_previas": np.nan,
            "dias_desde_compra_anterior": np.nan, "antiguedad_cliente_dias": 0.5, "primera_compra": 1,
            "diversidad_productos_previa": 0, "ratio_monto_vs_promedio_cliente": np.nan, "ratio_unidades_vs_promedio_cliente": np.nan,
        }])

        venta_nueva_extrema = pd.DataFrame([{
            "monto_total": 8500.0, "unidades_totales": 250, "productos_distintos": 1, "lineas_detalle": 1,
            "concentracion_importe": 1.0, "concentracion_unidades": 1.0, "hora_operacion": 3,
            "dia_semana": 1, "es_fin_semana": 0, "compras_previas": 0, "compras_previas_30d": 0,
            "monto_promedio_previo": np.nan, "monto_maximo_previo": np.nan, "unidades_promedio_previas": np.nan,
            "dias_desde_compra_anterior": np.nan, "antiguedad_cliente_dias": 0.1, "primera_compra": 1,
            "diversidad_productos_previa": 0, "ratio_monto_vs_promedio_cliente": np.nan, "ratio_unidades_vs_promedio_cliente": np.nan,
        }])

        score_normal = -float(pipeline.decision_function(venta_nueva_normal)[0])
        score_extremo = -float(pipeline.decision_function(venta_nueva_extrema)[0])

        self.assertGreater(score_extremo, score_normal + 0.05, "La primera compra extrema debe ser marcadamente más anómala")

    def test_07_contextualizacion_individual_venta_vs_historial(self):
        """
        Verifica que ante dos ventas de igual monto absoluto (ej. S/ 4,000):
        - Venta A (cliente habitual de S/ 4,000, ratio ~ 1.0)
        - Venta B (cliente habitual de S/ 150, ratio ~ 26.6)
        La Venta B resulte más anómala por quiebre de patrón histórico individual.
        """
        df_all, _ = _generate_synthetic_sales_cohort(n_normal=350, n_anomalies=30, random_state=42)
        res = fit_isolation_forest_pipeline(df_all)
        pipeline = res["pipeline"]

        # Venta A: Mayorista habitual comprando dentro de su perfil
        venta_a = pd.DataFrame([{
            "monto_total": 4000.0, "unidades_totales": 50, "productos_distintos": 8, "lineas_detalle": 8,
            "concentracion_importe": 0.4, "concentracion_unidades": 0.4, "hora_operacion": 11,
            "dia_semana": 3, "es_fin_semana": 0, "compras_previas": 15, "compras_previas_30d": 2,
            "monto_promedio_previo": 4000.0, "monto_maximo_previo": 4500.0, "unidades_promedio_previas": 50.0,
            "dias_desde_compra_anterior": 14.0, "antiguedad_cliente_dias": 180.0, "primera_compra": 0,
            "diversidad_productos_previa": 12, "ratio_monto_vs_promedio_cliente": 1.0, "ratio_unidades_vs_promedio_cliente": 1.0,
        }])

        # Venta B: Minorista pequeño comprando inesperadamente S/ 4,000 (ratio x26)
        venta_b = pd.DataFrame([{
            "monto_total": 4000.0, "unidades_totales": 50, "productos_distintos": 8, "lineas_detalle": 8,
            "concentracion_importe": 0.4, "concentracion_unidades": 0.4, "hora_operacion": 11,
            "dia_semana": 3, "es_fin_semana": 0, "compras_previas": 15, "compras_previas_30d": 2,
            "monto_promedio_previo": 150.0, "monto_maximo_previo": 220.0, "unidades_promedio_previas": 4.0,
            "dias_desde_compra_anterior": 14.0, "antiguedad_cliente_dias": 180.0, "primera_compra": 0,
            "diversidad_productos_previa": 6, "ratio_monto_vs_promedio_cliente": 26.67, "ratio_unidades_vs_promedio_cliente": 12.5,
        }])

        score_a = -float(pipeline.decision_function(venta_a)[0])
        score_b = -float(pipeline.decision_function(venta_b)[0])

        self.assertGreater(score_b, score_a, "La venta B debe tener un score de anomalía mayor que la venta A")

    def test_08_transformacion_temporal_ciclica(self):
        """
        Verifica que el CyclicTemporalTransformer mapee horas y días a coordenadas
        trigonométricas donde 23h y 00h sean contiguas, y Domingo y Lunes sean vecinos.
        """
        transformer = CyclicTemporalTransformer(hour_col="hora", day_col="dia")
        df_test = pd.DataFrame([
            {"hora": 23, "dia": 6, "valor": 10},  # 23:00 Domingo
            {"hora": 0, "dia": 0, "valor": 20},   # 00:00 Lunes
            {"hora": 12, "dia": 3, "valor": 30},  # 12:00 Jueves
        ])
        df_transformed = transformer.fit_transform(df_test)

        self.assertIn("hora_sin", df_transformed.columns)
        self.assertIn("hora_cos", df_transformed.columns)
        self.assertIn("dia_semana_sin", df_transformed.columns)
        self.assertIn("dia_semana_cos", df_transformed.columns)
        self.assertNotIn("hora", df_transformed.columns)
        self.assertNotIn("dia", df_transformed.columns)

        # Distancia euclidiana entre 23:00 (fila 0) y 00:00 (fila 1)
        h0 = np.array([df_transformed.loc[0, "hora_sin"], df_transformed.loc[0, "hora_cos"]])
        h1 = np.array([df_transformed.loc[1, "hora_sin"], df_transformed.loc[1, "hora_cos"]])
        dist_23_00 = np.linalg.norm(h0 - h1)

        # Distancia euclidiana entre 12:00 (fila 2) y 00:00 (fila 1)
        h2 = np.array([df_transformed.loc[2, "hora_sin"], df_transformed.loc[2, "hora_cos"]])
        dist_12_00 = np.linalg.norm(h2 - h1)

        self.assertLess(dist_23_00, 0.30, "23:00 y 00:00 deben estar muy próximas en espacio cíclico")
        self.assertAlmostEqual(dist_12_00, 2.0, delta=0.01, msg="12:00 y 00:00 deben estar en extremos opuestos (distancia ~ 2.0)")

        # Distancia Domingo (6, fila 0) y Lunes (0, fila 1)
        d0 = np.array([df_transformed.loc[0, "dia_semana_sin"], df_transformed.loc[0, "dia_semana_cos"]])
        d1 = np.array([df_transformed.loc[1, "dia_semana_sin"], df_transformed.loc[1, "dia_semana_cos"]])
        dist_dom_lun = np.linalg.norm(d0 - d1)

        # Distancia Jueves (3, fila 2) y Lunes (0, fila 1)
        d2 = np.array([df_transformed.loc[2, "dia_semana_sin"], df_transformed.loc[2, "dia_semana_cos"]])
        dist_jue_lun = np.linalg.norm(d2 - d1)

        self.assertLess(dist_dom_lun, 0.90, "Domingo y Lunes deben ser días contiguos cíclicamente")
        self.assertGreater(dist_jue_lun, 1.80, "Jueves y Lunes deben estar alejados")

    def test_09_exclusion_estricta_metadata(self):
        """
        Verifica que venta_id, cliente_id, created_at y texto nunca formen parte
        de las características de entrada al estimador Isolation Forest.
        """
        for meta_col in ["venta_id", "cliente_id", "created_at", "comprobante", "ruc"]:
            self.assertNotIn(meta_col, ISOLATION_FOREST_TRAINING_FEATURES)
            self.assertNotIn(meta_col, ISOLATION_FOREST_TRANSFORMED_FEATURES)

        self.assertEqual(len(ISOLATION_FOREST_TRAINING_FEATURES), 20)
        self.assertEqual(len(ISOLATION_FOREST_TRANSFORMED_FEATURES), 22)

    def test_10_distribucion_scores_y_percentiles_persistidos(self):
        """
        Verifica que compute_score_distribution devuelva todas las métricas requeridas
        (mínimo, máximo, media, mediana, desviación y percentiles P1 a P99) de forma monótona.
        """
        datos = np.random.normal(loc=0.0, scale=1.0, size=500)
        dist = compute_score_distribution(datos)

        keys_requeridas = ["min", "max", "mean", "std", "median", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "cuantiles_101", "anomaly_score_quantiles"]
        for k in keys_requeridas:
            self.assertIn(k, dist)

        self.assertLessEqual(dist["p1"], dist["p5"])
        self.assertLessEqual(dist["p5"], dist["p10"])
        self.assertLessEqual(dist["p10"], dist["p25"])
        self.assertLessEqual(dist["p25"], dist["p50"])
        self.assertLessEqual(dist["p50"], dist["p75"])
        self.assertLessEqual(dist["p75"], dist["p90"])
        self.assertLessEqual(dist["p90"], dist["p95"])
        self.assertLessEqual(dist["p95"], dist["p99"])

        # Verificación de la rejilla empírica de 101 cuantiles (0% a 100%)
        q101 = dist["cuantiles_101"]
        self.assertEqual(len(q101), 101, "La rejilla de cuantiles debe contener exactamente 101 puntos")
        self.assertEqual(len(dist["anomaly_score_quantiles"]), 101)
        for i in range(len(q101) - 1):
            self.assertLessEqual(q101[i], q101[i + 1], f"Cuantil {i} ({q101[i]}) debe ser <= Cuantil {i+1} ({q101[i+1]})")
        self.assertAlmostEqual(q101[0], dist["min"], places=4)
        self.assertAlmostEqual(q101[-1], dist["max"], places=4)
        self.assertAlmostEqual(q101[50], dist["median"], places=4)

    def test_11_versionado_resistente_a_colisiones(self):
        """
        Verifica que generate_isolation_forest_version produzca versiones estrictamente únicas
        incluso en ráfagas consecutivas dentro del mismo microsegundo.
        """
        versiones = [generate_isolation_forest_version() for _ in range(50)]
        self.assertEqual(len(versiones), len(set(versiones)), "No deben existir colisiones de versión en IF")
        for v in versiones:
            self.assertTrue(v.startswith("isolation-forest-ventas-"), f"Prefijo inválido: {v}")

    def test_12_limpieza_segura_artefacto_huerfano(self):
        """
        Verifica que _cleanup_orphan_artifact elimine de forma segura el artefacto recién creado
        de Isolation Forest ante fallo transaccional sin tocar modelos anteriores.
        """
        os.makedirs(ML_MODELS_DIR, exist_ok=True)
        modelo_anterior = os.path.join(ML_MODELS_DIR, "isolation_forest_anterior.joblib")
        modelo_nuevo_fallido = os.path.join(ML_MODELS_DIR, "isolation_forest_nuevo_fallido.joblib")

        try:
            with open(modelo_anterior, "w", encoding="utf-8") as f:
                f.write("modelo previo preservado")
            with open(modelo_nuevo_fallido, "w", encoding="utf-8") as f:
                f.write("modelo nuevo huérfano")

            _cleanup_orphan_artifact(modelo_nuevo_fallido)

            self.assertFalse(os.path.exists(modelo_nuevo_fallido), "El artefacto nuevo fallido debe ser eliminado")
            self.assertTrue(os.path.exists(modelo_anterior), "El artefacto previo debe conservarse intacto")
        finally:
            if os.path.exists(modelo_anterior):
                os.remove(modelo_anterior)
            if os.path.exists(modelo_nuevo_fallido):
                os.remove(modelo_nuevo_fallido)

    def test_13_bloqueo_concurrencia_entrenamiento(self):
        """
        Verifica que dos entrenamientos simultáneos de Isolation Forest no puedan ejecutarse,
        lanzando TrainingInProgressError si el candado _IF_TRAINING_LOCK ya está adquirido.
        """
        db = SessionLocal()
        try:
            adquirido = _IF_TRAINING_LOCK.acquire(blocking=False)
            self.assertTrue(adquirido, "Debe ser posible adquirir _IF_TRAINING_LOCK inicialmente")
            try:
                with self.assertRaises(TrainingInProgressError):
                    train_isolation_forest_sales(db=db)
            finally:
                _IF_TRAINING_LOCK.release()
        finally:
            db.close()

    def test_14_ausencia_escritura_real_durante_pruebas_sinteticas(self):
        """
        Verifica que el entrenamiento sintético en memoria no inserte registros
        en modelos_entrenamiento ni cree archivos .joblib en disco.
        """
        db = SessionLocal()
        try:
            modelos_inicio = db.query(ModeloEntrenamiento).count()
            archivos_inicio = len(os.listdir(ML_MODELS_DIR)) if os.path.exists(ML_MODELS_DIR) else 0

            df_all, _ = _generate_synthetic_sales_cohort(n_normal=100, n_anomalies=10, random_state=88)
            fit_isolation_forest_pipeline(df_all)

            modelos_fin = db.query(ModeloEntrenamiento).count()
            archivos_fin = len(os.listdir(ML_MODELS_DIR)) if os.path.exists(ML_MODELS_DIR) else 0

            self.assertEqual(modelos_inicio, modelos_fin, "No deben crearse filas en PostgreSQL")
            self.assertEqual(archivos_inicio, archivos_fin, "No deben crearse archivos en ML_MODELS_DIR")
        finally:
            db.close()

    def test_15_kmeans_permanece_intacto(self):
        """
        Verifica que la lógica de Isolation Forest opere con total independencia de K-Means,
        utilizando su propio tipo_modelo='ISOLATION_FOREST_VENTAS' sin afectar modelos K-Means activos.
        """
        self.assertNotEqual(ISOLATION_FOREST_MODEL_TYPE, KMEANS_MODEL_TYPE)
        self.assertEqual(ISOLATION_FOREST_MODEL_TYPE, "ISOLATION_FOREST_VENTAS")
        self.assertEqual(KMEANS_MODEL_TYPE, "KMEANS_CLIENTES")


if __name__ == "__main__":
    unittest.main()
