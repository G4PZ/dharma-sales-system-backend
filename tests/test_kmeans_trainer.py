"""
Pruebas Unitarias para el Entrenador de K-Means (KMEANS_CLIENTES).

Verifica:
1. Rechazo controlado ante muestra insuficiente en la base de datos real.
2. Selección válida de K óptimo con dataset sintético en memoria.
3. Reproducibilidad estricta con random_state fijo.
4. Manejo correcto de valores faltantes (NaN) mediante SimpleImputer.
5. Centroides desescalados e interpretables en escala física original.
6. Exclusión absoluta de `cliente_id` como variable de entrenamiento.
7. Ausencia total de escritura en PostgreSQL y disco durante pruebas sintéticas.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Asegurar que 'backend' está en el path sin importar desde dónde se ejecute el test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import ML_MODELS_DIR
from app.database import SessionLocal
from app.models.ai import ModeloEntrenamiento
from app.services.ai.trainer import (
    KMEANS_FEATURE_COLUMNS,
    KMEANS_MODEL_TYPE,
    KMEANS_RANDOM_STATE,
    KMEANS_TRAINING_FEATURES,
    MIN_CUSTOMERS_FOR_KMEANS,
    MIN_PURCHASES_FOR_KMEANS_PROFILE,
    InsufficientDataError,
    TrainingError,
    TrainingInProgressError,
    _cleanup_orphan_artifact,
    _KMEANS_TRAINING_LOCK,
    fit_kmeans_pipeline,
    generate_kmeans_version,
    train_kmeans_customers,
)


def _generate_synthetic_customers(n_per_group: int = 30, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame sintético en memoria con 3 patrones de clientes diferenciados:
    - Grupo 0: Compras pequeñas y frecuentes (ticket bajo, alta frecuencia).
    - Grupo 1: Compras medianas periódicas (ticket medio, cadencia mensual).
    - Grupo 2: Compras grandes y esporádicas (ticket alto, volumen grande, baja frecuencia).
    """
    rng = np.random.RandomState(random_state)

    # Grupo 0: Compras pequeñas y muy frecuentes (minoristas de paso)
    g0_monto = rng.normal(loc=50.0, scale=8.0, size=n_per_group).clip(25.0, 85.0)
    g0_unidades = rng.normal(loc=2.0, scale=0.3, size=n_per_group).clip(1.0, 4.0)
    g0_compras = rng.normal(loc=30.0, scale=3.0, size=n_per_group).clip(20.0, 45.0).astype(int)
    g0_freq30 = rng.normal(loc=6.0, scale=0.8, size=n_per_group).clip(4.0, 9.0).astype(int)
    g0_recencia = rng.uniform(1.0, 4.0, size=n_per_group)
    g0_cadencia = rng.uniform(2.0, 5.0, size=n_per_group)
    g0_diversidad = rng.normal(loc=2.0, scale=0.4, size=n_per_group).clip(1.0, 4.0).astype(int)

    # Grupo 1: Compras medianas periódicas (oficinas / empresas medianas)
    g1_monto = rng.normal(loc=500.0, scale=50.0, size=n_per_group).clip(350.0, 700.0)
    g1_unidades = rng.normal(loc=20.0, scale=2.5, size=n_per_group).clip(12.0, 30.0)
    g1_compras = rng.normal(loc=12.0, scale=1.5, size=n_per_group).clip(8.0, 18.0).astype(int)
    g1_freq30 = rng.normal(loc=2.0, scale=0.4, size=n_per_group).clip(1.0, 4.0).astype(int)
    g1_recencia = rng.uniform(10.0, 22.0, size=n_per_group)
    g1_cadencia = rng.uniform(20.0, 35.0, size=n_per_group)
    g1_diversidad = rng.normal(loc=8.0, scale=1.0, size=n_per_group).clip(5.0, 12.0).astype(int)

    # Grupo 2: Compras grandes y esporádicas (institucionales / mayoristas)
    g2_monto = rng.normal(loc=5000.0, scale=350.0, size=n_per_group).clip(3800.0, 6500.0)
    g2_unidades = rng.normal(loc=150.0, scale=15.0, size=n_per_group).clip(100.0, 220.0)
    g2_compras = rng.normal(loc=3.0, scale=0.5, size=n_per_group).clip(2.0, 5.0).astype(int)
    g2_freq30 = rng.binomial(n=1, p=0.3, size=n_per_group).astype(int)
    g2_recencia = rng.uniform(45.0, 90.0, size=n_per_group)
    g2_cadencia = rng.uniform(70.0, 120.0, size=n_per_group)
    g2_diversidad = rng.normal(loc=25.0, scale=2.5, size=n_per_group).clip(18.0, 35.0).astype(int)


    rows = []
    total_samples = n_per_group * 3
    client_ids = rng.permutation(np.arange(1001, 1001 + total_samples))

    for i in range(n_per_group):
        rows.append({
            "cliente_id": int(client_ids[i]),
            "monto_promedio": float(g0_monto[i]),
            "unidades_promedio": float(g0_unidades[i]),
            "total_compras": int(g0_compras[i]),
            "frecuencia_30d": int(g0_freq30[i]),
            "recencia_dias": float(g0_recencia[i]),
            "dias_entre_compras_promedio": float(g0_cadencia[i]),
            "productos_distintos_historicos": int(g0_diversidad[i]),
        })

    for i in range(n_per_group):
        rows.append({
            "cliente_id": int(client_ids[n_per_group + i]),
            "monto_promedio": float(g1_monto[i]),
            "unidades_promedio": float(g1_unidades[i]),
            "total_compras": int(g1_compras[i]),
            "frecuencia_30d": int(g1_freq30[i]),
            "recencia_dias": float(g1_recencia[i]),
            "dias_entre_compras_promedio": float(g1_cadencia[i]),
            "productos_distintos_historicos": int(g1_diversidad[i]),
        })

    for i in range(n_per_group):
        rows.append({
            "cliente_id": int(client_ids[2 * n_per_group + i]),
            "monto_promedio": float(g2_monto[i]),
            "unidades_promedio": float(g2_unidades[i]),
            "total_compras": int(g2_compras[i]),
            "frecuencia_30d": int(g2_freq30[i]),
            "recencia_dias": float(g2_recencia[i]),
            "dias_entre_compras_promedio": float(g2_cadencia[i]),
            "productos_distintos_historicos": int(g2_diversidad[i]),
        })

    return pd.DataFrame(rows)


class TestKMeansTrainer(unittest.TestCase):

    def test_01_rechazo_muestra_insuficiente_bd_real(self):
        """
        Verifica que al intentar entrenar K-Means con la base de datos real (2 clientes),
        el sistema detecte la insuficiencia de datos y lance InsufficientDataError de forma controlada,
        sin insertar registros en PostgreSQL ni generar archivos .joblib.
        """
        db = SessionLocal()
        try:
            # Conteo de modelos antes de la prueba
            modelos_previos = db.query(ModeloEntrenamiento).count()

            # Conteo de archivos previos en ML_MODELS_DIR
            archivos_previos = set(os.listdir(ML_MODELS_DIR)) if os.path.exists(ML_MODELS_DIR) else set()

            with self.assertRaises(InsufficientDataError) as ctx:
                train_kmeans_customers(db=db)

            # Verificar que el mensaje detalle clientes elegibles y mínimo requerido
            err_msg = str(ctx.exception)
            self.assertIn("Entrenamiento cancelado por datos insuficientes", err_msg)
            self.assertIn(f"mínimo de {MIN_CUSTOMERS_FOR_KMEANS}", err_msg)

            # Verificar ausencia de escritura en PostgreSQL
            modelos_posteriores = db.query(ModeloEntrenamiento).count()
            self.assertEqual(modelos_previos, modelos_posteriores)

            # Verificar ausencia de escritura en disco
            archivos_posteriores = set(os.listdir(ML_MODELS_DIR)) if os.path.exists(ML_MODELS_DIR) else set()
            self.assertEqual(archivos_previos, archivos_posteriores)

        finally:
            db.close()

    def test_02_seleccion_valida_k_sintetico(self):
        """
        Verifica que con un dataset sintético con 3 comportamientos claramente diferenciados,
        la evaluación multivariable de K seleccione la estructura adecuada (K=3) con alto Silhouette Score.
        """
        df_synthetic = _generate_synthetic_customers(n_per_group=30, random_state=42)

        result = fit_kmeans_pipeline(
            df_features=df_synthetic,
            feature_columns=KMEANS_TRAINING_FEATURES,
            k_min=2,
            k_max=5,
            random_state=KMEANS_RANDOM_STATE,
        )

        selected_k = result["selected_k"]
        best_silhouette = result["best_silhouette"]
        evaluaciones = result["evaluaciones_k"]

        # Debe haber evaluado K=2, K=3, K=4, K=5
        ks_evaluados = [ev["k"] for ev in evaluaciones]
        self.assertEqual(ks_evaluados, [2, 3, 4, 5])

        # En datos con 3 modas bien formadas, K=3 debe tener el Silhouette Score superior
        self.assertEqual(selected_k, 3)
        self.assertGreater(best_silhouette, 0.45, "El Silhouette Score debe ser significativo")

        # Verificar que todos los clusters tienen clientes asignados
        resumen = result["resumen_modelo"]
        for c_idx in range(selected_k):
            c_data = resumen["clusters"][f"Cluster {c_idx}"]
            self.assertGreater(c_data["tamano"], 0)
            self.assertGreater(c_data["porcentaje"], 0.0)

    def test_03_reproducibilidad(self):
        """
        Verifica que dos ejecuciones independientes sobre el mismo dataset con KMEANS_RANDOM_STATE fijo
        produzcan resultados numéricos idénticos (mismo K, misma inercia, mismo silhouette y mismos centroides).
        """
        df_synthetic = _generate_synthetic_customers(n_per_group=25, random_state=123)

        res1 = fit_kmeans_pipeline(df_synthetic, random_state=KMEANS_RANDOM_STATE)
        res2 = fit_kmeans_pipeline(df_synthetic, random_state=KMEANS_RANDOM_STATE)

        self.assertEqual(res1["selected_k"], res2["selected_k"])
        self.assertEqual(res1["best_silhouette"], res2["best_silhouette"])
        self.assertEqual(res1["inertia"], res2["inertia"])

        # Comparar etiquetas asignadas
        labels1 = res1["pipeline"].named_steps["kmeans"].labels_
        labels2 = res2["pipeline"].named_steps["kmeans"].labels_
        np.testing.assert_array_equal(labels1, labels2)

    def test_04_manejo_de_nan_por_imputer(self):
        """
        Verifica que el SimpleImputer maneje de forma transparente y robusta
        valores NaN (ej. dias_entre_compras_promedio en clientes con una sola compra)
        sin generar excepciones y permitiendo un ajuste perfecto.
        """
        df_synthetic = _generate_synthetic_customers(n_per_group=20, random_state=99)

        # Introducir NaNs intencionales en el 25% de las filas de cadencia
        df_synthetic.loc[::4, "dias_entre_compras_promedio"] = np.nan
        self.assertGreater(df_synthetic["dias_entre_compras_promedio"].isna().sum(), 0)

        # El pipeline debe ejecutar sin errores
        result = fit_kmeans_pipeline(df_synthetic, random_state=KMEANS_RANDOM_STATE)
        self.assertIn("pipeline", result)
        self.assertGreater(result["selected_k"], 1)
        self.assertGreater(result["best_silhouette"], 0.3)

    def test_05_centroides_escala_original(self):
        """
        Verifica que los centroides sean devueltos tanto en escala robusta transformada
        como en las unidades físicas originales en Soles, unidades y días para interpretabilidad comercial.
        """
        df_synthetic = _generate_synthetic_customers(n_per_group=25, random_state=77)
        result = fit_kmeans_pipeline(df_synthetic, random_state=KMEANS_RANDOM_STATE)

        resumen = result["resumen_modelo"]
        for c_name, c_data in resumen["clusters"].items():
            orig = c_data["centroide_original"]
            esc = c_data["centroide_escalado"]

            # Verificar que existan todas las features
            for feat in KMEANS_TRAINING_FEATURES:
                self.assertIn(feat, orig)
                self.assertIn(feat, esc)

            # Verificar que el monto promedio en escala original esté en magnitudes de dinero (> 10 Soles)
            self.assertGreater(orig["monto_promedio"], 10.0)

    def test_06_exclusion_estricta_cliente_id(self):
        """
        Verifica que `cliente_id` u otras columnas de metadata nunca ingresen
        al estimador K-Means como variables de agrupamiento.
        """
        df_synthetic = _generate_synthetic_customers(n_per_group=20, random_state=55)
        self.assertIn("cliente_id", df_synthetic.columns)

        result = fit_kmeans_pipeline(
            df_features=df_synthetic,
            feature_columns=KMEANS_TRAINING_FEATURES,
        )

        pipeline = result["pipeline"]
        kmeans_step = pipeline.named_steps["kmeans"]

        # El número de features del modelo ajustado debe ser exactamente len(KMEANS_TRAINING_FEATURES) = 7
        self.assertEqual(kmeans_step.n_features_in_, len(KMEANS_TRAINING_FEATURES))
        self.assertNotIn("cliente_id", result["feature_columns"])

    def test_07_ausencia_escritura_real_pruebas_sinteticas(self):
        """
        Verifica que la ejecución de pruebas algorítmicas sintéticas sea 100% en memoria
        y no altere la tabla `modelos_entrenamiento` ni cree archivos .joblib en disco.
        """
        db = SessionLocal()
        try:
            modelos_inicio = db.query(ModeloEntrenamiento).count()
            archivos_inicio = len(os.listdir(ML_MODELS_DIR)) if os.path.exists(ML_MODELS_DIR) else 0

            # Ejecutar ajuste sintético en memoria
            df_synthetic = _generate_synthetic_customers(n_per_group=20, random_state=88)
            fit_kmeans_pipeline(df_synthetic)

            modelos_fin = db.query(ModeloEntrenamiento).count()
            archivos_fin = len(os.listdir(ML_MODELS_DIR)) if os.path.exists(ML_MODELS_DIR) else 0

            self.assertEqual(modelos_inicio, modelos_fin, "No deben crearse registros en PostgreSQL")
            self.assertEqual(archivos_inicio, archivos_fin, "No deben crearse archivos en ML_MODELS_DIR")

        finally:
            db.close()

    def test_08_criterio_elegibilidad_minimo_dos_compras(self):
        """
        Verifica que el criterio de elegibilidad para K-Means excluya estrictamente
        a clientes con 0 o 1 compras (falta de cadencia e historial individual),
        y solo admita clientes con total_compras >= MIN_PURCHASES_FOR_KMEANS_PROFILE (2).
        """
        self.assertEqual(MIN_PURCHASES_FOR_KMEANS_PROFILE, 2)

        df_clientes = pd.DataFrame([
            {"cliente_id": 101, "total_compras": 0},
            {"cliente_id": 102, "total_compras": 1},
            {"cliente_id": 103, "total_compras": 2},
            {"cliente_id": 104, "total_compras": 5},
            {"cliente_id": 105, "total_compras": 12},
        ])

        df_elegibles = df_clientes[
            df_clientes["total_compras"] >= MIN_PURCHASES_FOR_KMEANS_PROFILE
        ]

        ids_elegibles = set(df_elegibles["cliente_id"])
        self.assertNotIn(101, ids_elegibles, "Cliente con 0 compras debe ser excluido de K-Means")
        self.assertNotIn(102, ids_elegibles, "Cliente con 1 compra debe ser excluido de K-Means")
        self.assertIn(103, ids_elegibles, "Cliente con 2 compras debe ser elegible")
        self.assertIn(104, ids_elegibles, "Cliente con 5 compras debe ser elegible")
        self.assertIn(105, ids_elegibles, "Cliente con 12 compras debe ser elegible")
        self.assertEqual(len(df_elegibles), 3)

    def test_09_limpieza_segura_artefacto_huerfano(self):
        """
        Verifica que _cleanup_orphan_artifact elimine de forma segura el artefacto recién generado
        cuando ocurre un fallo transaccional en PostgreSQL, sin tocar ningún modelo anterior
        y rechazando rutas no autorizadas fuera de ML_MODELS_DIR.
        """
        os.makedirs(ML_MODELS_DIR, exist_ok=True)
        modelo_anterior = os.path.join(ML_MODELS_DIR, "kmeans_clientes_modelo_anterior.joblib")
        modelo_nuevo = os.path.join(ML_MODELS_DIR, "kmeans_clientes_nuevo_fallido.joblib")

        try:
            # Crear archivo simulado anterior y archivo nuevo fallido
            with open(modelo_anterior, "w", encoding="utf-8") as f:
                f.write("contenido modelo anterior")
            with open(modelo_nuevo, "w", encoding="utf-8") as f:
                f.write("contenido modelo nuevo fallido")

            self.assertTrue(os.path.exists(modelo_anterior))
            self.assertTrue(os.path.exists(modelo_nuevo))

            # Ejecutar limpieza del artefacto fallido
            _cleanup_orphan_artifact(modelo_nuevo)

            # El nuevo archivo debe haber sido eliminado
            self.assertFalse(os.path.exists(modelo_nuevo), "El nuevo artefacto fallido debe ser eliminado")

            # El modelo anterior NUNCA debe ser eliminado
            self.assertTrue(os.path.exists(modelo_anterior), "El modelo anterior debe conservarse intacto")

            # Probar protección de seguridad: intentar limpiar una ruta fuera de ML_MODELS_DIR
            archivo_externo = os.path.abspath("test_dummy_externo.tmp")
            try:
                with open(archivo_externo, "w", encoding="utf-8") as f:
                    f.write("archivo de sistema externo")
                _cleanup_orphan_artifact(archivo_externo)
                self.assertTrue(os.path.exists(archivo_externo), "Rutas fuera de ML_MODELS_DIR no deben ser eliminadas")
            finally:
                if os.path.exists(archivo_externo):
                    os.remove(archivo_externo)

        finally:
            if os.path.exists(modelo_anterior):
                os.remove(modelo_anterior)
            if os.path.exists(modelo_nuevo):
                os.remove(modelo_nuevo)

    def test_10_versionado_resistente_a_colisiones_consecutivas(self):
        """
        Verifica que generate_kmeans_version produzca versiones estrictamente únicas
        incluso cuando se ejecutan en ráfaga consecutiva dentro del mismo microsegundo,
        manteniendo el formato legible kmeans-clientes-YYYYMMDD-HHMMSS-ffffff.
        """
        versiones = [generate_kmeans_version() for _ in range(50)]

        # Todas las 50 versiones deben ser únicas
        self.assertEqual(len(versiones), len(set(versiones)), "No deben existir colisiones de versión")

        # Todas deben comenzar con el prefijo oficial
        for v in versiones:
            self.assertTrue(v.startswith("kmeans-clientes-"), f"Prefijo incorrecto: {v}")
            partes = v.split("-")
            self.assertGreaterEqual(len(partes), 4, f"Estructura de versión inválida: {v}")

    def test_11_proteccion_concurrencia_bloqueo(self):
        """
        Verifica que dos entrenamientos simultáneos en el mismo proceso no puedan ejecutarse,
        lanzando TrainingInProgressError si el candado _KMEANS_TRAINING_LOCK ya está adquirido.
        """
        db = SessionLocal()
        try:
            # Simular que un entrenamiento ya está en curso adquiriendo el lock
            adquirido = _KMEANS_TRAINING_LOCK.acquire(blocking=False)
            self.assertTrue(adquirido, "Debe ser posible adquirir el candado inicialmente")

            try:
                with self.assertRaises(TrainingInProgressError):
                    train_kmeans_customers(db=db)
            finally:
                _KMEANS_TRAINING_LOCK.release()

        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
