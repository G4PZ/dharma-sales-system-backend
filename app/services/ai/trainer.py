"""
Servicio de Entrenamiento y Persistencia de Modelos Machine Learning.

En esta fase se implementa de forma exclusiva el entrenamiento del MODELO 1:
- Algoritmo: K-Means (Segmentación no supervisada de clientes).
- Unidad de análisis: CLIENTE.
- Fuente de datos: `build_customer_features` de `feature_engineering.py`.
- Preprocesamiento integrado: SimpleImputer (mediana) + RobustScaler.
- Selección dinámica de K: Basada en Silhouette Score e Inercia (K = 2, 3, 4...).
- Reproducibilidad: `random_state = 42` y `n_init = 10` fijos y documentados.
- Persistencia Atómica: Artefacto empaquetado (.joblib) en ML_MODELS_DIR.
- Trazabilidad y Transaccionalidad: Registro en `modelos_entrenamiento` y gestión de estado ACTIVO / OBSOLETO.

Regla de Control de Volumen:
- No se realiza entrenamiento con muestras insuficientes. Se requiere un mínimo de clientes
  con historial de compras para construir clusters representativos y evaluar Silhouette Score.
"""

from datetime import datetime, timezone
import logging
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sqlalchemy.orm import Session

from app.config import ML_MODELS_DIR
from app.models.ai import ModeloEntrenamiento
from app.services.ai.feature_engineering import (
    ISOLATION_FOREST_FEATURE_COLUMNS,
    KMEANS_FEATURE_COLUMNS,
    build_customer_features,
    build_sales_features,
)


logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES DE ENTRENAMIENTO K-MEANS
# =============================================================================

KMEANS_MODEL_TYPE = "KMEANS_CLIENTES"

# Historial mínimo de compras requerido para que un cliente sea elegible en K-Means:
# Un cliente con 0 o 1 compras todavía no posee suficiente comportamiento histórico
# para establecer cadencia, intervalo entre compras o un patrón individual mínimamente estable.
# K-Means: Requiere historial mínimo del cliente (total_compras >= 2).
# Isolation Forest (Fase 4): Podrá evaluar transacciones individuales incluso en primera compra
# basándose en la distribución global del comportamiento de ventas de toda la empresa.
MIN_PURCHASES_FOR_KMEANS_PROFILE: int = 2

# Volumen mínimo de clientes elegibles requerido para entrenar con significancia estadística
# y permitir evaluar Silhouette Score en múltiples valores de K sin clusters degenerados.
MIN_CUSTOMERS_FOR_KMEANS: int = 15

# Semilla fija para reproducibilidad matemática estricta
KMEANS_RANDOM_STATE: int = 42

# Número de inicializaciones independientes para evitar mínimos locales
KMEANS_N_INIT: int = 10

# Rango de K candidatos para búsqueda de estructura óptima
K_MIN_CANDIDATE: int = 2
K_MAX_CANDIDATE: int = 5

# Selección metodológica de variables para K-Means:
# Se excluye 'monto_total_acumulado' para evitar redundancia y colinealidad matemática
# con (monto_promedio * total_compras), lo que sobreponderaría artificialmente el volumen monetario.
# Se excluye 'antiguedad_cliente_dias' para que la antigüedad del registro en el sistema
# no distorsione la segmentación de actividad comercial real (capturada por recencia y frecuencia).
KMEANS_TRAINING_FEATURES: List[str] = [
    "monto_promedio",
    "unidades_promedio",
    "total_compras",
    "frecuencia_30d",
    "recencia_dias",
    "dias_entre_compras_promedio",
    "productos_distintos_historicos",
]

# =============================================================================
# CONSTANTES DE ENTRENAMIENTO ISOLATION FOREST (VENTAS)
# =============================================================================

ISOLATION_FOREST_MODEL_TYPE = "ISOLATION_FOREST_VENTAS"

# Umbral operativo mínimo de ventas completadas requeridas para entrenar:
# Se define en 50 ventas como guard de ingeniería para asegurar un dataset
# multivariable mínimamente representativo que evite sobreajuste en distribuciones ralas.
# Nota metodológica: 50 no es una ley estadística universal, es un guard de ingeniería del sistema.
MIN_SALES_FOR_ISOLATION_FOREST: int = 50

# Hiperparámetros de scikit-learn fijos y reproducibles
ISOLATION_FOREST_RANDOM_STATE: int = 42
ISOLATION_FOREST_N_ESTIMATORS: int = 200
ISOLATION_FOREST_CONTAMINATION: str = "auto"
ISOLATION_FOREST_MAX_SAMPLES: str = "auto"
ISOLATION_FOREST_N_JOBS: int = -1

# Selección de características de entrada para Isolation Forest (20 variables de entrada)
# Excluye metadatos relacionales: venta_id, cliente_id, created_at, RUC, comprobante, serie, etc.
ISOLATION_FOREST_TRAINING_FEATURES: List[str] = [
    # 1. Características intrínsecas de la venta actual
    "monto_total",
    "unidades_totales",
    "productos_distintos",
    "lineas_detalle",
    "concentracion_importe",
    "concentracion_unidades",
    # 2. Variables temporales locales (hora_operacion y dia_semana se transformarán cíclicamente a sin/cos)
    "hora_operacion",
    "dia_semana",
    "es_fin_semana",
    # 3. Contexto histórico del cliente (estrictamente anterior a la venta)
    "compras_previas",
    "compras_previas_30d",
    "monto_promedio_previo",
    "monto_maximo_previo",
    "unidades_promedio_previas",
    "dias_desde_compra_anterior",
    "antiguedad_cliente_dias",
    "primera_compra",
    "diversidad_productos_previa",
    # 4. Ratios contextuales relativos (venta vs promedio histórico individual)
    "ratio_monto_vs_promedio_cliente",
    "ratio_unidades_vs_promedio_cliente",
]

# Lista final de 22 variables tras la codificación cíclica (elimina hora y día lineales)
ISOLATION_FOREST_TRANSFORMED_FEATURES: List[str] = [
    "monto_total",
    "unidades_totales",
    "productos_distintos",
    "lineas_detalle",
    "concentracion_importe",
    "concentracion_unidades",
    "hora_sin",
    "hora_cos",
    "dia_semana_sin",
    "dia_semana_cos",
    "es_fin_semana",
    "compras_previas",
    "compras_previas_30d",
    "monto_promedio_previo",
    "monto_maximo_previo",
    "unidades_promedio_previas",
    "dias_desde_compra_anterior",
    "antiguedad_cliente_dias",
    "primera_compra",
    "diversidad_productos_previa",
    "ratio_monto_vs_promedio_cliente",
    "ratio_unidades_vs_promedio_cliente",
]


# =============================================================================
# CONTROL DE CONCURRENCIA Y EXCEPCIONES DE ENTRENAMIENTO
# =============================================================================

# Bloqueos a nivel de proceso independientes para evitar concurrencias simultáneas
_KMEANS_TRAINING_LOCK = threading.Lock()
_IF_TRAINING_LOCK = threading.Lock()

# Secuencia para garantizar versiones estrictamente únicas en colisiones de microsegundo
_LAST_VERSION_TIMESTAMP: str = ""
_VERSION_SEQ: int = 0
_VERSION_LOCK = threading.Lock()

_IF_LAST_VERSION_TIMESTAMP: str = ""
_IF_VERSION_SEQ: int = 0
_IF_VERSION_LOCK = threading.Lock()


class TrainingError(Exception):
    """Excepción base para fallos durante el proceso de entrenamiento de IA."""
    pass


class InsufficientDataError(TrainingError):
    """Excepción lanzada cuando el volumen de clientes elegibles no alcanza el umbral mínimo."""
    pass


class TrainingInProgressError(TrainingError):
    """Excepción lanzada cuando se intenta iniciar un entrenamiento mientras otro ya está en ejecución."""
    pass


# =============================================================================
# PIPELINE Y LÓGICA MATEMÁTICA PURA (AISLABLE PARA PRUEBAS EN MEMORIA)
# =============================================================================

def fit_kmeans_pipeline(
    df_features: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    k_min: int = K_MIN_CANDIDATE,
    k_max: int = K_MAX_CANDIDATE,
    random_state: int = KMEANS_RANDOM_STATE,
    n_init: int = KMEANS_N_INIT,
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de preprocesamiento, evaluación de hiperparámetros
    y entrenamiento de K-Means en memoria sobre un DataFrame.

    Pasos:
    1. Filtrado de columnas exclusivas de entrenamiento (nunca incluye cliente_id ni texto).
    2. Imputación de valores faltantes (SimpleImputer con estrategia 'median' para cadencia).
    3. Escalado robusto (RobustScaler para resistir clientes con importes o volúmenes extremos).
    4. Evaluación de candidatos K calculando Inercia y Silhouette Score.
    5. Selección de K óptimo por mejor Silhouette Score (con verificación de no degeneración).
    6. Construcción y ajuste del Pipeline final serializable.
    7. Cálculo de centroides interpretables en escala original.
    """
    if feature_columns is None:
        feature_columns = KMEANS_TRAINING_FEATURES

    # Validar que las columnas requeridas existan en el DataFrame
    missing_cols = [col for col in feature_columns if col not in df_features.columns]
    if missing_cols:
        raise TrainingError(f"Faltan columnas requeridas en el dataset de entrada: {missing_cols}")

    # Extraer estrictamente la matriz numérica de entrenamiento (sin identificadores)
    X_raw = df_features[feature_columns].copy().astype(float)
    n_samples = len(X_raw)

    if n_samples < 2:
        raise InsufficientDataError(
            f"Se requieren al menos 2 muestras para evaluar clusters; recibidas: {n_samples}."
        )

    # Adaptar rango de K al tamaño real del dataset (K debe ser < n_samples)
    upper_k = min(k_max, n_samples - 1)
    if upper_k < k_min:
        raise InsufficientDataError(
            f"El número de muestras ({n_samples}) no permite evaluar K en el rango [{k_min}, {upper_k}]."
        )

    # 1. Ajustar Imputador de mediana (robusto ante outliers de cadencia para compras únicas)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    # 2. Ajustar RobustScaler (escalado por mediana y rango intercuartil IQR)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 3. Evaluar candidatos K en el espacio escalado
    evaluaciones: List[Dict[str, Any]] = []
    best_k: Optional[int] = None
    best_score: float = -1.0
    best_kmeans: Optional[KMeans] = None

    for k in range(k_min, upper_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_scaled)

        inertia = float(km.inertia_)

        # Silhouette score requiere al menos 2 clusters y al menos 2 muestras por cluster para tener sentido
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            score = -1.0
        else:
            score = float(silhouette_score(X_scaled, labels))

        evaluacion = {
            "k": k,
            "inertia": round(inertia, 4),
            "silhouette": round(score, 4),
        }
        evaluaciones.append(evaluacion)

        # Selección principal: mayor Silhouette Score
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = km

    # Fallback si ninguna configuración arrojó score positivo
    if best_k is None or best_kmeans is None:
        best_k = k_min
        best_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=n_init)
        best_kmeans.fit(X_scaled)
        best_score = float(silhouette_score(X_scaled, best_kmeans.labels_))

    # 4. Construir Pipeline completo de preprocesamiento + modelo para inferencia exacta
    final_pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("kmeans", best_kmeans),
    ])

    # 5. Obtener centroides en escala original para interpretabilidad comercial
    centers_scaled = best_kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)

    # 6. Describir perfiles y tamaños de cada cluster (Cluster 0, Cluster 1, ...)
    final_labels = best_kmeans.labels_
    cluster_counts = pd.Series(final_labels).value_counts().to_dict()

    resumen_clusters: Dict[str, Any] = {
        "features": feature_columns,
        "n_clusters": best_k,
        "total_muestras": n_samples,
        "clusters": {},
    }

    for c_idx in range(best_k):
        c_name = f"Cluster {c_idx}"
        count = int(cluster_counts.get(c_idx, 0))
        pct = round(float(count / n_samples) * 100.0, 2)

        centroide_orig = {
            col: round(float(val), 4)
            for col, val in zip(feature_columns, centers_original[c_idx])
        }
        centroide_esc = {
            col: round(float(val), 4)
            for col, val in zip(feature_columns, centers_scaled[c_idx])
        }

        resumen_clusters["clusters"][c_name] = {
            "cluster_id": c_idx,
            "tamano": count,
            "porcentaje": pct,
            "centroide_original": centroide_orig,
            "centroide_escalado": centroide_esc,
        }

    return {
        "pipeline": final_pipeline,
        "selected_k": best_k,
        "best_silhouette": round(best_score, 4),
        "inertia": round(float(best_kmeans.inertia_), 4),
        "evaluaciones_k": evaluaciones,
        "feature_columns": feature_columns,
        "total_muestras": n_samples,
        "resumen_modelo": resumen_clusters,
    }


# =============================================================================
# SERIALIZACIÓN ATÓMICA Y GESTIÓN DE ARTEFACTOS JOB_LIB
# =============================================================================

def _save_artifact_atomically(artifact: Dict[str, Any], target_path: str) -> None:
    """
    Escribe un artefacto .joblib de forma atómica en disco para evitar archivos parciales o corruptos:
    1. Escribe en un archivo temporal en el mismo directorio.
    2. Utiliza os.replace para el reemplazo atómico en el sistema de archivos.
    """
    dir_path = os.path.dirname(target_path)
    os.makedirs(dir_path, exist_ok=True)

    temp_path = target_path + f".tmp_{os.getpid()}_{int(datetime.now(timezone.utc).timestamp())}"
    try:
        joblib.dump(artifact, temp_path, compress=3)
        os.replace(temp_path, target_path)
    except Exception as exc:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise TrainingError(f"Error al escribir el artefacto en disco: {exc}") from exc


def _cleanup_orphan_artifact(artifact_path: Optional[str]) -> None:
    """
    Elimina de forma segura y controlada un artefacto recién creado si falla la persistencia en BD,
    evitando archivos huérfanos en ML_MODELS_DIR sin registro correspondiente en modelos_entrenamiento.

    Garantías de seguridad:
    1. Solo se elimina el archivo específico indicado para esa ejecución.
    2. Nunca se tocan ni eliminan modelos anteriores ni el directorio base.
    3. La ruta debe residir obligatoriamente dentro de ML_MODELS_DIR.
    4. La operación está protegida ante excepciones de I/O y debidamente registrada en los logs.
    """
    if not artifact_path:
        return

    try:
        norm_models_dir = os.path.realpath(os.path.abspath(ML_MODELS_DIR))
        norm_artifact = os.path.realpath(os.path.abspath(artifact_path))

        # Validación estricta: asegurar que el archivo resida dentro del directorio de modelos
        if not norm_artifact.startswith(norm_models_dir) or norm_artifact == norm_models_dir:
            logger.warning(
                f"[SEGURIDAD] Intento de limpieza de artefacto fuera de ML_MODELS_DIR ignorado: '{artifact_path}'"
            )
            return

        if os.path.exists(norm_artifact):
            os.remove(norm_artifact)
            logger.warning(
                f"Artefacto huérfano eliminado exitosamente tras fallo de persistencia en BD: '{norm_artifact}'"
            )
    except Exception as exc:
        logger.error(
            f"Error al intentar eliminar el artefacto huérfano '{artifact_path}': {exc}"
        )


def generate_kmeans_version(reference_time: Optional[datetime] = None) -> str:
    """
    Genera un identificador de versión único, reproducible y cronológico para K-Means.
    Formato: kmeans-clientes-YYYYMMDD-HHMMSS-ffffff
    Resistente a dos entrenamientos ejecutados en el mismo segundo manteniendo orden temporal
    y legibilidad humana directa. En caso de colisión en el mismo microsegundo, anexa secuencia.
    """
    global _LAST_VERSION_TIMESTAMP, _VERSION_SEQ
    now_utc = reference_time or datetime.now(timezone.utc)
    ts = now_utc.strftime("%Y%m%d-%H%M%S-%f")

    with _VERSION_LOCK:
        if ts == _LAST_VERSION_TIMESTAMP:
            _VERSION_SEQ += 1
            return f"kmeans-clientes-{ts}-{_VERSION_SEQ:02d}"
        else:
            _LAST_VERSION_TIMESTAMP = ts
            _VERSION_SEQ = 0
            return f"kmeans-clientes-{ts}"


def generate_isolation_forest_version(reference_time: Optional[datetime] = None) -> str:
    """
    Genera un identificador de versión único, reproducible y cronológico para Isolation Forest.
    Formato: isolation-forest-ventas-YYYYMMDD-HHMMSS-ffffff
    Resistente a dos entrenamientos ejecutados en el mismo segundo manteniendo orden temporal
    y legibilidad humana directa. En caso de colisión en el mismo microsegundo, anexa secuencia.
    """
    global _IF_LAST_VERSION_TIMESTAMP, _IF_VERSION_SEQ
    now_utc = reference_time or datetime.now(timezone.utc)
    ts = now_utc.strftime("%Y%m%d-%H%M%S-%f")

    with _IF_VERSION_LOCK:
        if ts == _IF_LAST_VERSION_TIMESTAMP:
            _IF_VERSION_SEQ += 1
            return f"isolation-forest-ventas-{ts}-{_IF_VERSION_SEQ:02d}"
        else:
            _IF_LAST_VERSION_TIMESTAMP = ts
            _IF_VERSION_SEQ = 0
            return f"isolation-forest-ventas-{ts}"


# =============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO PARA DHARMA SALES SYSTEM
# =============================================================================

def train_kmeans_customers(
    db: Session,
    reference_datetime: Optional[datetime] = None,
    min_customers: int = MIN_CUSTOMERS_FOR_KMEANS,
) -> Dict[str, Any]:
    """
    Ejecuta el flujo completo de entrenamiento real de K-Means sobre la base de datos de Dharma:
    1. Protege la concurrencia a nivel de proceso mediante un bloqueo exclusivo.
    2. Extrae el dataset de clientes mediante `build_customer_features`.
    3. Filtra clientes elegibles (total_compras >= MIN_PURCHASES_FOR_KMEANS_PROFILE = 2).
       Clientes con 0 o 1 compras quedan excluidos del entrenamiento de perfiles por falta de historial.
    4. Valida el umbral mínimo de clientes elegibles (min_customers = 15).
       Si no se alcanza, RECHAZA el entrenamiento sin entrenar, sin guardar archivo y sin insertar en BD.
    5. Ajusta el pipeline no supervisado (Imputación + RobustScaler + KMeans multievaluado).
    6. Versiona y serializa atómicamente el artefacto `.joblib` en `ML_MODELS_DIR`.
    7. Registra transaccionalmente la metadata en `modelos_entrenamiento` y activa la nueva versión,
       marcando los modelos K-Means anteriores como OBSOLETO bajo bloqueo de fila (with_for_update).
    8. En caso de error en BD tras crear el archivo, limpia de forma segura el artefacto huérfano.
    9. Retorna el resultado estructurado y la entidad persistida.
    """
    # 1. Protección contra concurrencia simultánea
    if not _KMEANS_TRAINING_LOCK.acquire(blocking=False):
        raise TrainingInProgressError(
            "Ya existe un proceso de entrenamiento de K-Means en ejecución. "
            "Espere a que finalice antes de iniciar uno nuevo."
        )

    artifact_path: Optional[str] = None

    try:
        logger.info("Iniciando verificación de datos para entrenamiento de K-Means Clientes...")

        # 2. Extracción de características históricas de clientes
        df_raw = build_customer_features(db=db, reference_datetime=reference_datetime)
        total_clientes_sistema = len(df_raw)

        # 3. Filtrar clientes elegibles: mínimo MIN_PURCHASES_FOR_KMEANS_PROFILE (2) compras completadas
        # Clientes con 0 o 1 compras no poseen historial suficiente para calcular cadencia ni patrones estables.
        # Nota conceptual: Isolation Forest sí podrá analizar transacciones individuales de estos clientes
        # basándose en la distribución global del comportamiento de ventas de toda la empresa.
        df_elegibles = df_raw[df_raw["total_compras"] >= MIN_PURCHASES_FOR_KMEANS_PROFILE].copy()
        clientes_elegibles = len(df_elegibles)
        clientes_excluidos = total_clientes_sistema - clientes_elegibles

        logger.info(
            f"Clientes totales: {total_clientes_sistema} | "
            f"Elegibles (compras >= {MIN_PURCHASES_FOR_KMEANS_PROFILE}): {clientes_elegibles} | "
            f"Excluidos (< {MIN_PURCHASES_FOR_KMEANS_PROFILE} compras): {clientes_excluidos}"
        )

        # 4. Validación estricta de volumen mínimo de datos
        if clientes_elegibles < min_customers:
            error_msg = (
                f"Entrenamiento cancelado por datos insuficientes: se encontraron {clientes_elegibles} "
                f"clientes elegibles con al menos {MIN_PURCHASES_FOR_KMEANS_PROFILE} compras, pero se requiere "
                f"un mínimo de {min_customers} para construir clusters representativos y evaluar Silhouette Score de forma confiable."
            )
            logger.warning(error_msg)
            raise InsufficientDataError(error_msg)

        # 5. Ajuste del pipeline y selección de K en memoria
        version = generate_kmeans_version()
        logger.info(f"Entrenando pipeline K-Means versión '{version}' con {clientes_elegibles} clientes...")

        fit_result = fit_kmeans_pipeline(
            df_features=df_elegibles,
            feature_columns=KMEANS_TRAINING_FEATURES,
            k_min=K_MIN_CANDIDATE,
            k_max=K_MAX_CANDIDATE,
            random_state=KMEANS_RANDOM_STATE,
            n_init=KMEANS_N_INIT,
        )

        selected_k = fit_result["selected_k"]
        best_silhouette = fit_result["best_silhouette"]
        inertia = fit_result["inertia"]
        resumen_modelo = fit_result["resumen_modelo"]

        # 6. Definir ruta y serializar artefacto atómicamente
        artifact_filename = f"{version}.joblib"
        artifact_path = os.path.join(ML_MODELS_DIR, artifact_filename)

        artifact_payload = {
            "pipeline": fit_result["pipeline"],
            "feature_columns": KMEANS_TRAINING_FEATURES,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": clientes_elegibles,
            "n_clusters": selected_k,
            "metrics": {
                "selected_k": selected_k,
                "best_silhouette": best_silhouette,
                "inertia": inertia,
                "evaluaciones_k": fit_result["evaluaciones_k"],
            },
            "resumen_modelo": resumen_modelo,
        }

        logger.info(f"Serializando artefacto en '{artifact_path}'...")
        _save_artifact_atomically(artifact_payload, artifact_path)

        # 7. Persistencia transaccional en PostgreSQL con bloqueo de filas activas
        try:
            # Desactivar bajo bloqueo (with_for_update) cualquier modelo K-Means anterior
            modelos_activos = (
                db.query(ModeloEntrenamiento)
                .filter(
                    ModeloEntrenamiento.tipo_modelo == KMEANS_MODEL_TYPE,
                    ModeloEntrenamiento.estado == "ACTIVO",
                )
                .with_for_update()
                .all()
            )
            for mod_antiguo in modelos_activos:
                mod_antiguo.estado = "OBSOLETO"

            # Registrar el nuevo modelo activo
            nuevo_modelo = ModeloEntrenamiento(
                tipo_modelo=KMEANS_MODEL_TYPE,
                version=version,
                parametros={
                    "features_utilizadas": KMEANS_TRAINING_FEATURES,
                    "selected_k": selected_k,
                    "k_candidates": [ev["k"] for ev in fit_result["evaluaciones_k"]],
                    "random_state": KMEANS_RANDOM_STATE,
                    "n_init": KMEANS_N_INIT,
                    "imputer": "SimpleImputer(strategy='median')",
                    "scaler": "RobustScaler()",
                    "min_compras_perfil": MIN_PURCHASES_FOR_KMEANS_PROFILE,
                },
                metricas={
                    "selected_k": selected_k,
                    "best_silhouette": best_silhouette,
                    "inertia": inertia,
                    "evaluaciones_k": fit_result["evaluaciones_k"],
                },
                total_muestras=clientes_elegibles,
                ruta_artefacto=artifact_path,
                estado="ACTIVO",
                resumen_modelo=resumen_modelo,
            )
            db.add(nuevo_modelo)
            db.commit()
            db.refresh(nuevo_modelo)
            logger.info(f"Modelo K-Means versión '{version}' registrado exitosamente como ACTIVO (ID: {nuevo_modelo.id}).")

        except Exception as db_exc:
            db.rollback()
            logger.error(f"Fallo transaccional al persistir modelo en base de datos: {db_exc}")
            # Limpiar el artefacto recién creado para evitar huérfanos sin registro en BD
            _cleanup_orphan_artifact(artifact_path)
            raise TrainingError(f"Error al registrar metadatos en modelos_entrenamiento: {db_exc}") from db_exc

        return {
            "status": "ENTRENADO_Y_ACTIVO",
            "modelo_id": nuevo_modelo.id,
            "version": version,
            "selected_k": selected_k,
            "best_silhouette": best_silhouette,
            "inertia": inertia,
            "total_muestras": clientes_elegibles,
            "ruta_artefacto": artifact_path,
            "evaluaciones_k": fit_result["evaluaciones_k"],
            "resumen_modelo": resumen_modelo,
        }

    finally:
        _KMEANS_TRAINING_LOCK.release()


# Alias de compatibilidad con fases anteriores
train_kmeans_clients = train_kmeans_customers


# =============================================================================
# TRANSFORMADOR TEMPORAL CÍCLICO PARA ISOLATION FOREST
# =============================================================================

class CyclicTemporalTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador de scikit-learn para variables temporales continuas hacia coordenadas cíclicas:
    - hora_operacion (0-23) -> hora_sin, hora_cos (periodo 24.0)
    - dia_semana (0-6) -> dia_semana_sin, dia_semana_cos (periodo 7.0)

    Resuelve la discontinuidad de frontera (ej. 23:00 con 00:00 y Domingo con Lunes).
    Elimina las columnas originales lineales para evitar duplicación y redundancia dimensional.
    """
    def __init__(
        self,
        hour_col: str = "hora_operacion",
        day_col: str = "dia_semana",
    ):
        self.hour_col = hour_col
        self.day_col = day_col
        self.feature_names_in_: Optional[List[str]] = None
        self.feature_names_out_: Optional[List[str]] = None

    def fit(self, X: Any, y: Any = None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"feat_{i}" for i in range(X.shape[1])]

        out_names: List[str] = []
        for col in self.feature_names_in_:
            if col == self.hour_col:
                out_names.extend(["hora_sin", "hora_cos"])
            elif col == self.day_col:
                out_names.extend(["dia_semana_sin", "dia_semana_cos"])
            else:
                out_names.append(col)
        self.feature_names_out_ = out_names
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_in_)

        res_dict: Dict[str, Any] = {}
        for col in (self.feature_names_in_ or df.columns):
            if col == self.hour_col:
                rad_h = 2.0 * np.pi * df[col].astype(float) / 24.0
                res_dict["hora_sin"] = np.sin(rad_h)
                res_dict["hora_cos"] = np.cos(rad_h)
            elif col == self.day_col:
                rad_d = 2.0 * np.pi * df[col].astype(float) / 7.0
                res_dict["dia_semana_sin"] = np.sin(rad_d)
                res_dict["dia_semana_cos"] = np.cos(rad_d)
            elif col in df.columns:
                res_dict[col] = df[col]

        return pd.DataFrame(res_dict, index=df.index)

    def get_feature_names_out(self, input_features: Any = None) -> List[str]:
        return list(self.feature_names_out_ or [])


def compute_score_distribution(scores: np.ndarray) -> Dict[str, Any]:
    """
    Calcula los estadísticos descriptivos, percentiles ejecutivos y una rejilla empírica
    de 101 cuantiles (0% a 100% en pasos de 1%) de la distribución de scores continuos
    en el conjunto de entrenamiento para permitir la posterior contextualización precisa en inferencia.
    """
    scores_clean = scores[~np.isnan(scores)]
    if len(scores_clean) == 0:
        return {}

    # Rejilla empírica fina de 101 cuantiles (0%, 1%, ..., 99%, 100%) para interpolación continua
    quantiles_grid = [round(float(q), 5) for q in np.percentile(scores_clean, np.linspace(0, 100, 101))]

    return {
        "min": round(float(np.min(scores_clean)), 5),
        "max": round(float(np.max(scores_clean)), 5),
        "mean": round(float(np.mean(scores_clean)), 5),
        "std": round(float(np.std(scores_clean)), 5),
        "median": round(float(np.median(scores_clean)), 5),
        "p1": round(float(np.percentile(scores_clean, 1)), 5),
        "p5": round(float(np.percentile(scores_clean, 5)), 5),
        "p10": round(float(np.percentile(scores_clean, 10)), 5),
        "p25": round(float(np.percentile(scores_clean, 25)), 5),
        "p50": round(float(np.percentile(scores_clean, 50)), 5),
        "p75": round(float(np.percentile(scores_clean, 75)), 5),
        "p90": round(float(np.percentile(scores_clean, 90)), 5),
        "p95": round(float(np.percentile(scores_clean, 95)), 5),
        "p99": round(float(np.percentile(scores_clean, 99)), 5),
        "cuantiles_101": quantiles_grid,
        "anomaly_score_quantiles": quantiles_grid,
    }


# =============================================================================
# PIPELINE Y AJUSTE EN MEMORIA PARA ISOLATION FOREST
# =============================================================================

def fit_isolation_forest_pipeline(
    df_features: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    n_estimators: int = ISOLATION_FOREST_N_ESTIMATORS,
    contamination: Any = ISOLATION_FOREST_CONTAMINATION,
    max_samples: Any = ISOLATION_FOREST_MAX_SAMPLES,
    random_state: int = ISOLATION_FOREST_RANDOM_STATE,
    n_jobs: int = ISOLATION_FOREST_N_JOBS,
) -> Dict[str, Any]:
    """
    Ajusta en memoria el pipeline de Isolation Forest para operaciones de venta:
    1. Filtra las 20 columnas de entrada numéricas (excluyendo IDs y metadata).
    2. Aplica transformación temporal cíclica (hora_sin/cos, dia_semana_sin/cos).
    3. Imputa valores faltantes en variables de historial mediante mediana aprendida.
    4. Ajusta el ensamble IsolationForest de scikit-learn.
    5. Calcula la distribución de scores continuos (decision_function y anomaly_score).
    """
    if feature_columns is None:
        feature_columns = ISOLATION_FOREST_TRAINING_FEATURES

    missing = [c for c in feature_columns if c not in df_features.columns]
    if missing:
        raise TrainingError(f"Faltan columnas requeridas para Isolation Forest: {missing}")

    X_in = df_features[feature_columns].copy()
    n_samples = len(X_in)

    pipeline = Pipeline([
        ("cyclic", CyclicTemporalTransformer(hour_col="hora_operacion", day_col="dia_semana")),
        ("imputer", SimpleImputer(strategy="median")),
        (
            "isolation_forest",
            IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                max_samples=max_samples,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        ),
    ])

    pipeline.fit(X_in)

    # decision_function: valores menores (negativos) = más anómalos; valores mayores = más normales
    raw_decision = pipeline.decision_function(X_in)
    # anomaly_score (-decision_function): valores mayores = más anómalos
    raw_anomaly = -raw_decision

    dist_decision = compute_score_distribution(raw_decision)
    dist_anomaly = compute_score_distribution(raw_anomaly)

    return {
        "pipeline": pipeline,
        "n_samples": n_samples,
        "raw_decision_scores": raw_decision,
        "raw_anomaly_scores": raw_anomaly,
        "distribucion_decision_function": dist_decision,
        "distribucion_anomaly_score": dist_anomaly,
        "input_features": feature_columns,
        "transformed_features": ISOLATION_FOREST_TRANSFORMED_FEATURES,
    }


# =============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO ISOLATION FOREST
# =============================================================================

def train_isolation_forest_sales(
    db: Session,
    reference_datetime: Optional[datetime] = None,
    min_sales: int = MIN_SALES_FOR_ISOLATION_FOREST,
    n_estimators: int = ISOLATION_FOREST_N_ESTIMATORS,
    contamination: Any = ISOLATION_FOREST_CONTAMINATION,
    random_state: int = ISOLATION_FOREST_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Ejecuta el flujo completo de entrenamiento real de Isolation Forest para operaciones de venta:
    1. Bloqueo de concurrencia exclusivo en proceso (_IF_TRAINING_LOCK).
    2. Extracción de características a nivel de venta mediante build_sales_features(db).
    3. Validación del guard de volumen mínimo (min_sales = 50). Si no se alcanza,
       RECHAZA el entrenamiento controladamente sin persistir archivo ni registrar en BD.
    4. Ajuste del pipeline en memoria (cíclico + SimpleImputer + IsolationForest).
    5. Cálculo de métricas continuas de score y percentiles empíricos de referencia.
    6. Versionado y serialización atómica en ML_MODELS_DIR.
    7. Persistencia transaccional en modelos_entrenamiento con bloqueo pesimista (with_for_update)
       para modelos previos ISOLATION_FOREST_VENTAS, dejando intactos los modelos K-Means.
    8. Limpieza garantizada de artefacto huérfano ante fallo de commit.
    """
    if not _IF_TRAINING_LOCK.acquire(blocking=False):
        raise TrainingInProgressError(
            "Ya existe un proceso de entrenamiento de Isolation Forest en ejecución. "
            "Espere a que finalice antes de iniciar uno nuevo."
        )

    artifact_path: Optional[str] = None

    try:
        logger.info("Iniciando verificación de datos para entrenamiento de Isolation Forest Ventas...")

        df_raw = build_sales_features(db=db)
        total_ventas = len(df_raw)

        logger.info(f"Ventas completadas encontradas en el sistema: {total_ventas}")

        if total_ventas < min_sales:
            error_msg = (
                f"Entrenamiento cancelado por datos insuficientes: se encontraron {total_ventas} "
                f"ventas completadas, pero se requiere un mínimo de {min_sales} "
                f"para construir un modelo de Isolation Forest estadísticamente representativo."
            )
            logger.warning(error_msg)
            raise InsufficientDataError(error_msg)

        version = generate_isolation_forest_version()
        logger.info(f"Entrenando pipeline Isolation Forest versión '{version}' con {total_ventas} ventas...")

        fit_result = fit_isolation_forest_pipeline(
            df_features=df_raw,
            feature_columns=ISOLATION_FOREST_TRAINING_FEATURES,
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

        artifact_filename = f"isolation_forest_ventas_{version}.joblib"
        artifact_path = os.path.join(ML_MODELS_DIR, artifact_filename)

        artifact_payload = {
            "pipeline": fit_result["pipeline"],
            "input_features": ISOLATION_FOREST_TRAINING_FEATURES,
            "transformed_features": ISOLATION_FOREST_TRANSFORMED_FEATURES,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": total_ventas,
            "parameters": {
                "n_estimators": n_estimators,
                "contamination": contamination,
                "max_samples": ISOLATION_FOREST_MAX_SAMPLES,
                "random_state": random_state,
                "temporal_encoding": "cyclic_sin_cos",
                "imputer": "SimpleImputer(strategy='median')",
            },
            "metrics": {
                "distribucion_decision_function": fit_result["distribucion_decision_function"],
                "distribucion_anomaly_score": fit_result["distribucion_anomaly_score"],
            },
            "anomaly_score_quantiles": fit_result["distribucion_anomaly_score"].get("anomaly_score_quantiles", []),
            "resumen_modelo": {
                "total_ventas_entrenamiento": total_ventas,
                "features_entrada": len(ISOLATION_FOREST_TRAINING_FEATURES),
                "features_transformadas": len(ISOLATION_FOREST_TRANSFORMED_FEATURES),
                "score_anomalia_mediana": fit_result["distribucion_anomaly_score"].get("median"),
                "score_anomalia_p95": fit_result["distribucion_anomaly_score"].get("p95"),
            },
        }

        logger.info(f"Serializando artefacto Isolation Forest en '{artifact_path}'...")
        _save_artifact_atomically(artifact_payload, artifact_path)

        try:
            # Desactivar modelos anteriores exclusivamente de tipo ISOLATION_FOREST_VENTAS
            # KMEANS_CLIENTES permanece 100% intacto
            modelos_activos = (
                db.query(ModeloEntrenamiento)
                .filter(
                    ModeloEntrenamiento.tipo_modelo == ISOLATION_FOREST_MODEL_TYPE,
                    ModeloEntrenamiento.estado == "ACTIVO",
                )
                .with_for_update()
                .all()
            )
            for mod_antiguo in modelos_activos:
                mod_antiguo.estado = "OBSOLETO"

            nuevo_modelo = ModeloEntrenamiento(
                tipo_modelo=ISOLATION_FOREST_MODEL_TYPE,
                version=version,
                parametros=artifact_payload["parameters"],
                metricas=artifact_payload["metrics"],
                total_muestras=total_ventas,
                ruta_artefacto=artifact_path,
                estado="ACTIVO",
                resumen_modelo=artifact_payload["resumen_modelo"],
            )
            db.add(nuevo_modelo)
            db.commit()
            db.refresh(nuevo_modelo)
            logger.info(
                f"Modelo Isolation Forest versión '{version}' registrado exitosamente como ACTIVO (ID: {nuevo_modelo.id})."
            )

        except Exception as db_exc:
            db.rollback()
            logger.error(f"Fallo transaccional al persistir modelo Isolation Forest en BD: {db_exc}")
            _cleanup_orphan_artifact(artifact_path)
            raise TrainingError(f"Error al registrar metadatos de Isolation Forest en modelos_entrenamiento: {db_exc}") from db_exc

        return {
            "status": "ENTRENADO_Y_ACTIVO",
            "modelo_id": nuevo_modelo.id,
            "version": version,
            "total_muestras": total_ventas,
            "ruta_artefacto": artifact_path,
            "metricas": artifact_payload["metrics"],
            "resumen_modelo": artifact_payload["resumen_modelo"],
        }

    finally:
        _IF_TRAINING_LOCK.release()
