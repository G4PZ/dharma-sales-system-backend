"""
Motor Contextual de Detección de Operaciones y Alertas de Venta (Fase 5).

Arquitectura y Principios Operativos:
1. Isolation Forest como Detector Principal:
   - Determina el puntaje continuo de atipicidad (raw_anomaly_score = -decision_function).
   - Calcula el índice relativo (anomaly_index de 0.0 a 1.0) mediante interpolación
     lineal sobre la rejilla empírica de 101 cuantiles aprendida durante el entrenamiento.
   - anomaly_index representa la posición percentilar relativa respecto al histórico;
     NO representa probabilidad de fraude ni certeza delictiva.

2. K-Means como Perfil Contextual Opcional:
   - Enriquece la interpretación comercial del cliente cuando este cuenta con historial suficiente
     (mínimo MIN_PURCHASES_FOR_KMEANS_PROFILE = 2 compras previas).
   - K-Means NO modifica ni altera arbitrariamente el puntaje de anomalía de la venta.
   - Si no hay modelo K-Means activo o el cliente es nuevo, la venta se analiza normalmente
     contra la distribución global (modelo_kmeans_id = NULL).

3. Anti-Data-Leakage Estricto:
   - Se utiliza build_sale_features_for_sale y build_customer_features_before_sale para
     garantizar que ninguna venta actual ni posterior contamine el perfil contextual.

4. Umbrales Operativos Configurables:
   - NORMAL: anomaly_index < 0.90 (no genera alerta persistente).
   - BAJO:   0.90 <= anomaly_index < 0.95 (alerta persistente).
   - MEDIO:  0.95 <= anomaly_index < 0.99 (alerta persistente).
   - ALTO:   anomaly_index >= 0.99 (alerta persistente).

5. Explicabilidad Determinística:
   - Desglose de factores contextuales observados (factores_contribuyentes), nunca feature_importance.
   - Vocabulario técnico y neutral: "operación inusual", "patrón atípico", "desviación respecto al histórico".
   - Prohibido el uso de "fraude confirmado", "delito" o "probabilidad de fraude".

6. Idempotencia y Carga Segura:
   - Carga restringida estrictamente a ML_MODELS_DIR con validación de ruta canónica.
   - Caché en memoria invalidable por versión y timestamp del artefacto.
   - Idempotencia transaccional ante re-análisis con los mismos modelos.
"""

from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import ML_MODELS_DIR
from app.models.ai import AlertaVenta, ModeloEntrenamiento
from app.models.sale import Venta
from app.services.ai.feature_engineering import (
    KMEANS_FEATURE_COLUMNS,
    build_customer_features_before_sale,
    build_sale_features_for_sale,
)
from app.services.ai.trainer import (
    ISOLATION_FOREST_MODEL_TYPE,
    ISOLATION_FOREST_TRAINING_FEATURES,
    KMEANS_MODEL_TYPE,
    KMEANS_TRAINING_FEATURES,
    MIN_PURCHASES_FOR_KMEANS_PROFILE,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES Y UMBRALES OPERATIVOS CONFIGURABLES
# =============================================================================

# Umbrales sobre el índice relativo (anomaly_index: 0.00 a 1.00)
# Representan posiciones percentilares operativas en la distribución aprendida.
THRESHOLD_NORMAL: float = 0.90
THRESHOLD_BAJO: float = 0.90
THRESHOLD_MEDIO: float = 0.95
THRESHOLD_ALTO: float = 0.99

LEVEL_NORMAL: str = "NORMAL"
LEVEL_BAJO: str = "BAJO"
LEVEL_MEDIO: str = "MEDIO"
LEVEL_ALTO: str = "ALTO"

VALID_ALERT_LEVELS = {LEVEL_BAJO, LEVEL_MEDIO, LEVEL_ALTO}


# =============================================================================
# EXCEPCIONES ESPECÍFICAS DEL DETECTOR
# =============================================================================

class ModelNotAvailableError(Exception):
    """Lanzada cuando no existe un modelo activo requerido en la base de datos o en disco."""
    pass


class DetectionError(Exception):
    """Lanzada ante incongruencias de datos o fallos en el proceso de inferencia."""
    pass


# =============================================================================
# CARGA SEGURA Y CACHÉ DE ARTEFACTOS EN MEMORIA
# =============================================================================

# Estructura de caché: {tipo_modelo: {"id": int, "version": str, "mtime": float, "artifact": dict}}
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_model_cache() -> None:
    """Limpia la memoria caché de artefactos ML (útil para pruebas y recargas)."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()


def _validate_safe_model_path(file_path: str) -> str:
    """
    Valida de forma estricta que la ruta del artefacto resida dentro de ML_MODELS_DIR
    mediante resolución de rutas canónicas, previniendo Directory Traversal.
    """
    real_models_dir = os.path.realpath(ML_MODELS_DIR)
    real_file_path = os.path.realpath(file_path)

    try:
        common = os.path.commonpath([real_models_dir, real_file_path])
    except ValueError:
        raise ValueError(f"Acceso denegado: la ruta '{file_path}' no pertenece a la misma raíz que ML_MODELS_DIR.")

    if common != real_models_dir:
        raise ValueError(f"Acceso denegado: el archivo '{file_path}' reside fuera de ML_MODELS_DIR.")

    if not os.path.isfile(real_file_path):
        raise FileNotFoundError(f"El artefacto del modelo no existe en la ruta física: '{file_path}'")

    return real_file_path


def load_active_model_cached(
    db: Session,
    model_type: str,
) -> Optional[Tuple[ModeloEntrenamiento, Dict[str, Any]]]:
    """
    Consulta en la base de datos el modelo ACTIVO más reciente del tipo solicitado,
    verifica su archivo físico y lo carga a través de la memoria caché.

    Si el modelo cambió de versión, ID o fecha de modificación en disco, la caché se invalida
    automáticamente y se recarga el nuevo artefacto.
    """
    active_record = (
        db.query(ModeloEntrenamiento)
        .filter(
            ModeloEntrenamiento.tipo_modelo == model_type,
            ModeloEntrenamiento.estado == "ACTIVO",
        )
        .order_by(ModeloEntrenamiento.id.desc())
        .first()
    )

    if not active_record:
        return None

    safe_path = _validate_safe_model_path(active_record.ruta_artefacto)
    current_mtime = os.path.getmtime(safe_path)

    cached_entry = _MODEL_CACHE.get(model_type)
    if (
        cached_entry
        and cached_entry.get("id") == active_record.id
        and cached_entry.get("version") == active_record.version
        and cached_entry.get("mtime") == current_mtime
    ):
        return active_record, cached_entry["artifact"]

    # Carga segura y almacenamiento en caché
    try:
        artifact = joblib.load(safe_path)
    except Exception as exc:
        logger.error(f"Error al deserializar artefacto .joblib '{safe_path}': {exc}")
        raise DetectionError(f"No fue posible cargar el artefacto del modelo {model_type}: {exc}") from exc

    _MODEL_CACHE[model_type] = {
        "id": active_record.id,
        "version": active_record.version,
        "mtime": current_mtime,
        "artifact": artifact,
    }

    return active_record, artifact


# =============================================================================
# CÁLCULO DEL ÍNDICE DE ANOMALÍA Y UMBRALES
# =============================================================================

def calculate_anomaly_index(
    raw_anomaly_score: float,
    quantiles_reference: List[float],
) -> float:
    """
    Calcula el índice relativo interpretable (anomaly_index) en rango [0.0, 1.0]
    mediante interpolación lineal sobre la rejilla empírica de 101 cuantiles (0% a 100%).

    Significado Metodológico:
    - anomaly_index = 0.97 indica que el score de atipicidad de esta venta se ubica
      aproximadamente en el percentil 97 de la distribución histórica con la cual se entrenó el modelo.
    - NO representa 97% de probabilidad de fraude ni certeza delictiva.

    Manejo de Extremos:
    - Scores <= cuantil 0% (mínimo histórico) -> 0.0000.
    - Scores >= cuantil 100% (máximo histórico) -> 1.0000.
    """
    if not quantiles_reference or len(quantiles_reference) < 2:
        # Fallback defensivo ante rejilla no disponible
        return 0.5

    q_arr = np.array(quantiles_reference, dtype=float)

    # Fronteras absolutas
    if raw_anomaly_score <= q_arr[0]:
        return 0.0
    if raw_anomaly_score >= q_arr[-1]:
        return 1.0

    pcts = np.linspace(0.0, 1.0, len(q_arr))
    interpolated = float(np.interp(raw_anomaly_score, q_arr, pcts))
    return round(float(np.clip(interpolated, 0.0, 1.0)), 4)


def determine_alert_level(anomaly_index: float) -> str:
    """
    Determina el nivel operativo de la alerta a partir del índice relativo.
    NORMAL: < 0.90
    BAJO:   0.90 <= index < 0.95
    MEDIO:  0.95 <= index < 0.99
    ALTO:   index >= 0.99
    """
    if anomaly_index >= THRESHOLD_ALTO:
        return LEVEL_ALTO
    elif anomaly_index >= THRESHOLD_MEDIO:
        return LEVEL_MEDIO
    elif anomaly_index >= THRESHOLD_BAJO:
        return LEVEL_BAJO
    else:
        return LEVEL_NORMAL


# =============================================================================
# EXPLICABILIDAD CONTEXTUAL DETERMINÍSTICA
# =============================================================================

def generate_contextual_factors(
    df_sale_features: pd.DataFrame,
    anomaly_index: float,
    raw_anomaly_score: float,
    customer_has_kmeans_profile: bool,
    kmeans_profile_name: Optional[str] = None,
    kmeans_centroid_summary: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Construye una lista estructurada de factores contextuales y observados en los datos
    que explican de forma verificable por qué la operación se desvía del patrón habitual.

    Reglas de Transparencia Algorítmica:
    - Se denominan 'factores_contribuyentes' y NO 'feature_importance' (ya que Isolation
      Forest no genera pesos paramétricos por variable en inferencia individual).
    - Para clientes sin historial, se contextualiza explícitamente la evaluación contra la
      distribución global sin estigmatizar el estado de cliente nuevo.
    - Se seleccionan y priorizan entre 2 y 5 factores más destacados.
    """
    factors: List[Dict[str, Any]] = []
    row = df_sale_features.iloc[0]

    monto_total = float(row.get("monto_total", 0.0))
    unidades_totales = int(row.get("unidades_totales", 0))
    compras_previas = int(row.get("compras_previas", 0))
    monto_promedio_previo = row.get("monto_promedio_previo")
    unidades_promedio_previas = row.get("unidades_promedio_previas")
    concentracion_importe = float(row.get("concentracion_importe", 0.0))
    hora_operacion = int(row.get("hora_operacion", 12))
    ratio_monto = row.get("ratio_monto_vs_promedio_cliente")
    ratio_unidades = row.get("ratio_unidades_vs_promedio_cliente")

    # 1. Factor de desviación de monto vs historial individual (si existe historial previo)
    if compras_previas > 0 and pd.notna(monto_promedio_previo) and float(monto_promedio_previo) > 0:
        avg_monto = float(monto_promedio_previo)
        r_monto = float(ratio_monto) if pd.notna(ratio_monto) else (monto_total / avg_monto)
        if r_monto >= 2.0:
            factors.append({
                "factor": "monto_vs_historial",
                "valor_actual": round(monto_total, 2),
                "valor_referencia": round(avg_monto, 2),
                "ratio": round(r_monto, 2),
                "descripcion": (
                    f"El monto actual (S/ {monto_total:,.2f}) es aproximadamente {r_monto:.1f} veces "
                    f"el ticket promedio histórico del cliente (S/ {avg_monto:,.2f})."
                ),
            })
        elif r_monto <= 0.15:
            factors.append({
                "factor": "monto_vs_historial_bajo",
                "valor_actual": round(monto_total, 2),
                "valor_referencia": round(avg_monto, 2),
                "ratio": round(r_monto, 2),
                "descripcion": (
                    f"El monto actual (S/ {monto_total:,.2f}) es inusualmente bajo respecto "
                    f"al ticket promedio histórico del cliente (S/ {avg_monto:,.2f})."
                ),
            })

    # 2. Factor de desviación de volumen físico (unidades)
    if compras_previas > 0 and pd.notna(unidades_promedio_previas) and float(unidades_promedio_previas) > 0:
        avg_units = float(unidades_promedio_previas)
        r_units = float(ratio_unidades) if pd.notna(ratio_unidades) else (unidades_totales / avg_units)
        if r_units >= 2.5:
            factors.append({
                "factor": "unidades_vs_historial",
                "valor_actual": unidades_totales,
                "valor_referencia": round(avg_units, 1),
                "ratio": round(r_units, 2),
                "descripcion": (
                    f"El volumen de {unidades_totales} unidades supera en {r_units:.1f} veces "
                    f"el promedio habitual del cliente ({avg_units:.1f} unidades)."
                ),
            })

    # 3. Factor de concentración de catálogo / producto dominante
    if concentracion_importe >= 0.80:
        pct = round(concentracion_importe * 100.0, 1)
        factors.append({
            "factor": "concentracion_importe",
            "valor_actual": round(concentracion_importe, 3),
            "descripcion": f"El {pct}% del importe total se concentra en una sola línea de producto.",
        })

    # 4. Factor de horario atípico
    if hora_operacion < 7 or hora_operacion >= 22:
        factors.append({
            "factor": "horario_operacion",
            "valor_actual": f"{hora_operacion:02d}:00 hrs",
            "descripcion": f"Operación registrada fuera del horario comercial habitual ({hora_operacion:02d}:00 hrs).",
        })

    # 5. Factor de cliente sin historial (Cold Start)
    if compras_previas < MIN_PURCHASES_FOR_KMEANS_PROFILE:
        factors.append({
            "factor": "cliente_sin_historial",
            "valor_actual": compras_previas,
            "descripcion": (
                "Cliente sin historial individual suficiente; la operación fue evaluada "
                "principalmente contra el comportamiento global aprendido."
            ),
        })

    # 6. Factor contextual K-Means (si el cliente tiene perfil asignado)
    if customer_has_kmeans_profile and kmeans_profile_name:
        desc_perfil = f"Cliente contextualizado comercialmente en '{kmeans_profile_name}'."
        if kmeans_centroid_summary and "ticket_promedio" in kmeans_centroid_summary:
            desc_perfil += f" Ticket de referencia del perfil: S/ {kmeans_centroid_summary['ticket_promedio']:,.2f}."
        factors.append({
            "factor": "perfil_contextual_cliente",
            "valor_actual": kmeans_profile_name,
            "descripcion": desc_perfil,
        })

    # 7. Resumen de posición en la distribución del modelo
    factors.append({
        "factor": "posicion_distribucion_global",
        "valor_actual": anomaly_index,
        "raw_score": round(raw_anomaly_score, 5),
        "descripcion": (
            f"El score de atipicidad se ubica en el percentil {anomaly_index * 100:.1f} "
            f"de la distribución histórica de operaciones aprendida por el modelo."
        ),
    })

    # Limitar a máximo 5 factores para claridad ejecutiva
    return factors[:5]


def build_alert_description(
    level: str,
    factors: List[Dict[str, Any]],
    has_kmeans_profile: bool,
    kmeans_profile_name: Optional[str] = None,
) -> str:
    """
    Construye una descripción textual neutral, determinística y basada en datos.

    Prohibición Estricta:
    - NO utiliza palabras como: 'fraude', 'delito', 'fraude confirmado'.
    - Emplea: 'operación inusual', 'patrón atípico', 'desviación respecto al histórico'.
    """
    # Identificar factores destacados
    has_monto = any(f["factor"] == "monto_vs_historial" for f in factors)
    has_units = any(f["factor"] == "unidades_vs_historial" for f in factors)
    has_cold_start = any(f["factor"] == "cliente_sin_historial" for f in factors)
    has_conc = any(f["factor"] == "concentracion_importe" for f in factors)

    if has_cold_start:
        base = (
            "Operación con patrón atípico respecto a la distribución global de ventas. "
            "El cliente no cuenta con compras previas suficientes para establecer un perfil individual."
        )
        if has_conc:
            base += " Se observa una elevada concentración del importe en una sola línea de producto."
        return base

    if has_monto and has_units:
        base = (
            "Operación con comportamiento significativamente inusual respecto a la distribución histórica. "
            "Tanto el importe total como el volumen de unidades superan ampliamente el comportamiento habitual del cliente."
        )
    elif has_monto:
        base = (
            "Operación con desviación marcada respecto al historial del cliente. "
            "El monto de la venta supera notablemente el ticket promedio habitual registrado."
        )
    elif has_units:
        base = (
            "Operación con volumen de unidades inusual respecto al historial de compras del cliente."
        )
    elif has_conc:
        base = (
            "Operación con atipicidad estructural debido a una concentración muy elevada del importe total en un solo producto."
        )
    else:
        base = (
            "Operación con desviación estadística respecto a la distribución histórica aprendida por el modelo de ventas."
        )

    if has_kmeans_profile and kmeans_profile_name:
        base += f" Evaluado bajo el contexto comercial '{kmeans_profile_name}'."

    return base


# =============================================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS CONTEXTUAL
# =============================================================================

def analyze_sale(
    db: Session,
    venta_id: int,
    persist_alert: bool = True,
) -> Dict[str, Any]:
    """
    Ejecuta el análisis contextual completo de una operación de venta combinando:
    1. Isolation Forest (obligatorio para análisis): calcula raw_anomaly_score y anomaly_index.
    2. K-Means (opcional contextual): clasifica al cliente si posee >= 2 compras previas.
    3. Historial individual: integrado en las features contextuales sin data leakage.
    4. Generación determinística de factores observados y descripción legible.
    5. Persistencia idempotente de AlertaVenta si nivel != NORMAL y persist_alert=True.

    Retorna un diccionario estructurado con los resultados del análisis.
    """
    # 1. Validar existencia de la venta
    venta = db.query(Venta).filter(Venta.id == venta_id).first()
    if not venta:
        raise ValueError(f"No se encontró la venta con ID {venta_id}")

    # 2. Obtener modelo Isolation Forest activo (obligatorio)
    if_result = load_active_model_cached(db=db, model_type=ISOLATION_FOREST_MODEL_TYPE)
    if not if_result:
        raise ModelNotAvailableError(
            "El análisis IA aún no está disponible porque no existe un modelo Isolation Forest entrenado y activo."
        )
    if_model_record, if_artifact = if_result
    if_pipeline = if_artifact["pipeline"]

    # Extraer referencia empírica de cuantiles (preferencia: anomaly_score_quantiles, fallback cuantiles_101)
    quantiles_reference = (
        if_artifact.get("anomaly_score_quantiles")
        or if_artifact.get("metrics", {}).get("distribucion_anomaly_score", {}).get("anomaly_score_quantiles")
        or if_artifact.get("metrics", {}).get("distribucion_anomaly_score", {}).get("cuantiles_101")
        or []
    )

    # 3. Construir características de la venta analizada (exactamente las 20 features del modelo)
    df_sale_features = build_sale_features_for_sale(db=db, venta_id=venta.id)
    X_sale = df_sale_features[ISOLATION_FOREST_TRAINING_FEATURES]

    # 4. Inferencia con Isolation Forest
    # decision_function: negativo = atípico; positivo = típico
    raw_decision = float(if_pipeline.decision_function(X_sale)[0])
    raw_anomaly = -raw_decision

    # 5. Calcular índice relativo interpretable (0.0 a 1.0)
    anomaly_index = calculate_anomaly_index(
        raw_anomaly_score=raw_anomaly,
        quantiles_reference=quantiles_reference,
    )

    # 6. Determinar nivel operativo de alerta
    nivel_alerta = determine_alert_level(anomaly_index)

    # 7. Contextualización K-Means (opcional)
    kmeans_result = load_active_model_cached(db=db, model_type=KMEANS_MODEL_TYPE)
    modelo_kmeans_id: Optional[int] = None
    kmeans_version: Optional[str] = None
    cluster_perfil_id: Optional[int] = None
    perfil_nombre: Optional[str] = None
    kmeans_centroid_info: Optional[Dict[str, Any]] = None

    if kmeans_result:
        km_record, km_artifact = kmeans_result
        # Evaluar si el cliente tiene compras previas suficientes (mínimo 2)
        try:
            df_customer_before = build_customer_features_before_sale(db=db, venta_id=venta.id)
            total_compras_previas = int(df_customer_before["total_compras"].iloc[0])
        except Exception as exc:
            logger.warning(f"No fue posible extraer historial previo del cliente para K-Means: {exc}")
            total_compras_previas = 0

        if total_compras_previas >= MIN_PURCHASES_FOR_KMEANS_PROFILE:
            km_pipeline = km_artifact["pipeline"]
            km_feature_cols = km_artifact.get("feature_columns") or KMEANS_TRAINING_FEATURES
            X_cust = df_customer_before[km_feature_cols]
            try:
                cluster_pred = int(km_pipeline.predict(X_cust)[0])
                modelo_kmeans_id = km_record.id
                kmeans_version = km_record.version
                cluster_perfil_id = cluster_pred
                perfil_nombre = f"Perfil {cluster_pred}"

                # Extraer información de centroides si está disponible en resumen
                resumen = km_artifact.get("resumen_modelo", {})
                perfiles_info = resumen.get("perfiles", {})
                if str(cluster_pred) in perfiles_info:
                    kmeans_centroid_info = perfiles_info[str(cluster_pred)]
            except Exception as km_exc:
                logger.warning(f"Fallo al inferir cluster K-Means para venta {venta_id}: {km_exc}")
                modelo_kmeans_id = None
                cluster_perfil_id = None
                perfil_nombre = None
        else:
            # Cliente con < 2 compras previas: no se aplica K-Means (NULL)
            modelo_kmeans_id = None
            cluster_perfil_id = None
            perfil_nombre = None

    # 8. Generar factores explicativos contextuales y descripción legible
    has_kmeans = cluster_perfil_id is not None
    factores = generate_contextual_factors(
        df_sale_features=df_sale_features,
        anomaly_index=anomaly_index,
        raw_anomaly_score=raw_anomaly,
        customer_has_kmeans_profile=has_kmeans,
        kmeans_profile_name=perfil_nombre,
        kmeans_centroid_summary=kmeans_centroid_info,
    )
    descripcion = build_alert_description(
        level=nivel_alerta,
        factors=factores,
        has_kmeans_profile=has_kmeans,
        kmeans_profile_name=perfil_nombre,
    )

    # 9. Persistencia de Alerta e Idempotencia
    alerta_id: Optional[int] = None
    alerta_generada = False
    ya_analizada = False

    if nivel_alerta in VALID_ALERT_LEVELS and persist_alert:
        # Verificar si ya existe una alerta previa para esta venta y esta combinación de versiones de modelos
        query_existente = db.query(AlertaVenta).filter(
            AlertaVenta.venta_id == venta.id,
            AlertaVenta.modelo_isolation_forest_id == if_model_record.id,
        )
        if modelo_kmeans_id is not None:
            query_existente = query_existente.filter(AlertaVenta.modelo_kmeans_id == modelo_kmeans_id)
        else:
            query_existente = query_existente.filter(AlertaVenta.modelo_kmeans_id.is_(None))

        alerta_previa = query_existente.first()

        if alerta_previa:
            alerta_id = alerta_previa.id
            alerta_generada = True
            ya_analizada = True
            logger.info(
                f"La venta {venta.id} ya contaba con alerta previa ID {alerta_id} para los mismos modelos. Retornando existente."
            )
        else:
            nueva_alerta = AlertaVenta(
                venta_id=venta.id,
                cliente_id=venta.cliente_id,
                modelo_isolation_forest_id=if_model_record.id,
                modelo_kmeans_id=modelo_kmeans_id,
                puntaje_anomalia=anomaly_index,
                nivel_alerta=nivel_alerta,
                cluster_perfil_id=cluster_perfil_id,
                perfil_nombre=perfil_nombre,
                descripcion=descripcion,
                factores_contribuyentes=factores,
                estado_revision="PENDIENTE",
            )
            db.add(nueva_alerta)
            try:
                db.commit()
                db.refresh(nueva_alerta)
                alerta_id = nueva_alerta.id
                alerta_generada = True
                logger.info(
                    f"AlertaVenta ID {alerta_id} creada exitosamente para venta {venta.id} con nivel {nivel_alerta}."
                )
            except Exception as db_exc:
                db.rollback()
                logger.error(f"Error al persistir AlertaVenta para venta {venta.id}: {db_exc}")
                raise DetectionError(f"Error al registrar la alerta de venta en base de datos: {db_exc}") from db_exc

    return {
        "venta_id": venta.id,
        "cliente_id": venta.cliente_id,
        "nivel_alerta": nivel_alerta,
        "puntaje_anomalia": anomaly_index,
        "raw_anomaly_score": round(raw_anomaly, 5),
        "alerta_generada": alerta_generada,
        "alerta_id": alerta_id,
        "ya_analizada": ya_analizada,
        "modelo_isolation_forest_id": if_model_record.id,
        "modelo_isolation_forest_version": if_model_record.version,
        "modelo_kmeans_id": modelo_kmeans_id,
        "modelo_kmeans_version": kmeans_version,
        "cluster_perfil_id": cluster_perfil_id,
        "perfil_nombre": perfil_nombre,
        "descripcion": descripcion,
        "factores_contribuyentes": factores,
    }
