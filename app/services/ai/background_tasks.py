"""
Módulo de Tareas en Segundo Plano (BackgroundTasks) para el Motor IA (Fase 6).

Responsabilidades y Principios de Diseño:
1. Desacoplamiento Absoluto de la Transacción Comercial:
   - Se ejecuta únicamente después de que la venta haya completado su transacción atómica y commit en PostgreSQL.
   - Es incapaz de revertir o afectar una venta confirmada.

2. Independencia de Sesión de Base de Datos:
   - NUNCA reutiliza la sesión SQLAlchemy del request HTTP original (que ya puede estar cerrada al ejecutarse la tarea).
   - Abre y administra su propia instancia limpia mediante SessionLocal(), garantizando su cierre en el bloque finally.

3. Tolerancia a Fallos y Resiliencia Operativa:
   - ModelNotAvailableError: No se considera error crítico; se registra a nivel INFO (comportamiento normal sin modelos activos).
   - Excepciones Inesperadas: Se capturan con logger.exception, se ejecuta rollback exclusivo sobre la sesión de fondo
     y se absorbe el error para proteger el servicio principal.

4. Idempotencia y Eficiencia:
   - Invoca analyze_sale(db, venta_id, persist_alert=True), delegando la detección, contextualización y prevención
     de alertas duplicadas al motor contextual de la Fase 5.
"""

import logging
from typing import Any, Dict, Optional

from app.database import SessionLocal
from app.services.ai import detector
from app.services.ai.detector import ModelNotAvailableError

logger = logging.getLogger(__name__)


def process_sale_ai_background(venta_id: int) -> Optional[Dict[str, Any]]:
    """
    Wrapper independiente y no bloqueante para el análisis IA de una venta en segundo plano.

    Parámetros:
    - venta_id: ID entero de la venta confirmada en base de datos.

    Retorna:
    - Diccionario con los resultados del análisis si se completó, o None si fue omitido o falló.
    """
    logger.info(f"Iniciando análisis IA automático para venta_id={venta_id}")
    db = SessionLocal()
    try:
        resultado = detector.analyze_sale(
            db=db,
            venta_id=venta_id,
            persist_alert=True,
        )

        nivel = resultado.get("nivel_alerta", "DESCONOCIDO")
        alerta_id = resultado.get("alerta_id")
        ya_analizada = resultado.get("ya_analizada", False)

        if nivel == "NORMAL":
            logger.info(f"Análisis IA completado para venta_id={venta_id}: NORMAL (sin alerta)")
        else:
            if ya_analizada:
                logger.info(
                    f"Análisis IA completado para venta_id={venta_id}: {nivel} "
                    f"(alerta previa existente ID {alerta_id})"
                )
            else:
                logger.info(
                    f"Alerta IA generada para venta_id={venta_id}: {nivel} "
                    f"(nueva Alerta ID {alerta_id})"
                )

        return resultado

    except ModelNotAvailableError:
        # Estado operativo esperado cuando todavía no existe un modelo Isolation Forest activo
        logger.info(
            f"Análisis IA omitido para venta_id={venta_id}: no existe modelo Isolation Forest activo."
        )
        return None

    except Exception as exc:
        # Captura de cualquier fallo inesperado para garantizar que la venta nunca se vea afectada
        logger.exception(
            f"Error inesperado durante el análisis IA automático de la venta_id={venta_id}: {exc}"
        )
        try:
            db.rollback()
        except Exception:
            pass
        return None

    finally:
        db.close()
        logger.debug(f"Sesión de base de datos cerrada para análisis IA de venta_id={venta_id}")
