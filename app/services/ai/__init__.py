"""
Módulo de Inteligencia Artificial de Dharma Sales System.

Proporciona capacidades analíticas mediante Machine Learning utilizando scikit-learn:
1. K-Means: Descubrimiento no supervisado de perfiles de clientes basados en su historial comercial.
2. Isolation Forest: Detección de operaciones de venta inusuales respecto al patrón global e individual.

Ambos modelos son independientes y se coordinan para emitir alertas explicables sin interferir
en el flujo transaccional de ventas ni en la facturación electrónica SUNAT.
"""

from typing import List

__all__: List[str] = []
