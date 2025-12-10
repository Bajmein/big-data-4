"""
Módulo de Modelado.

Contiene modelos de Machine Learning para:
- Clasificación: Predicción de probabilidad de devolución (Random Forest)
- Clustering: Segmentación de clientes (K-Means)
- Evaluación: Métricas y visualización de resultados
"""

from .classifier import ReturnClassifier
from .clustering import CustomerSegmentation
from .evaluation import ModelEvaluator

__all__ = ["ReturnClassifier", "CustomerSegmentation", "ModelEvaluator"]
