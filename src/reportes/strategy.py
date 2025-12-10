"""
[BI-02] Módulo de Estrategia y Arquitectura de Negocio.

Este módulo genera propuestas estratégicas basadas en los análisis.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Recommendation:
    """Estructura para una recomendación de negocio."""

    category: str
    title: str
    description: str
    priority: str
    impact: str
    effort: str
    kpi_target: Optional[str] = None


class BusinessStrategy:
    """Generador de estrategias de negocio basadas en datos."""

    def __init__(self) -> None:
        self.recommendations = []
        self.architecture_proposal = None

    def analyze_manager_insights(
        self, manager_analysis: pd.DataFrame
    ) -> list[Recommendation]:
        """Genera recomendaciones basadas en análisis de gerentes."""
        recs = []
        if not manager_analysis.empty:
            avg_return = manager_analysis["return_rate"].mean()
            # Identificar gerentes con alta tasa
            problem_managers = manager_analysis[
                manager_analysis["return_rate"] > avg_return
            ]

            if not problem_managers.empty:
                managers_list = ", ".join(problem_managers["gerente"].tolist())
                recs.append(
                    Recommendation(
                        category="Operacional",
                        title="Programa de Capacitación Focalizada",
                        description=f"Implementar capacitación para gerentes con tasas de devolución superiores al promedio: {managers_list}.",
                        priority="Alta",
                        impact="Alto (Reducción 5-10% devoluciones)",
                        effort="Medio",
                        kpi_target="Tasa de Devolución por Gerente < Promedio",
                    )
                )
        return recs

    def analyze_shipping_insights(
        self, shipping_analysis: pd.DataFrame
    ) -> list[Recommendation]:
        """Genera recomendaciones basadas en análisis de envíos."""
        recs = []
        if not shipping_analysis.empty:
            # Identificar peor método
            worst_shipping = shipping_analysis.sort_values(
                "return_rate", ascending=False
            ).iloc[0]
            recs.append(
                Recommendation(
                    category="Logística",
                    title=f"Auditoría de Proveedor de {worst_shipping['tipo_envio']}",
                    description=f"El método {worst_shipping['tipo_envio']} tiene la mayor tasa de devolución ({worst_shipping['return_rate']:.1%}). Realizar auditoría de calidad.",
                    priority="Media",
                    impact="Medio",
                    effort="Bajo",
                    kpi_target="Reducir tasa de devolución de envío en 20%",
                )
            )
        return recs

    def analyze_temporal_insights(
        self, temporal_analysis: pd.DataFrame
    ) -> list[Recommendation]:
        """Genera recomendaciones basadas en patrones temporales."""
        recs = []
        # Ejemplo genérico si hay datos
        if not temporal_analysis.empty:
            recs.append(
                Recommendation(
                    category="Planificación",
                    title="Ajuste de Stock Estacional",
                    description="Alinear inventario con picos de devolución identificados en el análisis temporal.",
                    priority="Baja",
                    impact="Bajo",
                    effort="Alto",
                    kpi_target="Optimización de Inventario",
                )
            )
        return recs

    def analyze_segment_insights(
        self, cluster_profiles: pd.DataFrame
    ) -> list[Recommendation]:
        """Genera recomendaciones basadas en segmentos."""
        recs = []
        if not cluster_profiles.empty and "segment_name" in cluster_profiles.columns:
            if "Problemático" in cluster_profiles["segment_name"].values:
                recs.append(
                    Recommendation(
                        category="Cliente",
                        title="Intervención Segmento Problemático",
                        description="Contactar proactivamente a clientes del segmento 'Problemático' para entender causas de insatisfacción.",
                        priority="Alta",
                        impact="Alto",
                        effort="Medio",
                        kpi_target="Churn Rate < 5%",
                    )
                )
            if "Premium" in cluster_profiles["segment_name"].values:
                recs.append(
                    Recommendation(
                        category="Cliente",
                        title="Programa VIP para Segmento Premium",
                        description="Crear incentivos de retención para el segmento de alto valor y baja devolución.",
                        priority="Media",
                        impact="Alto (Retención)",
                        effort="Bajo",
                        kpi_target="NPS > 70",
                    )
                )
        return recs

    def generate_bigdata_architecture(self) -> dict[str, any]:
        """Genera propuesta de arquitectura Big Data."""
        return {
            "components": [
                "Ingesta: Apache Kafka",
                "Procesamiento: Apache Spark Streaming",
                "Almacenamiento: Delta Lake",
                "ML Ops: MLflow",
                "Visualización: Apache Superset",
            ],
            "data_flow": "Transacciones -> Kafka -> Spark (Enriquecimiento + ML Predicción) -> Delta Lake -> Superset",
            "implementation_phases": [
                "Fase 1: Piloto de Ingesta (Kafka)",
                "Fase 2: Procesamiento Batch (Spark)",
                "Fase 3: Real-time Scoring",
                "Fase 4: Dashboarding Automatizado",
            ],
            "estimated_benefits": [
                "Detección de fraude en tiempo real",
                "Predicción dinámica de inventario",
                "Escalabilidad a millones de transacciones",
            ],
        }

    def analyze_all_insights(
        self,
        manager_analysis: pd.DataFrame,
        shipping_analysis: pd.DataFrame,
        temporal_analysis: pd.DataFrame,
        cluster_profiles: pd.DataFrame,
    ) -> None:
        """Ejecuta análisis completo."""
        self.recommendations.extend(self.analyze_manager_insights(manager_analysis))
        self.recommendations.extend(self.analyze_shipping_insights(shipping_analysis))
        self.recommendations.extend(self.analyze_temporal_insights(temporal_analysis))
        self.recommendations.extend(self.analyze_segment_insights(cluster_profiles))
        self.architecture_proposal = self.generate_bigdata_architecture()

    def generate_action_plan(
        self, prioritized_recs: list[Recommendation]
    ) -> pd.DataFrame:
        """Genera plan de acción."""
        plan = []
        for rec in prioritized_recs:
            plan.append(
                {
                    "Prioridad": rec.priority,
                    "Acción": rec.title,
                    "Descripción": rec.description,
                    "Impacto": rec.impact,
                    "KPI": rec.kpi_target,
                }
            )
        return pd.DataFrame(plan)

    def generate_strategy_report(
        self, output_path: Optional[str] = None
    ) -> dict[str, any]:
        """Genera reporte completo."""
        return {
            "executive_summary": "El análisis revela oportunidades significativas de optimización en la gestión de gerentes y logística. La segmentación de clientes permite acciones focalizadas.",
            "recommendations": self.recommendations,
            "architecture": self.architecture_proposal,
        }
