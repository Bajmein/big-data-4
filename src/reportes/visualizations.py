"""
[BI-01] Módulo de Visualizaciones Ejecutivas.

Este módulo genera gráficos pulidos y profesionales para
el informe ejecutivo.
"""

from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ExecutiveVisualizer:
    """Generador de visualizaciones ejecutivas."""

    def __init__(
        self,
        style: str = "whitegrid",
        palette: str = "viridis",
        figsize: tuple[int, int] = (14, 8),
        dpi: int = 300,
    ) -> None:
        """Inicializa el visualizador ejecutivo."""
        sns.set_style(style)
        sns.set_palette(palette)
        self.figsize = figsize
        self.dpi = dpi
        self.palette = palette

    def create_kpi_dashboard(
        self,
        summary_stats: dict[str, any],
        classifier_metrics: dict[str, float],
        clustering_metrics: dict[str, float],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Crea dashboard con KPIs principales."""
        fig = plt.figure(figsize=self.figsize)

        # Crear layout de 2x2
        gs = fig.add_gridspec(2, 2)

        # KPI 1: Tasa de Devolución
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(
            0.5,
            0.6,
            f"{summary_stats.get('return_rate', 0):.1%}",
            ha="center",
            va="center",
            fontsize=40,
            color="darkred",
        )
        ax1.text(
            0.5,
            0.3,
            "Tasa de Devolución Global",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax1.axis("off")

        # KPI 2: Total Transacciones
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(
            0.5,
            0.6,
            f"{summary_stats.get('total_transactions', 0):,}",
            ha="center",
            va="center",
            fontsize=40,
            color="navy",
        )
        ax2.text(
            0.5,
            0.3,
            "Total Transacciones Analizadas",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax2.axis("off")

        # KPI 3: Precisión del Modelo (F1)
        ax3 = fig.add_subplot(gs[1, 0])
        f1 = classifier_metrics.get("val_f1", 0)
        ax3.text(
            0.5, 0.6, f"{f1:.2f}", ha="center", va="center", fontsize=40, color="green"
        )
        ax3.text(
            0.5,
            0.3,
            "Model F1-Score (Predictivo)",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax3.axis("off")

        # KPI 4: Calidad de Segmentación
        ax4 = fig.add_subplot(gs[1, 1])
        sil = clustering_metrics.get("silhouette_score", 0)
        ax4.text(
            0.5,
            0.6,
            f"{sil:.2f}",
            ha="center",
            va="center",
            fontsize=40,
            color="purple",
        )
        ax4.text(
            0.5,
            0.3,
            "Silhouette Score (Segmentación)",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax4.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
        return fig

    def plot_return_rate_trends(
        self,
        temporal_analysis: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualiza tendencias de tasa de devolución."""
        fig, ax = plt.subplots(figsize=self.figsize)
        if not temporal_analysis.empty and "mes" in temporal_analysis.columns:
            sns.lineplot(
                data=temporal_analysis,
                x="mes",
                y="return_rate",
                marker="o",
                linewidth=3,
                ax=ax,
            )
            ax.set_title("Tendencia de Tasa de Devolución Mensual", fontsize=16)
            ax.set_ylabel("Tasa de Devolución")
            ax.set_xlabel("Mes")
            ax.grid(True, linestyle="--", alpha=0.7)

            # Area sombreada
            ax.fill_between(
                temporal_analysis["mes"], 0, temporal_analysis["return_rate"], alpha=0.1
            )

        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
        return fig

    def plot_manager_performance(
        self,
        manager_analysis: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualiza rendimiento de gerentes."""
        fig, ax = plt.subplots(figsize=self.figsize)
        if not manager_analysis.empty:
            sns.barplot(
                data=manager_analysis.sort_values("return_rate", ascending=False),
                y="gerente",
                x="return_rate",
                palette="RdYlGn_r",
                ax=ax,
            )
            ax.set_title("Desempeño por Gerente (Tasa de Devolución)", fontsize=16)

            # Añadir valores
            for i, v in enumerate(
                manager_analysis.sort_values("return_rate", ascending=False)[
                    "return_rate"
                ]
            ):
                ax.text(v, i, f" {v:.1%}", va="center")

        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
        return fig

    def plot_shipping_analysis(
        self,
        shipping_analysis: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualiza análisis de envíos."""
        fig, ax1 = plt.subplots(figsize=self.figsize)

        if not shipping_analysis.empty:
            # Barras para volumen
            sns.barplot(
                data=shipping_analysis,
                x="tipo_envio",
                y="total_orders",
                color="lightgray",
                alpha=0.6,
                ax=ax1,
            )
            ax1.set_ylabel("Total Órdenes")

            # Línea para tasa
            ax2 = ax1.twinx()
            sns.lineplot(
                data=shipping_analysis,
                x="tipo_envio",
                y="return_rate",
                marker="o",
                color="red",
                linewidth=3,
                ax=ax2,
            )
            ax2.set_ylabel("Tasa de Devolución", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            plt.title("Análisis de Envíos: Volumen vs Devoluciones", fontsize=16)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
        return fig

    def plot_customer_segments(
        self,
        cluster_profiles: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualiza segmentos de clientes."""
        fig, ax = plt.subplots(figsize=self.figsize)

        if not cluster_profiles.empty:
            # Bubble chart simulado con scatterplot
            # X=Frecuencia, Y=Cantidad, Size=N_Customers, Color=Tasa_Devolucion

            sns.scatterplot(
                data=cluster_profiles,
                x="avg_frecuencia",
                y="avg_cantidad",
                size="n_customers",
                hue="avg_tasa_devolucion",
                sizes=(200, 2000),
                palette="viridis",
                ax=ax,
            )

            # Etiquetas
            for i, row in cluster_profiles.iterrows():
                ax.text(
                    row["avg_frecuencia"],
                    row["avg_cantidad"],
                    f"{row['segment_name']}\n(n={row['n_customers']})",
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

            ax.set_title("Mapa de Segmentos de Clientes", fontsize=16)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
        return fig

    def plot_return_reasons_pareto(
        self,
        reasons_analysis: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Crea diagrama de Pareto de razones."""
        fig, ax1 = plt.subplots(figsize=self.figsize)

        if not reasons_analysis.empty:
            df_sorted = reasons_analysis.sort_values("count", ascending=False)
            df_sorted["cum_percentage"] = (
                df_sorted["count"].cumsum() / df_sorted["count"].sum()
            )

            # Barras
            sns.barplot(data=df_sorted, x="razon", y="count", color="steelblue", ax=ax1)
            ax1.set_ylabel("Frecuencia")
            ax1.tick_params(axis="x", rotation=45)

            # Línea acumulada
            ax2 = ax1.twinx()
            sns.lineplot(
                data=df_sorted,
                x="razon",
                y="cum_percentage",
                color="red",
                marker="o",
                ax=ax2,
            )
            ax2.set_ylabel("Porcentaje Acumulado", color="red")
            ax2.set_ylim(0, 1.1)

            # Línea 80%
            ax2.axhline(0.8, color="green", linestyle="--")

            plt.title("Diagrama de Pareto: Razones de Devolución", fontsize=16)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
        return fig
