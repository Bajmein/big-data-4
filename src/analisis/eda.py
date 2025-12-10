"""
[DA-01, DA-02] Módulo de Análisis Exploratorio de Datos.

Este módulo proporciona funcionalidades para:
- [DA-01] Análisis de causa raíz de devoluciones
- [DA-02] Análisis de patrones temporales
"""

from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ExploratoryAnalysis:
    """Clase para realizar análisis exploratorio de datos de devoluciones."""

    def __init__(self, df: pd.DataFrame, figsize: tuple[int, int] = (12, 6)) -> None:
        """Inicializa el análisis exploratorio.

        Args:
            df: DataFrame transformado con columna 'is_returned'.
            figsize: Tamaño por defecto de las figuras.
        """
        self.df = df
        self.figsize = figsize
        if "is_returned" not in df.columns:
            # Intentar crearlo si existe estatus
            if "estatus" in df.columns:
                self.df["is_returned"] = (df["estatus"] == "Devuelto").astype(int)
            else:
                raise ValueError(
                    "El DataFrame debe contener la columna 'is_returned' o 'estatus'."
                )

    def summary_statistics(self) -> dict[str, any]:
        """Genera estadísticas resumen del dataset."""
        stats = {
            "total_transactions": len(self.df),
            "return_rate": self.df["is_returned"].mean(),
            "transactions_by_priority": self.df["prioridad"].value_counts().to_dict()
            if "prioridad" in self.df.columns
            else {},
            "transactions_by_shipping": self.df["tipo_envio"].value_counts().to_dict()
            if "tipo_envio" in self.df.columns
            else {},
        }
        return stats

    def analyze_by_manager(self, plot: bool = True) -> pd.DataFrame:
        """[DA-01] Analiza tasas de devolución por gerente."""
        if "gerente" not in self.df.columns:
            return pd.DataFrame()

        manager_stats = (
            self.df.groupby("gerente")
            .agg(
                total_orders=("is_returned", "count"),
                returns=("is_returned", "sum"),
                return_rate=("is_returned", "mean"),
            )
            .reset_index()
        )

        if plot:
            plt.figure(figsize=self.figsize)
            sns.barplot(
                data=manager_stats.sort_values("return_rate", ascending=False),
                x="return_rate",
                y="gerente",
                palette="viridis",
            )
            plt.title("Tasa de Devolución por Gerente")
            plt.xlabel("Tasa de Devolución")
            plt.tight_layout()

        return manager_stats

    def analyze_by_shipping(self, plot: bool = True) -> pd.DataFrame:
        """[DA-01] Analiza tasas de devolución por método de transporte."""
        if "tipo_envio" not in self.df.columns:
            return pd.DataFrame()

        shipping_stats = (
            self.df.groupby("tipo_envio")
            .agg(
                total_orders=("is_returned", "count"),
                returns=("is_returned", "sum"),
                return_rate=("is_returned", "mean"),
            )
            .reset_index()
        )

        if plot:
            plt.figure(figsize=self.figsize)
            sns.barplot(
                data=shipping_stats.sort_values("return_rate", ascending=False),
                x="return_rate",
                y="tipo_envio",
                palette="magma",
            )
            plt.title("Tasa de Devolución por Tipo de Envío")
            plt.tight_layout()

        return shipping_stats

    def analyze_by_priority(self, plot: bool = True) -> pd.DataFrame:
        """Analiza tasas de devolución por prioridad."""
        if "prioridad" not in self.df.columns:
            return pd.DataFrame()

        priority_stats = (
            self.df.groupby("prioridad")
            .agg(
                total_orders=("is_returned", "count"),
                returns=("is_returned", "sum"),
                return_rate=("is_returned", "mean"),
            )
            .reset_index()
        )

        if plot:
            plt.figure(figsize=self.figsize)
            sns.barplot(
                data=priority_stats, x="prioridad", y="return_rate", palette="Blues"
            )
            plt.title("Tasa de Devolución por Prioridad")
            plt.tight_layout()

        return priority_stats

    def analyze_temporal_patterns(self, plot: bool = True) -> pd.DataFrame:
        """[DA-02] Analiza patrones de devolución en el tiempo."""
        if "mes" not in self.df.columns:
            return pd.DataFrame()

        temporal_stats = (
            self.df.groupby("mes")
            .agg(
                return_rate=("is_returned", "mean"),
                total_orders=("is_returned", "count"),
            )
            .reset_index()
        )

        if plot:
            plt.figure(figsize=self.figsize)
            sns.lineplot(data=temporal_stats, x="mes", y="return_rate", marker="o")
            plt.title("Tendencia Mensual de Devoluciones")
            plt.ylabel("Tasa de Devolución")
            plt.xlabel("Mes")
            plt.grid(True)
            plt.tight_layout()

        return temporal_stats

    def analyze_return_reasons(self, plot: bool = True) -> pd.DataFrame:
        """Analiza las razones de devolución más frecuentes."""
        if "razon" not in self.df.columns:
            return pd.DataFrame()

        # Filtrar solo devoluciones con razón
        returns_df = self.df[self.df["is_returned"] == 1].copy()
        reason_counts = returns_df["razon"].value_counts().reset_index()
        reason_counts.columns = ["razon", "count"]
        reason_counts["percentage"] = (
            reason_counts["count"] / reason_counts["count"].sum()
        )

        if plot and not reason_counts.empty:
            plt.figure(figsize=self.figsize)
            sns.barplot(data=reason_counts, x="count", y="razon", palette="Reds_r")
            plt.title("Razones de Devolución Más Frecuentes")
            plt.tight_layout()

        return reason_counts

    def correlation_analysis(self, plot: bool = True) -> pd.DataFrame:
        """Calcula matriz de correlación."""
        # Seleccionar solo columnas numéricas
        numeric_df = self.df.select_dtypes(include=["number"])
        corr_matrix = numeric_df.corr()

        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Matriz de Correlación")
            plt.tight_layout()

        return corr_matrix

    def generate_full_report(
        self, output_dir: Optional[str] = None
    ) -> dict[str, pd.DataFrame]:
        """Genera reporte completo de EDA."""
        report = {
            "summary": pd.DataFrame([self.summary_statistics()]),
            "by_manager": self.analyze_by_manager(plot=False),
            "by_shipping": self.analyze_by_shipping(plot=False),
            "temporal": self.analyze_temporal_patterns(plot=False),
            "reasons": self.analyze_return_reasons(plot=False),
            "correlations": self.correlation_analysis(plot=False),
        }

        if output_dir:
            # Aquí se implementaría el guardado de gráficos si fuese necesario
            pass

        return report
