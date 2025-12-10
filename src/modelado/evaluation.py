"""
[DS-03] Módulo de Evaluación de Modelos.

Este módulo proporciona funcionalidades para evaluar y visualizar
el rendimiento de los modelos.
"""

from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)


class ModelEvaluator:
    """Clase para evaluar modelos de clasificación y clustering."""

    def __init__(
        self, figsize: tuple[int, int] = (10, 8), style: str = "whitegrid"
    ) -> None:
        """Inicializa el evaluador."""
        self.figsize = figsize
        sns.set_style(style)

    def evaluate_classifier(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Calcula métricas completas de clasificación."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics["roc_auc"] = 0.0

        return metrics

    def classification_report_df(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> pd.DataFrame:
        """Genera reporte de clasificación como DataFrame."""
        report = classification_report(y_true, y_pred, output_dict=True)
        return pd.DataFrame(report).transpose()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[list[str]] = None,
        normalize: bool = False,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Genera visualización de la matriz de confusión."""
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        if labels is None:
            labels = ["No Devuelto", "Devuelto"]

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path)
        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Genera curva ROC."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        fig = plt.figure(figsize=self.figsize)
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        return fig

    def plot_feature_importance(
        self,
        feature_names: list[str],
        importances: np.ndarray,
        top_n: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualiza importancia de features."""
        indices = np.argsort(importances)[::-1][:top_n]

        fig = plt.figure(figsize=self.figsize)
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path)
        return fig

    def plot_clusters_2d(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: tuple[str, str] = ("Feature 1", "Feature 2"),
        centers: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualiza clusters en 2D."""
        fig = plt.figure(figsize=self.figsize)
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", marker="o", alpha=0.5)

        if centers is not None:
            plt.scatter(
                centers[:, 0], centers[:, 1], c="red", s=200, alpha=0.9, marker="X"
            )

        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title("Cluster Visualization")

        if save_path:
            plt.savefig(save_path)
        return fig

    def plot_cluster_profiles(
        self,
        profiles_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualiza perfiles de clusters."""
        # Normalizar para visualización (radar chart sería ideal, pero barplot es más robusto)
        # Aquí haremos un barplot de las características promedio

        cols_to_plot = [c for c in profiles_df.columns if c.startswith("avg_")]
        df_melted = profiles_df.melt(
            id_vars=["cluster", "segment_name"], value_vars=cols_to_plot
        )

        fig = plt.figure(figsize=self.figsize)
        sns.barplot(data=df_melted, x="variable", y="value", hue="segment_name")
        plt.title("Perfiles de Segmentos")
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path)
        return fig
