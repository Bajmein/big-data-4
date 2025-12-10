"""
[DS-02] Módulo de Clustering para Segmentación de Clientes.

Este módulo implementa K-Means clustering para segmentar clientes.
Features para clustering: [Cantidad, Frecuencia, Tasa_Devolucion]
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib


class CustomerSegmentation:
    """Modelo K-Means para segmentación de clientes."""

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
        feature_columns: Optional[list[str]] = None,
    ) -> None:
        """Inicializa el modelo de segmentación."""
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.feature_columns = (
            feature_columns
            if feature_columns
            else ["cantidad", "frecuencia", "tasa_devolucion"]
        )
        self.is_fitted = False

    def prepare_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara features agregadas por cliente."""
        # Verificar columnas requeridas
        req_cols = ["tipo_cliente", "cantidad", "is_returned"]
        for col in req_cols:
            if col not in df.columns:
                # Intentar inferir tipo_cliente si es ID orden (asumiendo 1 cliente por orden para demo)
                if col == "tipo_cliente" and "id_orden" in df.columns:
                    print(
                        "Advertencia: 'tipo_cliente' no encontrado, usando 'id_orden' como proxy de cliente."
                    )
                    df = df.copy()
                    df["tipo_cliente"] = df["id_orden"]
                else:
                    raise KeyError(f"Falta columna requerida: {col}")

        # Agregación
        df_agg = (
            df.groupby("tipo_cliente")
            .agg(
                cantidad=("cantidad", "sum"),
                frecuencia=("cantidad", "count"),  # Número de transacciones
                tasa_devolucion=("is_returned", "mean"),
            )
            .reset_index()
        )

        return df_agg

    def find_optimal_k(
        self,
        df: pd.DataFrame,
        k_range: tuple[int, int] = (2, 10),
        plot: bool = True,
    ) -> dict[str, any]:
        """Encuentra K óptimo usando Elbow Method."""
        X = df[self.feature_columns].values
        n_samples = X.shape[0]

        if n_samples < 2:
            print("Advertencia: Insuficientes datos para clustering (n < 2).")
            return {"k_values": [], "inertias": [], "optimal_k": 1}

        # Ajustar rango de K segun muestras disponibles
        max_k = min(k_range[1], n_samples - 1)
        if max_k < k_range[0]:
            max_k = k_range[
                0
            ]  # Intentar al menos el minimo, aunque fallará si n_samples < k_min

        # Si aun asi n_samples < k_range[0], ajustar k_range[0]
        start_k = k_range[0]
        if n_samples <= start_k:
            start_k = 2
            max_k = min(n_samples - 1, 10)

        if max_k < start_k:
            print(
                f"Advertencia: n_samples={n_samples} insuficiente para K-Means con K>={start_k}"
            )
            return {"k_values": [], "inertias": [], "optimal_k": 1}

        X_scaled = self.scaler.fit_transform(X)  # Escalar temporalmente para la prueba

        inertias = []
        k_values = list(range(start_k, max_k + 1))

        for k in k_values:
            try:
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                km.fit(X_scaled)
                inertias.append(km.inertia_)
            except ValueError as e:
                print(f"Error fitting K={k}: {e}")
                break

        # Determinar codo simple
        optimal_k = min(4, len(k_values) + 1) if k_values else 1
        # Si tenemos suficientes valores, intentar ser más inteligente, si no default
        if len(k_values) >= 3:
            # Heuristica simple: 4 o max disponible
            optimal_k = min(4, k_values[-1])
        elif k_values:
            optimal_k = k_values[0]

        if plot and k_values:
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, inertias, "bo-")
            plt.xlabel("Número de Clusters (K)")
            plt.ylabel("Inercia (SSE)")
            plt.title("Método del Codo para K Óptimo")
            plt.grid(True)
            plt.show()

        return {"k_values": k_values, "inertias": inertias, "optimal_k": optimal_k}

    def fit(self, df: pd.DataFrame) -> "CustomerSegmentation":
        """Entrena el modelo de clustering."""
        X = df[self.feature_columns].values
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)
        self.is_fitted = True
        self.cluster_centers_ = self.scaler.inverse_transform(
            self.model.cluster_centers_
        )

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Asigna clientes a clusters."""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Entrena y predice."""
        self.fit(df)
        return self.predict(df)

    def get_cluster_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera perfiles descriptivos."""
        if "cluster" not in df.columns:
            raise ValueError("El DataFrame debe tener columna 'cluster'")

        profiles = (
            df.groupby("cluster")
            .agg(
                n_customers=("cluster", "count"),
                avg_cantidad=("cantidad", "mean"),
                avg_frecuencia=("frecuencia", "mean"),
                avg_tasa_devolucion=("tasa_devolucion", "mean"),
            )
            .reset_index()
        )

        # Asignar nombres
        def name_segment(row):
            if row["avg_tasa_devolucion"] > 0.5:
                return "Problemático"
            elif row["avg_cantidad"] > profiles["avg_cantidad"].mean():
                return "Premium"
            else:
                return "Estándar"

        profiles["segment_name"] = profiles.apply(name_segment, axis=1)
        return profiles

    def calculate_silhouette(self, df: pd.DataFrame) -> float:
        """Calcula Silhouette Score."""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)

        if len(np.unique(labels)) < 2:
            return 0.0

        return silhouette_score(X_scaled, labels)

    def save_model(self, filepath: str) -> None:
        """Guarda modelo."""
        joblib.dump(self, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "CustomerSegmentation":
        """Carga modelo."""
        return joblib.load(filepath)
