"""
[DS-01] Módulo de Clasificación para Predicción de Devoluciones.

Este módulo implementa un modelo Random Forest para predecir
la probabilidad de que una transacción resulte en devolución.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import joblib


class ReturnClassifier:
    """Clasificador Random Forest para predicción de devoluciones."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        feature_columns: Optional[list[str]] = None,
        target_column: str = "is_returned",
    ) -> None:
        """Inicializa el clasificador Random Forest."""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        # Features por defecto si no se especifican
        self.feature_columns = (
            feature_columns
            if feature_columns
            else ["cantidad", "prioridad", "tipo_envio", "mes", "trimestre"]
        )
        self.target_column = target_column
        self.is_fitted = False

    def prepare_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepara features y target."""
        # 🚨 PREVENCION DE DATA LEAKAGE 🚨
        # Aseguramos explícitamente que NO se usen columnas prohibidas
        forbidden_cols = {
            "razon",
            "estatus",
            "is_returned",
            "Razon",
            "Estatus",
            "id_orden",
            "fecha",
        }
        # Filtramos las columnas de features definidas
        safe_features = [
            col for col in self.feature_columns if col not in forbidden_cols
        ]

        if len(safe_features) != len(self.feature_columns):
            removed = set(self.feature_columns) - set(safe_features)
            print(f"⚠️ Alerta: Se eliminaron columnas prohibidas de features: {removed}")

        # Verificar features disponibles en el DF
        available_features = [col for col in safe_features if col in df.columns]

        if len(available_features) < len(safe_features):
            missing = set(safe_features) - set(available_features)
            print(
                f"Advertencia: Faltan features {missing}. Usando disponibles: {available_features}"
            )

        X = df[available_features].fillna(
            0
        )  # Simple imputación para features numéricas faltantes

        y = None
        if self.target_column in df.columns:
            y = df[self.target_column]

        return X, y

    def fit(self, df: pd.DataFrame, validation_split: float = 0.2) -> dict[str, float]:
        """Entrena el modelo."""
        X, y = self.prepare_features(df)

        if y is None:
            raise ValueError(
                f"La columna target '{self.target_column}' no existe en el DataFrame."
            )

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Calcular métricas básicas
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)

        # Para F1, Precision, Recall necesitamos predicciones
        from sklearn.metrics import f1_score, precision_score, recall_score

        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

        val_precision = precision_score(y_val, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, zero_division=0)

        return {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predice devoluciones."""
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado.")

        X, _ = self.prepare_features(df)
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predice probabilidades."""
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado.")

        X, _ = self.prepare_features(df)
        return self.model.predict_proba(X)

    def cross_validate(self, df: pd.DataFrame, cv: int = 5) -> dict[str, float]:
        """Realiza validación cruzada."""
        X, y = self.prepare_features(df)
        if y is None:
            raise ValueError("Target no encontrado para CV.")

        scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1")

        return {"cv_scores": scores, "cv_mean": scores.mean(), "cv_std": scores.std()}

    def feature_importance(self) -> pd.DataFrame:
        """Obtiene importancia de features."""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        available_features = [
            col for col in self.feature_columns
        ]  # Ajustar según lo usado en fit si cambió dinámicamente
        # Nota: Idealmente deberíamos guardar las columnas exactas usadas en fit
        if len(self.model.feature_importances_) != len(available_features):
            # Fallback si hubo discrepancia
            # Intentar usar feature_names_in_ si existe (sklearn > 1.0)
            if hasattr(self.model, "feature_names_in_"):
                available_features = self.model.feature_names_in_

        importance_df = pd.DataFrame(
            {
                "feature": available_features,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return importance_df

    def save_model(self, filepath: str) -> None:
        """Guarda modelo."""
        joblib.dump(self, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "ReturnClassifier":
        """Carga modelo."""
        return joblib.load(filepath)
