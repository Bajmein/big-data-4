"""
[DE-01] Módulo de Ingesta y Fusión de Datos.

Este módulo maneja la carga de archivos CSV y Excel, y la fusión de datos
de transacciones con información de estatus de órdenes.

Fuentes de datos soportadas:
- Produccion.xlsx (Hojas: 'Transacciones', 'Estatus')
- Archivos CSV: Transacciones.csv, Estatus.csv
"""

from pathlib import Path
import pandas as pd


class DataLoader:
    """Clase para cargar y fusionar datos de transacciones y estatus.

    Esta clase proporciona métodos para:
    - Cargar archivos desde Excel o CSV
    - Fusionar datos usando Left Join
    - Validar la integridad de los datos cargados

    Attributes:
        data_dir: Directorio donde se encuentran los archivos.
        excel_file: Nombre del archivo Excel principal.
        transactions_file: Nombre del archivo CSV de transacciones (fallback).
        status_file: Nombre del archivo CSV de estatus (fallback).
    """

    def __init__(
        self,
        data_dir: str | Path = "data/",
        excel_file: str = "Produccion.xlsx",
        transactions_file: str = "Transacciones.csv",
        status_file: str = "Estatus.csv",
    ) -> None:
        """Inicializa el DataLoader con las rutas de los archivos."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"El directorio {self.data_dir} no existe.")

        self.excel_path = self.data_dir / excel_file
        self.transactions_path = self.data_dir / transactions_file
        self.status_path = self.data_dir / status_file

    def _load_from_source(self, sheet_name: str, csv_path: Path) -> pd.DataFrame:
        """Helper para cargar desde Excel si existe, sino desde CSV."""
        df = None
        if self.excel_path.exists():
            try:
                # Leer desde Excel
                df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            except ValueError as e:
                print(
                    f"Advertencia: No se pudo leer la hoja '{sheet_name}' del Excel. Error: {e}"
                )
                # Fallback a CSV

        if df is None and csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8")

        if df is None:
            raise FileNotFoundError(
                f"No se encontraron datos para {sheet_name}. "
                f"Se buscó en {self.excel_path} (hoja: {sheet_name}) y {csv_path}"
            )

        # Normalizar columnas: minusculas y espacios a guiones bajos
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        return df

    def load_transactions(self) -> pd.DataFrame:
        """Carga el archivo de transacciones."""
        df = self._load_from_source("Transacciones", self.transactions_path)

        # Validar duplicados en id_orden
        if "id_orden" in df.columns and df["id_orden"].duplicated().any():
            print(
                f"Advertencia: Se encontraron {df['id_orden'].duplicated().sum()} IDs duplicados en transacciones."
            )

        # Parsear fecha si existe
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"])

        return df

    def load_status(self) -> pd.DataFrame:
        """Carga el archivo de estatus de órdenes."""
        return self._load_from_source("Estatus", self.status_path)

    def load_and_merge(self) -> pd.DataFrame:
        """Carga y fusiona los datos de transacciones con estatus."""
        df_trans = self.load_transactions()
        df_status = self.load_status()

        if "id_orden" not in df_trans.columns:
            raise KeyError(
                "La columna 'id_orden' no existe en los datos de Transacciones"
            )
        if "id_orden" not in df_status.columns:
            raise KeyError("La columna 'id_orden' no existe en los datos de Estatus")

        # Merge left join
        df_merged = pd.merge(df_trans, df_status, on="id_orden", how="left")

        return df_merged

    def validate_data(self, df: pd.DataFrame) -> dict[str, any]:
        """Valida la integridad del DataFrame fusionado."""
        required_cols = [
            "id_orden",
            "fecha",
            "cantidad",
            "prioridad",
            "tipo_cliente",
            "tipo_envio",
            "gerente",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]

        validation = {
            "total_records": len(df),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_ids": df["id_orden"].duplicated().sum()
            if "id_orden" in df.columns
            else 0,
            "is_valid": len(missing_cols) == 0,
        }

        if missing_cols:
            print(f"Columnas faltantes: {missing_cols}")

        return validation


def load_data(data_dir: str = "data/") -> pd.DataFrame:
    """Función helper para cargar datos directamente (interfaz simplificada)."""
    loader = DataLoader(data_dir=data_dir)
    return loader.load_and_merge()
