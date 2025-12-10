"""
Módulo de Ingesta de Datos.

Contiene funcionalidades para cargar y fusionar los archivos CSV
de transacciones y estatus de órdenes.
"""

from .loader import DataLoader

__all__ = ["DataLoader"]
