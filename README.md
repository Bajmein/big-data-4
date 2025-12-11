# 📉 Optimización de Devoluciones en Retail (Big Data Proyecto 4)

Este proyecto implementa un pipeline de Ciencia de Datos para analizar, predecir y optimizar las devoluciones en un entorno de retail. Utiliza técnicas de Machine Learning (Clasificación y Clustering) para identificar patrones de devolución y segmentar clientes, proporcionando una estrategia de negocio basada en datos.

## 🚀 Características Principales

*   **Ingesta y Procesamiento:** Fusión de datos transaccionales y de estado (ETL).
*   **Análisis Exploratorio (EDA):** Detección de causas raíz por gerente, transporte y temporalidad.
*   **Modelado Predictivo:** Clasificador **Random Forest** para predecir la probabilidad de devolución.
*   **Segmentación de Clientes:** Clustering **K-Means** para identificar perfiles de comportamiento.
*   **Reporte Automatizado:** Generación de dashboards ejecutivos y reportes estratégicos.

## 🛠️ Requisitos Previos

Antes de comenzar, asegúrate de tener instaladas las siguientes herramientas:

1.  **Python 3.11 o superior**
2.  **uv (Gestor de paquetes):**
    ```bash
    # Linux / macOS
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
3.  **Task (Gestor de tareas, opcional pero recomendado):**
    ```bash
    # Instalación via npm
    npm install -g @go-task/cli

    # O visita https://taskfile.dev/installation/ para otros métodos
    ```

## 📦 Instalación del Proyecto

1.  **Clonar el repositorio:**
    ```bash
    git clone <url-del-repo>
    cd bigdata-4
    ```

2.  **Instalar dependencias y entorno virtual con `uv`:**
    ```bash
    uv sync
    ```
    Este comando leerá `pyproject.toml`, creará un entorno virtual en `.venv` e instalará todas las librerías necesarias (pandas, scikit-learn, matplotlib, etc.).

## ▶️ Ejecución

El proyecto utiliza `Taskfile` para simplificar los comandos comunes.

### 1. Ejecutar el Pipeline Completo
Para ejecutar todo el proceso (ingesta -> modelado -> reporte):

```bash
task run
```
*Si no tienes instalado `task`, puedes usar el comando equivalente:*
```bash
uv run main.py
```

### 2. Verificar Calidad de Código (Linting)
Para revisar y corregir el estilo del código:

```bash
task check
```

### 📂 Salidas (Outputs)

Al finalizar la ejecución, se creará una carpeta `outputs/` con los siguientes artefactos:

*   `dashboard_kpis.png`: Dashboard visual con las métricas clave (Tasa de devolución, F1-Score, etc.).
*   `manager_performance.png`: Gráfico de desempeño por gerente.
*   `shipping_analysis.png`: Análisis de devoluciones por tipo de envío.
*   `customer_segments.png`: Visualización de los segmentos de clientes detectados.
*   `strategy_report.md`: Informe detallado con recomendaciones estratégicas y propuesta de arquitectura.

## 🏗️ Estructura del Proyecto

```text
.
├── data/                   # Archivos de entrada (Produccion.xlsx, etc.)
├── documentacion/          # Documentación (ARQUITECTURA.md, HOJA_DE_RUTA.md)
├── outputs/                # Resultados generados (Gráficos, Reportes)
├── src/                    # Código fuente
│   ├── ingesta/            # Carga y validación de datos
│   ├── procesamiento/      # Transformación y limpieza
│   ├── analisis/           # Análisis exploratorio (EDA)
│   ├── modelado/           # Modelos ML (Clasificador, Clustering)
│   └── reportes/           # Generación de visualizaciones y estrategia
├── main.py            # Punto de entrada principal
├── notebook.ipynb          # Cuaderno de experimentación
├── pyproject.toml          # Configuración de dependencias
├── taskfile.yaml           # Definición de tareas
└── README.md               # Este archivo
```

## 📊 Pipeline de Datos

1.  **Ingesta:** Carga datos de `data/`, realiza un *Left Join* entre transacciones y estatus.
2.  **Procesamiento:** Imputa valores nulos, codifica variables categóricas y extrae características temporales.
3.  **Modelado:**
    *   *Clasificación:* Entrena un modelo para predecir `is_returned`.
    *   *Clustering:* Agrupa clientes basado en frecuencia, monto y tasa de devolución.
4.  **Estrategia:** Analiza los resultados para sugerir acciones (ej. capacitación a gerentes específicos) y propone una arquitectura Big Data (Kafka + Spark) para escalar la solución.
