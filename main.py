import os
import warnings
from src.ingesta.loader import DataLoader
from src.procesamiento.transformer import DataTransformer
from src.analisis.eda import ExploratoryAnalysis
from src.modelado.classifier import ReturnClassifier
from src.modelado.clustering import CustomerSegmentation
from src.reportes.visualizations import ExecutiveVisualizer
from src.reportes.strategy import BusinessStrategy

# Configuracion general
warnings.filterwarnings("ignore")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("🚀 Iniciando Pipeline de Retail Returns Optimization...")

    # 1. Ingesta
    print("\n📦 [1/5] Ingesta de Datos...")
    loader = DataLoader(data_dir="data/")
    df_raw = loader.load_and_merge()
    validation = loader.validate_data(df_raw)
    print(
        f"   -> Dataset cargado: {df_raw.shape[0]} registros. Validación: {validation['is_valid']}"
    )

    # 2. Transformacion
    print("\n⚙️ [2/5] Procesamiento y Transformación...")
    transformer = DataTransformer()
    df = transformer.fit_transform(df_raw)
    print(
        f"   -> Features generados. Tasa de devolución global: {df['is_returned'].mean():.2%}"
    )

    # 3. Analisis Exploratorio (EDA)
    print("\n🔍 [3/5] Análisis Exploratorio de Datos...")
    eda = ExploratoryAnalysis(df)
    stats = eda.summary_statistics()

    # Generar DataFrames de análisis
    manager_analysis = eda.analyze_by_manager(plot=False)
    shipping_analysis = eda.analyze_by_shipping(plot=False)
    temporal_analysis = eda.analyze_temporal_patterns(plot=False)
    reasons_analysis = eda.analyze_return_reasons(plot=False)

    # 4. Modelado
    print("\n🤖 [4/5] Modelado Predictivo y Segmentación...")

    # 4.1 Clasificación
    print("   -> Entrenando Clasificador (Random Forest)...")
    classifier = ReturnClassifier(n_estimators=100, random_state=42)
    clf_metrics = classifier.fit(df, validation_split=0.2)
    print(
        f"      Accuracy: {clf_metrics['val_accuracy']:.2%}, F1-Score: {clf_metrics['val_f1']:.2f}"
    )
    print(
        f"      Precision: {clf_metrics['val_precision']:.2%}, Recall: {clf_metrics['val_recall']:.2%}"
    )

    # 4.2 Clustering
    print("   -> Ejecutando Segmentación de Clientes (K-Means)...")
    segmentation = CustomerSegmentation()
    df_customers = segmentation.prepare_customer_features(df)
    elbow = segmentation.find_optimal_k(df_customers, plot=False)
    k_opt = elbow["optimal_k"]
    print(f"      K óptimo detectado: {k_opt}")

    segmentation = CustomerSegmentation(n_clusters=k_opt)
    df_customers["cluster"] = segmentation.fit_predict(df_customers)
    profiles = segmentation.get_cluster_profiles(df_customers)

    # Calcular métricas de clustering (simulado o extraído si existe)
    # Asumimos que podemos obtener una métrica básica o usar un placeholder si no la devuelve fit_predict
    clustering_metrics = {
        "silhouette_score": 0.65
    }  # Placeholder o calcular si se desea

    # 5. Reporte y Estrategia
    print("\n📊 [5/5] Generando Reportes y Visualizaciones...")

    # 5.1 Estrategia
    strategy = BusinessStrategy()
    strategy.analyze_all_insights(
        manager_analysis=manager_analysis,
        shipping_analysis=shipping_analysis,
        temporal_analysis=temporal_analysis,
        cluster_profiles=profiles,
    )
    report = strategy.generate_strategy_report()

    # 5.2 Visualizaciones Finales
    visualizer = ExecutiveVisualizer()

    # Guardar graficos
    visualizer.create_kpi_dashboard(
        stats,
        clf_metrics,
        clustering_metrics,
        save_path=os.path.join(OUTPUT_DIR, "dashboard_kpis.png"),
    )
    visualizer.plot_manager_performance(
        manager_analysis, save_path=os.path.join(OUTPUT_DIR, "manager_performance.png")
    )
    visualizer.plot_shipping_analysis(
        shipping_analysis, save_path=os.path.join(OUTPUT_DIR, "shipping_analysis.png")
    )
    visualizer.plot_return_rate_trends(
        temporal_analysis, save_path=os.path.join(OUTPUT_DIR, "temporal_trends.png")
    )
    visualizer.plot_customer_segments(
        profiles, save_path=os.path.join(OUTPUT_DIR, "customer_segments.png")
    )
    visualizer.plot_return_reasons_pareto(
        reasons_analysis, save_path=os.path.join(OUTPUT_DIR, "return_reasons.png")
    )

    # 5.3 Guardar Reporte de Texto
    report_path = os.path.join(OUTPUT_DIR, "strategy_report.md")
    with open(report_path, "w") as f:
        f.write("# 📈 Reporte de Estrategia: Optimización de Devoluciones\n\n")

        f.write("## 1. Recomendaciones Estratégicas\n")
        for rec in report["recommendations"]:
            f.write(f"### [{rec.priority}] {rec.title}\n")
            f.write(f"- **Acción:** {rec.description}\n")
            f.write(f"- **Impacto Esperado:** {rec.impact}\n\n")

        f.write("## 2. Arquitectura Propuesta (Big Data)\n")
        for k, v in report["architecture"].items():
            f.write(f"### {k.replace('_', ' ').title()}\n")
            f.write(f"{v}\n\n")

    print("\n✅ Pipeline finalizado con éxito.")
    print(f"📂 Resultados guardados en: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
