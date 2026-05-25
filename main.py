"""
PINN-Thermal 2D: Primary execution entry point for scientific conduction analysis.
Standardized for academic publication (CMAME/JCP style).
"""
import time
import torch
from config import Config, device, logger
from sampling import create_training_data
from models import create_model, print_model_summary
from training import train_pinn_enhanced
from analytics import EnhancedThermalAnalyzer, analyze_convergence
from plotting import PublicationPlotter

def run_scientific_analysis():
    """Orchestrates the modular PINN workflow for high-fidelity heat transfer analysis."""
    logger.info("=" * 80)
    logger.info("Rede Neural Informada pela Física: Condução Térmica Bidimensional em Regime Permanente")
    logger.info("=" * 80)
    logger.info(f"Dispositivo: {device}")
    logger.info("=" * 80)

    # 1. Scientific Data Generation
    N_f = Config.DEFAULT_N_INTERIOR 
    N_boundary = Config.DEFAULT_N_BOUNDARY
    
    logger.info("Gerando malhas de colocation de alta densidade (Sobol)...")
    X_f, X_dir, T_dir_target, X_bottom, X_top = create_training_data(
        N_interior=N_f, N_boundary=N_boundary, interior_strategy="sobol"
    )
    
    # 2. Model Initialization
    logger.info("Construindo manifold neural profundo...")
    net = create_model(
        model_type="enhanced",
        layers=Config.DEFAULT_LAYERS,
        activation=Config.DEFAULT_ACTIVATION,
        hard_constraint_bc=Config.HARD_CONSTRAINT_BC
    ).to(device)
    
    print_model_summary(net)
    
    # 3. Hybrid Optimization Regime
    logger.info("Iniciando otimização híbrida de alta precisão...")
    start_time = time.time()
    
    hist = train_pinn_enhanced(
        net, X_f, X_dir, T_dir_target, X_bottom, X_top,
        epochs_adam=Config.DEFAULT_EPOCHS_ADAM, 
        epochs_lbfgs=Config.DEFAULT_EPOCHS_LBFGS,
        lr=Config.DEFAULT_LR,
        hard_constraint_bc=Config.HARD_CONSTRAINT_BC,
        verbose=True
    )
    
    logger.info(f"Tempo computacional total: {time.time() - start_time:.1f}s")

    # 4. Rigorous Post-Processing and Validation
    logger.info("Realizando análise rigorosa de erros e verificações de consistência física...")
    analyzer = EnhancedThermalAnalyzer(net, domain_size=Config.DOMAIN_SIZE, device_=device)
    analyzer.print_comprehensive_analysis()

    # Salvar relatório textual
    with open("scientific_report.txt", "w", encoding="utf-8") as f:
        f.write("RELATÓRIO TÉCNICO DE DESEMPENHO PINN\n")
        f.write("="*40 + "\n")
        f.write(f"MAE: {analyzer.metrics['MAE']:.4e} K\n")
        f.write(f"RMSE: {analyzer.metrics['RMSE']:.4e} K\n")
        f.write(f"R2: {analyzer.metrics['R2']:.6f}\n")
        f.write(f"Max Error: {analyzer.metrics['Max_Error']:.4e} K\n")
        f.write("="*40 + "\n")

    # Convergence Diagnostics
    conv_stats = analyze_convergence(hist)
    if conv_stats:
        logger.info(f"\nPerfil de Convergência: Perda Residual Final = {conv_stats['final_loss']:.2e}")

    # Geração de Gráficos Científicos e Dashboards
    logger.info("Gerando visualizações de nível de publicação e dashboards de diagnóstico...")
    training_data = {
        'X_f': X_f,
        'X_dir': X_dir,
        'X_bottom': X_bottom,
        'X_top': X_top
    }
    plotter = PublicationPlotter(analyzer, hist, output_dir="plots", training_data=training_data)
    plotter.run_all_enhanced_plots()
    
    logger.info("=" * 80)
    logger.info("AVALIAÇÃO TÉCNICA CONCLUÍDA: ARQUIVO PRONTO")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_scientific_analysis()
