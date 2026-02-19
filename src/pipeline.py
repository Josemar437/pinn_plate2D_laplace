"""
Orquestração do fluxo completo: treino, avaliação, relatório e geração de figuras.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

try:
    from .analytics import (
        EnhancedThermalAnalyzer,
        analyze_convergence,
        assess_publication_readiness,
    )
    from .config import Config, device, initialize_runtime, physical_scales
    from .models import create_model, print_model_summary
    from .plotting import PublicationPlotter
    from .sampling import create_training_data
    from .training import train_pinn_enhanced
except ImportError:  
    from analytics import (
        EnhancedThermalAnalyzer,
        analyze_convergence,
        assess_publication_readiness,
    )
    from config import Config, device, initialize_runtime, physical_scales
    from models import create_model, print_model_summary
    from plotting import PublicationPlotter
    from sampling import create_training_data
    from training import train_pinn_enhanced


def _json_safe(value: Any) -> Any:
    """Converte objetos de configuração para tipos JSON-serializáveis."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _build_reproducibility_snapshot(
    *, active_device: torch.device | None = None
) -> Dict[str, Any]:
    """
    Registra estado de execução para reprodutibilidade científica.

    Inclui configuração completa, escalas físicas de adimensionalização,
    seeds e dtype padrão usado pelo PyTorch.
    """
    config_payload: Dict[str, Any] = {}
    for attr in dir(Config):
        if not attr.isupper():
            continue
        value = getattr(Config, attr)
        if callable(value):
            continue
        config_payload[attr] = _json_safe(value)

    temp_scale, length_scale = physical_scales()
    resolved_device = active_device if active_device is not None else device
    return {
        "config": config_payload,
        "scales": {
            "temp_scale": temp_scale,
            "length_scale": length_scale,
        },
        "reproducibility": {
            "seed": int(Config.SEED),
            "numpy_seed": int(Config.SEED),
            "torch_seed": int(Config.SEED),
            "torch_default_dtype": str(torch.get_default_dtype()),
            "use_double": bool(Config.USE_DOUBLE),
            "device": str(resolved_device),
        },
    }


def run_scientific_analysis(
    *,
    output_dir: str | Path = Config.PLOTS_DIR,
    report_path: str | Path = Config.RESULTS_DIR / "scientific_report.txt",
    results_dir: str | Path | None = None,
    interior_strategy: str = Config.DEFAULT_INTERIOR_STRATEGY,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executa o fluxo completo do experimento com a configuração ativa.

    Retorna objetos úteis para inspeção programática após a execução.
    """
    runtime_logger, runtime_device = initialize_runtime(emit_device_banner=False)

    runtime_logger.info("=" * 80)
    runtime_logger.info(
        "Rede Neural Informada pela Física: Condução Térmica Bidimensional em Regime Permanente"
    )
    runtime_logger.info("=" * 80)
    runtime_logger.info("Dispositivo: %s", runtime_device)
    runtime_logger.info("=" * 80)

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_report_path = Path(report_path)
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_results_dir = (
        Path(results_dir) if results_dir is not None else resolved_report_path.parent
    )
    resolved_results_dir.mkdir(parents=True, exist_ok=True)

    training_data = create_training_data(
        N_interior=Config.DEFAULT_N_INTERIOR,
        N_boundary=Config.DEFAULT_N_BOUNDARY,
        N_data=Config.DEFAULT_N_DATA,
        interior_strategy=interior_strategy,
        device_=runtime_device,
    )

    runtime_logger.info("Construindo manifold neural profundo...")
    net = create_model(
        model_type="enhanced",
        layers=Config.DEFAULT_LAYERS,
        activation=Config.DEFAULT_ACTIVATION,
    ).to(runtime_device)
    print_model_summary(net)

    runtime_logger.info("Iniciando otimização híbrida de alta precisão...")
    start_time = time.time()
    history = train_pinn_enhanced(
        net=net,
        data=training_data,
        epochs_adam=Config.DEFAULT_EPOCHS_ADAM,
        epochs_lbfgs=Config.DEFAULT_EPOCHS_LBFGS,
        lr=Config.DEFAULT_LR,
        w_data=Config.DEFAULT_W_DATA,
        use_adaptive_weights=Config.USE_ADAPTIVE_WEIGHTING,
        verbose=verbose,
    )
    runtime_logger.info("Tempo computacional total: %.1fs", time.time() - start_time)

    analyzer = EnhancedThermalAnalyzer(
        net,
        domain_size=Config.DOMAIN_SIZE,
        device_=runtime_device,
    )
    analyzer.print_comprehensive_analysis()

    report_lines = [
        "RELATORIO TECNICO DE DESEMPENHO PINN",
        "=" * 60,
        f"MAE: {analyzer.metrics['MAE']:.6e} K",
        f"RMSE: {analyzer.metrics['RMSE']:.6e} K",
        f"MAPE: {analyzer.metrics['MAPE']:.6e} %",
        f"R2: {analyzer.metrics['R2']:.8f}",
        f"Max Error: {analyzer.metrics['max_error']:.6e} K",
        f"Relative L2 Error: {analyzer.metrics['relative_l2_error']:.6e}",
        f"Relative L2 Error vs FDM: {analyzer.metrics['relative_l2_error_vs_fdm']:.6e}",
        f"Boundary Field Error: {analyzer.metrics['boundary_field_error']:.6e}",
        f"Boundary BC Error ND: {analyzer.metrics['boundary_bc_error_nd']:.6e}",
        f"Boundary Dirichlet Error ND: {analyzer.metrics['boundary_dirichlet_error_nd']:.6e}",
        f"Boundary Neumann Error ND: {analyzer.metrics['boundary_neumann_error_nd']:.6e}",
        f"Boundary BC Dirichlet Left ND: {analyzer.metrics['boundary_bc_error_dirichlet_left']:.6e}",
        f"Boundary BC Dirichlet Right ND: {analyzer.metrics['boundary_bc_error_dirichlet_right']:.6e}",
        f"Boundary BC Dirichlet Ratio (R/L): {analyzer.metrics['boundary_bc_error_dirichlet_ratio_right_left']:.6e}",
        f"Boundary BC Neumann ND: {analyzer.metrics['boundary_bc_error_neumann']:.6e}",
        f"Midline y=0.5 MAE: {analyzer.metrics['midline_y_half_mae']:.6e}",
        f"Midline y=0.5 RMSE: {analyzer.metrics['midline_y_half_rmse']:.6e}",
        f"Midline y=0.5 Max Error: {analyzer.metrics['midline_y_half_max_error']:.6e}",
        f"Midline y=0.5 Curvature RMSE: {analyzer.metrics['midline_y_half_curvature_rmse']:.6e}",
        f"Midline y=0.5 Curvature Max: {analyzer.metrics['midline_y_half_curvature_max']:.6e}",
        f"PDE Residual Mean: {analyzer.metrics['pde_residual_mean']:.6e}",
        f"PDE Residual L2: {analyzer.metrics['pde_residual_l2']:.6e}",
        f"PDE Residual RMS: {analyzer.metrics['pde_residual_rms']:.6e}",
        f"PDE Residual Max: {analyzer.metrics['pde_residual_max']:.6e}",
        f"PDE Residual ND Mean: {analyzer.metrics['pde_residual_nd_mean']:.6e}",
        f"PDE Residual ND L2: {analyzer.metrics['pde_residual_nd_l2']:.6e}",
        f"PDE Residual ND RMS: {analyzer.metrics['pde_residual_nd_rms']:.6e}",
        f"PDE Residual ND Max: {analyzer.metrics['pde_residual_nd_max']:.6e}",
        f"PDE Residual L2 Normalized: {analyzer.metrics['pde_residual_l2_normalized']:.6e}",
        "=" * 60,
    ]
    resolved_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    conv_stats = analyze_convergence(history)
    readiness = assess_publication_readiness(history, analyzer.metrics)
    if conv_stats:
        runtime_logger.info(
            "\nPerfil de Convergência: Perda Total Final = %.2e",
            conv_stats["final_loss"],
        )
    runtime_logger.info(
        "Status para publicação: %s",
        readiness["status"],
    )
    if readiness["status"] != "Approved":
        runtime_logger.warning(
            "Resultados classificados como Request Changes. Falhas: %s",
            ", ".join(readiness["failed_checks"]),
        )

    report_append = [
        f"Publication Status: {readiness['status']}",
        f"Publication Ready: {readiness['publication_ready']}",
    ]
    if readiness["failed_checks"]:
        report_append.append(
            "Failed Checks: " + ", ".join(readiness["failed_checks"])
        )
    report_append.append("=" * 60)
    with resolved_report_path.open("a", encoding="utf-8") as fobj:
        fobj.write("\n".join(report_append) + "\n")

    metrics_payload = dict(analyzer.metrics)
    metrics_payload.update(
        {
            "fdm_iterations": analyzer.fdm_iterations,
            "final_loss": conv_stats.get("final_loss") if conv_stats else None,
            "total_steps": conv_stats.get("total_steps") if conv_stats else None,
            "total_epochs": conv_stats.get("total_epochs") if conv_stats else None,
            "loss_reduction": conv_stats.get("loss_reduction") if conv_stats else None,
            "publication_status": readiness["status"],
            "publication_ready": readiness["publication_ready"],
            "failed_checks": readiness["failed_checks"],
            "convergence_checks": readiness["checks"],
        }
    )
    (resolved_results_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (resolved_results_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (resolved_results_dir / "reproducibility_snapshot.json").write_text(
        json.dumps(
            _build_reproducibility_snapshot(active_device=runtime_device),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    np.save(resolved_results_dir / "pde_residual.npy", analyzer.pde_residual)

    plotting_data = {
        "X_f": training_data.X_f,
        "X_dir": training_data.X_dir,
        "X_bottom": training_data.X_bottom,
        "X_top": training_data.X_top,
        "X_data": training_data.X_data,
        "X_midline": training_data.X_midline,
    }
    plotter = PublicationPlotter(
        analyzer,
        history,
        output_dir=str(resolved_output_dir),
        training_data=plotting_data,
    )
    plotter.run_all_enhanced_plots()

    runtime_logger.info("=" * 80)
    runtime_logger.info("AVALIAÇÃO TÉCNICA CONCLUÍDA: ARQUIVO PRONTO")
    runtime_logger.info("=" * 80)

    return {
        "net": net,
        "history": history,
        "analyzer": analyzer,
        "training_data": training_data,
        "convergence": conv_stats,
        "report_path": str(resolved_report_path),
        "results_dir": str(resolved_results_dir),
        "plots_dir": str(resolved_output_dir),
    }

