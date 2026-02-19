"""
Busca de hiperparâmetros com Optuna para o treinamento da PINN térmica 2D.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

try:
    from .analytics import EnhancedThermalAnalyzer, pde_residual_l2_normalized
    from .config import Config, initialize_runtime
    from .models import create_model
    from .sampling import create_training_data
    from .training import train_pinn_enhanced
except ImportError:  
    from analytics import EnhancedThermalAnalyzer, pde_residual_l2_normalized
    from config import Config, initialize_runtime
    from models import create_model
    from sampling import create_training_data
    from training import train_pinn_enhanced


def nondimensional_tuning_objective(
    metrics: Dict[str, float],
    *,
    domain_size: int,
    w1: float | None = None,
    w2: float | None = None,
    w3: float | None = None,
) -> float:
    """
    Combina métricas adimensionais para tuning robusto a escala e malha.

    Fórmula:
        w1 * relative_l2_error
      + w2 * pde_residual_l2_normalized
      + w3 * boundary_bc_error_nd
    """
    relative_l2 = float(metrics.get("relative_l2_error", float("inf")))
    pde_l2 = float(metrics.get("pde_residual_l2", float("inf")))
    pde_l2_norm = float(
        metrics.get(
            "pde_residual_l2_normalized",
            pde_residual_l2_normalized(
                pde_l2,
                domain_size=domain_size,
            ),
        )
    )
    boundary_nd = float(
        metrics.get("boundary_bc_error_nd", metrics.get("boundary_bc_error", float("inf")))
    )
    rw1 = float(getattr(Config, "TUNING_W_REL_L2", 1.0) if w1 is None else w1)
    rw2 = float(getattr(Config, "TUNING_W_PDE_RESIDUAL", 1.0) if w2 is None else w2)
    rw3 = float(getattr(Config, "TUNING_W_BOUNDARY_BC", 1.0) if w3 is None else w3)
    return (rw1 * relative_l2) + (rw2 * pde_l2_norm) + (rw3 * boundary_nd)


def _load_anchor_params(best_path: str | None) -> Dict[str, Any]:
    """Carrega parâmetros de ancoragem de uma execução prévia do Optuna."""
    if not best_path:
        return {}

    resolved = Path(best_path)
    if not resolved.exists():
        return {}

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    params = payload.get("best_params", {})
    return params if isinstance(params, dict) else {}


def _bounded_log_range(
    anchor: float,
    factor_low: float,
    factor_high: float,
    abs_low: float,
    abs_high: float,
) -> tuple[float, float]:
    """Constrói intervalo logarítmico ao redor da âncora com limites absolutos."""
    low = max(abs_low, float(anchor) * float(factor_low))
    high = min(abs_high, float(anchor) * float(factor_high))
    if low >= high:
        return abs_low, abs_high
    return low, high


def _clip_int_range(
    anchor: int, *, low: int, high: int, span: int, step: int
) -> tuple[int, int]:
    """Gera faixa inteira centrada na âncora, respeitando limites globais."""
    left = max(low, int(anchor) - span)
    right = min(high, int(anchor) + span)
    if left >= right:
        return low, high
    left = int(round(left / step) * step)
    right = int(round(right / step) * step)
    if left < low:
        left = low
    if right > high:
        right = high
    if left >= right:
        return low, high
    return left, right


def _stability_score(history: Dict[str, list[float]]) -> float:
    """Calcula penalização de estabilidade a partir das séries de treino."""
    losses = np.asarray(history.get("adam_loss", []), dtype=np.float64)
    if losses.size < 8:
        return 1.0

    safe_losses = np.clip(losses, 1e-12, None)
    log_loss = np.log10(safe_losses)
    dlog = np.diff(log_loss)
    if dlog.size == 0:
        return 1.0

    median_abs_step = float(np.median(np.abs(dlog)))
    spike_ratio = float(np.mean(np.abs(dlog) > 0.09))
    tail_start = int(0.65 * dlog.size)
    tail_dlog = dlog[tail_start:] if tail_start < dlog.size else dlog
    tail_vol = float(np.std(tail_dlog))

    grad_norm = np.asarray(history.get("grad_norm", []), dtype=np.float64)
    if grad_norm.size >= 8:
        safe_grad = np.log10(np.clip(grad_norm, 1e-12, None))
        grad_spike_ratio = float(np.mean(np.abs(np.diff(safe_grad)) > 0.12))
    else:
        grad_spike_ratio = 0.0

    return (
        median_abs_step
        + (2.5 * spike_ratio)
        + tail_vol
        + (0.8 * grad_spike_ratio)
    )


def optimize_hyperparameters(
    *,
    n_trials: int = 30,
    timeout: int | None = None,
    epochs_adam: int = 500,
    epochs_lbfgs: int = 0,
    n_interior: int = 2000,
    n_boundary: int = 200,
    n_data: int = 2000,
    interior_strategy: str = Config.DEFAULT_INTERIOR_STRATEGY,
    eval_domain_size: int = 64,
    results_path: str = "runs/results/optuna_best.json",
    base_best_path: str | None = "runs/results/optuna_best.json",
    stability_lambda: float = 3.50e-1,
) -> Dict[str, Any]:
    """
    Executa estudo de otimização e salva o melhor resultado em JSON.

    O objetivo minimiza uma combinação adimensional de acurácia global,
    consistência física (PDE) e aderência às condições de contorno.
    """
    runtime_logger, runtime_device = initialize_runtime(emit_device_banner=False)

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna não está instalado. Instale com: pip install optuna"
        ) from exc

    eval_domain_size = max(16, int(eval_domain_size))
    runtime_logger.info(
        "Iniciando Optuna com %d trials (epochs_adam=%d, epochs_lbfgs=%d).",
        n_trials,
        epochs_adam,
        epochs_lbfgs,
    )

    anchor_params = _load_anchor_params(base_best_path)
    if anchor_params:
        runtime_logger.info("Busca ancorada em parâmetros prévios de: %s", base_best_path)
    else:
        runtime_logger.info("Busca sem âncora prévia; usando defaults atuais do Config.")

    training_data = create_training_data(
        N_interior=int(n_interior),
        N_boundary=int(n_boundary),
        N_data=int(n_data),
        interior_strategy=interior_strategy,
        device_=runtime_device,
    )

    sampler = optuna.samplers.TPESampler(seed=Config.SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    base_lr = float(anchor_params.get("lr", Config.DEFAULT_LR))
    base_w_data = float(anchor_params.get("w_data", Config.DEFAULT_W_DATA))
    base_w_dir = float(anchor_params.get("w_dir", Config.DEFAULT_WEIGHT_PARAMS["w_dir"]))
    base_w_neu = float(anchor_params.get("w_neu", Config.DEFAULT_WEIGHT_PARAMS["w_neu"]))
    base_alpha = float(anchor_params.get("alpha", Config.DEFAULT_WEIGHT_PARAMS["alpha"]))
    base_ratio = float(
        anchor_params.get(
            "max_weight_ratio", Config.DEFAULT_WEIGHT_PARAMS["max_weight_ratio"]
        )
    )
    base_change = float(
        anchor_params.get(
            "max_weight_change", Config.DEFAULT_WEIGHT_PARAMS["max_weight_change"]
        )
    )
    base_balance = int(
        anchor_params.get(
            "balance_frequency", Config.DEFAULT_WEIGHT_PARAMS["balance_frequency"]
        )
    )
    base_min_step = int(
        anchor_params.get(
            "min_balance_step", Config.DEFAULT_WEIGHT_PARAMS["min_balance_step"]
        )
    )
    base_window = int(anchor_params.get("window", Config.DEFAULT_WEIGHT_PARAMS["window"]))

    lr_low, lr_high = _bounded_log_range(base_lr, 4.00e-1, 2.00e0, 1.00e-5, 5.00e-3)
    w_data_low, w_data_high = _bounded_log_range(
        base_w_data, 5.00e-1, 2.20e0, 1.00e-1, 8.00e2
    )
    w_dir_low, w_dir_high = _bounded_log_range(
        base_w_dir, 5.00e-1, 2.20e0, 2.00e1, 2.00e3
    )
    w_neu_low, w_neu_high = _bounded_log_range(
        base_w_neu, 5.00e-1, 2.20e0, 1.00e0, 8.00e2
    )
    alpha_low, alpha_high = _bounded_log_range(base_alpha, 9.50e-1, 1.03e0, 8.50e-1, 9.95e-1)
    ratio_low, ratio_high = _bounded_log_range(base_ratio, 6.00e-1, 1.50e0, 5.00e1, 5.00e2)
    change_low, change_high = _bounded_log_range(
        base_change, 6.00e-1, 1.40e0, 5.00e-2, 3.00e-1
    )
    balance_low, balance_high = _clip_int_range(
        base_balance, low=50, high=400, span=120, step=10
    )
    min_step_low, min_step_high = _clip_int_range(
        base_min_step, low=200, high=1800, span=500, step=50
    )
    window_low, window_high = _clip_int_range(base_window, low=10, high=120, span=40, step=5)

    def objective(trial: "optuna.Trial") -> float:
        """Avalia uma configuração ancorada e retorna objetivo erro+estabilidade."""
        lr = trial.suggest_float("lr", lr_low, lr_high, log=True)
        w_data = trial.suggest_float("w_data", w_data_low, w_data_high, log=True)
        w_dir = trial.suggest_float("w_dir", w_dir_low, w_dir_high, log=True)
        w_neu = trial.suggest_float("w_neu", w_neu_low, w_neu_high, log=True)
        alpha = trial.suggest_float("alpha", alpha_low, alpha_high)
        max_weight_ratio = trial.suggest_float(
            "max_weight_ratio", ratio_low, ratio_high, log=True
        )
        max_weight_change = trial.suggest_float(
            "max_weight_change", change_low, change_high
        )
        balance_frequency = trial.suggest_int(
            "balance_frequency", balance_low, balance_high, step=10
        )
        min_balance_step = trial.suggest_int(
            "min_balance_step", min_step_low, min_step_high, step=50
        )
        window = trial.suggest_int("window", window_low, window_high, step=5)

        layers = list(Config.DEFAULT_LAYERS)
        weight_params = dict(Config.DEFAULT_WEIGHT_PARAMS)
        weight_params["w_dir"] = float(w_dir)
        weight_params["w_neu"] = float(w_neu)
        weight_params["w_pde"] = float(weight_params.get("w_pde", 1.0))
        weight_params["alpha"] = float(alpha)
        weight_params["max_weight_ratio"] = float(max_weight_ratio)
        weight_params["max_weight_change"] = float(max_weight_change)
        weight_params["balance_frequency"] = int(balance_frequency)
        weight_params["min_balance_step"] = int(min_balance_step)
        weight_params["window"] = int(window)
        weight_params["use_median"] = bool(weight_params.get("use_median", True))

        model = create_model(
            model_type="enhanced",
            layers=layers,
            activation=Config.DEFAULT_ACTIVATION,
        ).to(runtime_device)

        history = train_pinn_enhanced(
            net=model,
            data=training_data,
            epochs_adam=int(epochs_adam),
            epochs_lbfgs=int(epochs_lbfgs),
            lr=float(lr),
            weight_params=weight_params,
            w_data=float(w_data),
            use_adaptive_weights=True,
            verbose=False,
        )

        analyzer = EnhancedThermalAnalyzer(
            model, domain_size=eval_domain_size, device_=runtime_device
        )
        relative_l2 = float(analyzer.metrics["relative_l2_error"])
        rmse = float(analyzer.metrics["RMSE"])
        r2 = float(analyzer.metrics["R2"])
        pde_l2 = float(analyzer.metrics["pde_residual_l2"])
        pde_l2_normalized = float(
            analyzer.metrics.get(
                "pde_residual_l2_normalized",
                pde_residual_l2_normalized(
                    pde_l2,
                    domain_size=eval_domain_size,
                ),
            )
        )
        boundary_err_nd = float(
            analyzer.metrics.get(
                "boundary_bc_error_nd",
                analyzer.metrics.get("boundary_bc_error", float("inf")),
            )
        )
        stability = _stability_score(history)
        objective_value = nondimensional_tuning_objective(
            {
                "relative_l2_error": relative_l2,
                "pde_residual_l2": pde_l2,
                "pde_residual_l2_normalized": pde_l2_normalized,
                "boundary_bc_error_nd": boundary_err_nd,
            },
            domain_size=eval_domain_size,
        )

        if history.get("adam_loss"):
            trial.report(float(objective_value), step=len(history["adam_loss"]))
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not np.isfinite(objective_value):
            return float("inf")

        trial.set_user_attr("relative_l2_error", relative_l2)
        trial.set_user_attr("rmse", rmse)
        trial.set_user_attr("r2", r2)
        trial.set_user_attr("pde_residual_l2", pde_l2)
        trial.set_user_attr("pde_residual_l2_normalized", pde_l2_normalized)
        trial.set_user_attr("boundary_bc_error_nd", boundary_err_nd)
        trial.set_user_attr("stability", stability)
        trial.set_user_attr("objective", objective_value)
        trial.set_user_attr("layers", layers)
        trial.set_user_attr("use_adaptive_weights", True)
        return float(objective_value)

    study.optimize(objective, n_trials=int(n_trials), timeout=timeout)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params["use_adaptive_weights"] = True
    best_result: Dict[str, Any] = {
        "best_value_objective": float(best_trial.value),
        "best_value_relative_l2": float(
            best_trial.user_attrs.get("relative_l2_error", best_trial.value)
        ),
        # Compatibilidade retroativa com consumidores antigos do JSON.
        "best_value_mae": float(
            best_trial.user_attrs.get("relative_l2_error", best_trial.value)
        ),
        "best_params": best_params,
        "best_user_attrs": dict(best_trial.user_attrs),
        "n_trials": int(len(study.trials)),
        "fixed_config": {
            "layers": list(Config.DEFAULT_LAYERS),
            "activation": Config.DEFAULT_ACTIVATION,
            "use_adaptive_weights": True,
        },
        "anchor_params": anchor_params,
    }

    output_file = Path(results_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(best_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    runtime_logger.info(
        "Optuna finalizado. Melhor objetivo: %.4e | Relative L2: %.4e",
        best_result["best_value_objective"],
        best_result["best_value_relative_l2"],
    )
    runtime_logger.info("Resultados salvos em: %s", output_file)
    runtime_logger.info(
        "Configuração global não foi alterada automaticamente. "
        "Use o comando dedicado para aplicar os melhores parâmetros ao config.py."
    )
    return best_result

