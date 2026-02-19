"""
Interface de linha de comando para treino, tuning e sincronização de configuração.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config_sync import apply_best_to_config_file


def _apply_train_overrides(args: argparse.Namespace) -> None:
    """Sobrescreve parâmetros padrão do Config com flags da CLI."""
    from src.config import Config

    if args.epochs_adam is not None:
        Config.DEFAULT_EPOCHS_ADAM = int(args.epochs_adam)
    if args.epochs_lbfgs is not None:
        Config.DEFAULT_EPOCHS_LBFGS = int(args.epochs_lbfgs)
    if args.lr is not None:
        Config.DEFAULT_LR = float(args.lr)
    if args.n_interior is not None:
        Config.DEFAULT_N_INTERIOR = int(args.n_interior)
    if args.n_boundary is not None:
        Config.DEFAULT_N_BOUNDARY = int(args.n_boundary)
    if args.n_data is not None:
        Config.DEFAULT_N_DATA = int(args.n_data)
    if getattr(args, "n_midline", None) is not None:
        Config.DEFAULT_N_MIDLINE = int(args.n_midline)
    if args.interior_strategy is not None:
        Config.DEFAULT_INTERIOR_STRATEGY = str(args.interior_strategy).strip().lower()
    if args.w_data is not None:
        Config.DEFAULT_W_DATA = float(args.w_data)
    if args.fdm_nx is not None:
        Config.FDM_NX = int(args.fdm_nx)
    if args.fdm_ny is not None:
        Config.FDM_NY = int(args.fdm_ny)
    if args.domain_size is not None:
        Config.DOMAIN_SIZE = int(args.domain_size)
    if getattr(args, "activation", None) is not None:
        Config.DEFAULT_ACTIVATION = str(args.activation).strip().lower()
    if getattr(args, "input_normalization", None) is not None:
        Config.INPUT_NORMALIZATION = str(args.input_normalization).strip().lower()
    if getattr(args, "enable_hard_dirichlet_constraint", None) is not None:
        Config.ENABLE_HARD_DIRICHLET_CONSTRAINT = bool(
            args.enable_hard_dirichlet_constraint
        )
    if getattr(args, "right_dirichlet_multiplier", None) is not None:
        Config.RIGHT_DIRICHLET_MULTIPLIER = float(args.right_dirichlet_multiplier)
    if getattr(args, "left_dirichlet_weight_multiplier", None) is not None:
        Config.LEFT_DIRICHLET_WEIGHT_MULTIPLIER = float(
            args.left_dirichlet_weight_multiplier
        )
    if getattr(args, "right_dirichlet_loss_multiplier", None) is not None:
        Config.RIGHT_DIRICHLET_LOSS_MULTIPLIER = float(
            args.right_dirichlet_loss_multiplier
        )
    if getattr(args, "auto_balance_dirichlet_sides", None) is not None:
        Config.AUTO_BALANCE_DIRICHLET_SIDES = bool(args.auto_balance_dirichlet_sides)
    if getattr(args, "dirichlet_side_target_ratio", None) is not None:
        Config.DIRICHLET_SIDE_TARGET_RATIO = float(args.dirichlet_side_target_ratio)
    if getattr(args, "dirichlet_side_ratio_tolerance", None) is not None:
        Config.DIRICHLET_SIDE_RATIO_TOLERANCE = float(
            args.dirichlet_side_ratio_tolerance
        )
    if getattr(args, "right_dirichlet_loss_adapt_rate", None) is not None:
        Config.RIGHT_DIRICHLET_LOSS_ADAPT_RATE = float(
            args.right_dirichlet_loss_adapt_rate
        )
    if getattr(args, "left_dirichlet_loss_min_multiplier", None) is not None:
        Config.LEFT_DIRICHLET_LOSS_MIN_MULTIPLIER = float(
            args.left_dirichlet_loss_min_multiplier
        )
    if getattr(args, "left_dirichlet_loss_max_multiplier", None) is not None:
        Config.LEFT_DIRICHLET_LOSS_MAX_MULTIPLIER = float(
            args.left_dirichlet_loss_max_multiplier
        )
    if getattr(args, "right_dirichlet_loss_min_multiplier", None) is not None:
        Config.RIGHT_DIRICHLET_LOSS_MIN_MULTIPLIER = float(
            args.right_dirichlet_loss_min_multiplier
        )
    if getattr(args, "right_dirichlet_loss_max_multiplier", None) is not None:
        Config.RIGHT_DIRICHLET_LOSS_MAX_MULTIPLIER = float(
            args.right_dirichlet_loss_max_multiplier
        )
    if getattr(args, "curvature_reg_weight", None) is not None:
        Config.CURVATURE_REG_WEIGHT = float(args.curvature_reg_weight)
    if getattr(args, "left_importance_fraction", None) is not None:
        Config.LEFT_IMPORTANCE_FRACTION = float(args.left_importance_fraction)
    if getattr(args, "left_importance_band_fraction", None) is not None:
        Config.LEFT_IMPORTANCE_BAND_FRACTION = float(args.left_importance_band_fraction)
    if getattr(args, "enable_physics_polish", None) is not None:
        Config.ENABLE_PHYSICS_POLISH = bool(args.enable_physics_polish)
    if getattr(args, "physics_polish_iters", None) is not None:
        Config.PHYSICS_POLISH_ITERS = int(args.physics_polish_iters)
        Config.ENABLE_PHYSICS_POLISH = int(args.physics_polish_iters) > 0
    if getattr(args, "physics_polish_target_pde_mean", None) is not None:
        Config.PHYSICS_POLISH_TARGET_PDE_MEAN = float(args.physics_polish_target_pde_mean)
    if getattr(args, "physics_polish_target_pde_max", None) is not None:
        Config.PHYSICS_POLISH_TARGET_PDE_MAX = float(args.physics_polish_target_pde_max)


def _run_train(args: argparse.Namespace) -> None:
    """Executa treino completo e imprime resumo final de métricas."""
    from src.pipeline import run_scientific_analysis

    _apply_train_overrides(args)
    result = run_scientific_analysis(
        output_dir=args.output_dir,
        report_path=args.report_path,
        results_dir=args.results_dir,
        interior_strategy=args.interior_strategy,
        verbose=not args.quiet,
    )
    summary = {
        "mae": result["analyzer"].metrics["MAE"],
        "rmse": result["analyzer"].metrics["RMSE"],
        "r2": result["analyzer"].metrics["R2"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    """Monta parser com subcomandos de treino, tuning e sincronização de config."""
    parser = argparse.ArgumentParser(description="PINN Thermal 2D CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Treina o modelo PINN.")
    train_parser.add_argument("--epochs-adam", type=int, default=None)
    train_parser.add_argument("--epochs-lbfgs", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--n-interior", type=int, default=None)
    train_parser.add_argument(
        "--interior-strategy",
        type=str,
        default="hybrid",
        choices=["fdm", "lhs", "sobol", "hybrid"],
    )
    train_parser.add_argument("--n-boundary", type=int, default=None)
    train_parser.add_argument("--n-data", type=int, default=None)
    train_parser.add_argument("--n-midline", type=int, default=None)
    train_parser.add_argument("--w-data", type=float, default=None)
    train_parser.add_argument("--fdm-nx", type=int, default=None)
    train_parser.add_argument("--fdm-ny", type=int, default=None)
    train_parser.add_argument("--domain-size", type=int, default=None)
    train_parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=["tanh", "adaptive_tanh", "silu"],
    )
    train_parser.add_argument(
        "--input-normalization",
        type=str,
        default=None,
        choices=["zero_one", "minus_one_one", "none"],
    )
    train_parser.add_argument(
        "--enable-hard-dirichlet-constraint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    train_parser.add_argument("--right-dirichlet-multiplier", type=float, default=None)
    train_parser.add_argument("--left-dirichlet-weight-multiplier", type=float, default=None)
    train_parser.add_argument("--right-dirichlet-loss-multiplier", type=float, default=None)
    train_parser.add_argument(
        "--auto-balance-dirichlet-sides",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    train_parser.add_argument("--dirichlet-side-target-ratio", type=float, default=None)
    train_parser.add_argument("--dirichlet-side-ratio-tolerance", type=float, default=None)
    train_parser.add_argument("--right-dirichlet-loss-adapt-rate", type=float, default=None)
    train_parser.add_argument("--left-dirichlet-loss-min-multiplier", type=float, default=None)
    train_parser.add_argument("--left-dirichlet-loss-max-multiplier", type=float, default=None)
    train_parser.add_argument("--right-dirichlet-loss-min-multiplier", type=float, default=None)
    train_parser.add_argument("--right-dirichlet-loss-max-multiplier", type=float, default=None)
    train_parser.add_argument("--curvature-reg-weight", type=float, default=None)
    train_parser.add_argument("--left-importance-fraction", type=float, default=None)
    train_parser.add_argument("--left-importance-band-fraction", type=float, default=None)
    train_parser.add_argument(
        "--enable-physics-polish",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    train_parser.add_argument("--physics-polish-iters", type=int, default=None)
    train_parser.add_argument("--physics-polish-target-pde-mean", type=float, default=None)
    train_parser.add_argument("--physics-polish-target-pde-max", type=float, default=None)
    train_parser.add_argument("--output-dir", type=str, default="runs/plots")
    train_parser.add_argument(
        "--report-path", type=str, default="runs/results/scientific_report.txt"
    )
    train_parser.add_argument("--results-dir", type=str, default="runs/results")
    train_parser.add_argument("--quiet", action="store_true")

    train_best_parser = subparsers.add_parser(
        "train-best",
        help="Treina aplicando automaticamente os melhores parâmetros do Optuna.",
    )
    train_best_parser.add_argument(
        "--best-path", type=str, default="runs/results/optuna_best.json"
    )
    train_best_parser.add_argument("--epochs-adam", type=int, default=None)
    train_best_parser.add_argument("--epochs-lbfgs", type=int, default=None)
    train_best_parser.add_argument("--lr", type=float, default=None)
    train_best_parser.add_argument("--n-interior", type=int, default=None)
    train_best_parser.add_argument(
        "--interior-strategy",
        type=str,
        default="hybrid",
        choices=["fdm", "lhs", "sobol", "hybrid"],
    )
    train_best_parser.add_argument("--n-boundary", type=int, default=None)
    train_best_parser.add_argument("--n-data", type=int, default=None)
    train_best_parser.add_argument("--n-midline", type=int, default=None)
    train_best_parser.add_argument("--w-data", type=float, default=None)
    train_best_parser.add_argument("--fdm-nx", type=int, default=None)
    train_best_parser.add_argument("--fdm-ny", type=int, default=None)
    train_best_parser.add_argument("--domain-size", type=int, default=None)
    train_best_parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=["tanh", "adaptive_tanh", "silu"],
    )
    train_best_parser.add_argument(
        "--input-normalization",
        type=str,
        default=None,
        choices=["zero_one", "minus_one_one", "none"],
    )
    train_best_parser.add_argument(
        "--enable-hard-dirichlet-constraint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    train_best_parser.add_argument("--right-dirichlet-multiplier", type=float, default=None)
    train_best_parser.add_argument(
        "--left-dirichlet-weight-multiplier", type=float, default=None
    )
    train_best_parser.add_argument(
        "--right-dirichlet-loss-multiplier", type=float, default=None
    )
    train_best_parser.add_argument(
        "--auto-balance-dirichlet-sides",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    train_best_parser.add_argument("--dirichlet-side-target-ratio", type=float, default=None)
    train_best_parser.add_argument(
        "--dirichlet-side-ratio-tolerance", type=float, default=None
    )
    train_best_parser.add_argument(
        "--right-dirichlet-loss-adapt-rate", type=float, default=None
    )
    train_best_parser.add_argument(
        "--left-dirichlet-loss-min-multiplier", type=float, default=None
    )
    train_best_parser.add_argument(
        "--left-dirichlet-loss-max-multiplier", type=float, default=None
    )
    train_best_parser.add_argument(
        "--right-dirichlet-loss-min-multiplier", type=float, default=None
    )
    train_best_parser.add_argument(
        "--right-dirichlet-loss-max-multiplier", type=float, default=None
    )
    train_best_parser.add_argument("--curvature-reg-weight", type=float, default=None)
    train_best_parser.add_argument("--left-importance-fraction", type=float, default=None)
    train_best_parser.add_argument(
        "--left-importance-band-fraction", type=float, default=None
    )
    train_best_parser.add_argument(
        "--enable-physics-polish",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    train_best_parser.add_argument("--physics-polish-iters", type=int, default=None)
    train_best_parser.add_argument(
        "--physics-polish-target-pde-mean", type=float, default=None
    )
    train_best_parser.add_argument("--physics-polish-target-pde-max", type=float, default=None)
    train_best_parser.add_argument("--output-dir", type=str, default="runs/plots")
    train_best_parser.add_argument(
        "--report-path", type=str, default="runs/results/scientific_report.txt"
    )
    train_best_parser.add_argument("--results-dir", type=str, default="runs/results")
    train_best_parser.add_argument("--quiet", action="store_true")

    tune_parser = subparsers.add_parser(
        "tune", help="Executa busca de hiperparâmetros com Optuna."
    )
    tune_parser.add_argument("--n-trials", type=int, default=30)
    tune_parser.add_argument("--timeout", type=int, default=None)
    tune_parser.add_argument("--epochs-adam", type=int, default=500)
    tune_parser.add_argument("--epochs-lbfgs", type=int, default=0)
    tune_parser.add_argument("--n-interior", type=int, default=2000)
    tune_parser.add_argument(
        "--interior-strategy",
        type=str,
        default="hybrid",
        choices=["fdm", "lhs", "sobol", "hybrid"],
    )
    tune_parser.add_argument("--n-boundary", type=int, default=200)
    tune_parser.add_argument("--n-data", type=int, default=2000)
    tune_parser.add_argument("--eval-domain-size", type=int, default=64)
    tune_parser.add_argument(
        "--results-path", type=str, default="runs/results/optuna_best.json"
    )
    tune_parser.add_argument(
        "--base-best-path", type=str, default="runs/results/optuna_best.json"
    )
    tune_parser.add_argument("--stability-lambda", type=float, default=3.50e-1)

    apply_parser = subparsers.add_parser(
        "apply-best-config",
        help="Aplica no config.py os melhores hiperparâmetros do arquivo do Optuna.",
    )
    apply_parser.add_argument(
        "--best-path", type=str, default="runs/results/optuna_best.json"
    )
    apply_parser.add_argument("--config-path", type=str, default=None)

    return parser


def main() -> None:
    """Ponto de entrada da CLI."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["train", *sys.argv[1:]])
    command = args.command

    if command in {"train", "train-best", "tune"}:
        from src.config import initialize_runtime

        initialize_runtime(emit_device_banner=True)

    if command == "apply-best-config":
        result = apply_best_to_config_file(
            best_path=args.best_path, config_path=args.config_path
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if command == "train":
        _run_train(args)
        return

    if command == "train-best":
        from src.config import Config
        from src.optuna_config import apply_optuna_best_from_file

        payload = apply_optuna_best_from_file(Config, args.best_path)
        print(
            json.dumps(
                {
                    "loaded_best_file": str(Path(args.best_path)),
                    "applied_best_params": payload.get("best_params", {}),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        _run_train(args)
        return

    if command == "tune":
        from src.tuning import optimize_hyperparameters

        best = optimize_hyperparameters(
            n_trials=args.n_trials,
            timeout=args.timeout,
            epochs_adam=args.epochs_adam,
            epochs_lbfgs=args.epochs_lbfgs,
            n_interior=args.n_interior,
            interior_strategy=args.interior_strategy,
            n_boundary=args.n_boundary,
            n_data=args.n_data,
            eval_domain_size=args.eval_domain_size,
            results_path=args.results_path,
            base_best_path=args.base_best_path,
            stability_lambda=args.stability_lambda,
        )
        print(json.dumps(best, indent=2, ensure_ascii=False))
        return

    parser.error(f"Comando desconhecido: {command}")


if __name__ == "__main__":
    main()
