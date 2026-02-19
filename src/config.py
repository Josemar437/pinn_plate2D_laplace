"""
Parâmetros globais de física, treino e execução do experimento PINN térmico 2D.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


class Config:
    """Centraliza parâmetros físicos, numéricos e de treinamento do PINN."""

    # Precisão numérica e reprodutibilidade.
    USE_DOUBLE: bool = True
    SEED: int = 42

    # Geometria física e temperaturas de contorno.
    T_LEFT: float = 200.0
    T_RIGHT: float = 100.0
    LX: float = 1.0
    LY: float = 1.0

    # Parâmetros do solver FDM de referência.
    FDM_NX: int = 121
    FDM_NY: int = 121
    FDM_TOL: float = 1e-8
    FDM_MAX_ITER: int = 100000
    FDM_OMEGA: float = 1.7

    # Hiperparâmetros de treinamento.
    DEFAULT_EPOCHS_ADAM: int = 4000
    # L-BFGS agressivo para polimento final após Adam+RAR.
    DEFAULT_EPOCHS_LBFGS: int = 500
    DEFAULT_LR: float = 1.59e-3

    # Tamanhos dos conjuntos de dados.
    DEFAULT_N_INTERIOR: int = 1024
    DEFAULT_N_BOUNDARY: int = 240
    DEFAULT_N_DATA: int = 3000
    DEFAULT_N_MIDLINE: int = 128
    DEFAULT_INTERIOR_STRATEGY: str = "hybrid"
    ENABLE_LEFT_IMPORTANCE_SAMPLING: bool = True
    LEFT_IMPORTANCE_FRACTION: float = 2.50e-1
    LEFT_IMPORTANCE_BAND_FRACTION: float = 1.20e-1
    ENABLE_RIGHT_IMPORTANCE_SAMPLING: bool = True
    RIGHT_IMPORTANCE_FRACTION: float = 3.00e-1
    RIGHT_IMPORTANCE_BAND_FRACTION: float = 1.20e-1
    RIGHT_DIRICHLET_MULTIPLIER: float = 2.00e0
    COLLOCATION_AUDIT_ENABLED: bool = True
    COLLOCATION_AUDIT_BAND_FRACTION: float = 1.00e-1

    # Parâmetros de visualização.
    DOMAIN_SIZE: int = 150
    DPI: int = 600

    # Arquitetura padrão da rede.
    DEFAULT_LAYERS: list[int] = [2, 128, 128, 128, 128, 128, 128, 128, 128, 1]
    DEFAULT_ACTIVATION: str = "adaptive_tanh"
    # Faixa [-1, 1] reduz saturação de tanh e melhora condicionamento numérico.
    INPUT_NORMALIZATION: str = "minus_one_one"
    # Impõe Dirichlet em x=0 e x=Lx por construção arquitetural:
    # T = T_lin(x) + x*(Lx-x)*N(x,y), de modo que T(0,y)=T_left e T(Lx,y)=T_right.
    ENABLE_HARD_DIRICHLET_CONSTRAINT: bool = False
    ADAPTIVE_TANH_INIT: float = 1.00e0
    ADAPTIVE_TANH_MIN_SLOPE: float = 5.00e-2

    # Pesos dos termos de perda.
    DEFAULT_W_DATA: float = 1.25e0
    LEFT_DIRICHLET_WEIGHT_MULTIPLIER: float = 2.50e0
    RIGHT_DIRICHLET_LOSS_MULTIPLIER: float = 3.00e0
    AUTO_BALANCE_DIRICHLET_SIDES: bool = True
    DIRICHLET_SIDE_TARGET_RATIO: float = 1.00e0
    DIRICHLET_SIDE_RATIO_TOLERANCE: float = 5.00e-2
    # Taxa maior acelera correção de assimetria entre bordas esquerda/direita.
    RIGHT_DIRICHLET_LOSS_ADAPT_RATE: float = 2.00e-1
    # Balanceamento híbrido: razão de erro (MAE) + proxy NTK por norma de gradiente.
    ENABLE_DIRICHLET_NTK_BALANCE: bool = True
    DIRICHLET_NTK_BALANCE_FREQUENCY: int = 25
    DIRICHLET_NTK_BLEND: float = 5.00e-1
    LEFT_DIRICHLET_LOSS_MIN_MULTIPLIER: float = 5.00e-1
    LEFT_DIRICHLET_LOSS_MAX_MULTIPLIER: float = 8.00e0
    RIGHT_DIRICHLET_LOSS_MIN_MULTIPLIER: float = 5.00e-1
    RIGHT_DIRICHLET_LOSS_MAX_MULTIPLIER: float = 8.00e0
    CURVATURE_REG_WEIGHT: float = 5.00e-2
    HYPERPARAM_SIG_FIGS: int = 3
    DEFAULT_WEIGHT_PARAMS: Dict[str, Any] = {
        "w_pde": 1.00e0,
        "w_dir": 2.17e1,
        "w_neu": 7.43e2,
        "balance_frequency": 60,
        "alpha": 9.93e-1,
        "max_weight_ratio": 4.24e2,
        "min_balance_step": 750,
        "window": 30,
        "max_weight_change": 1.93e-1,
        "use_median": True,
        "strategy": "grad_norm_balance",
    }
    USE_ADAPTIVE_WEIGHTING: bool = True

    LR_SCHED_PATIENCE: int = 300
    LR_SCHED_FACTOR: float = 6.00e-1
    LR_SCHED_COOLDOWN: int = 100
    LR_SCHED_MIN_LR: float = 1.00e-6
    LR_SCHED_EMA_BETA: float = 9.90e-1
    LR_SCHEDULER_MODE: str = "cosine"
    LR_COSINE_T_MAX: int = 4000
    LR_COSINE_MIN_LR: float = 1.00e-6
    GRAD_CLIP_NORM: float = 8.00e-1
    LBFGS_LR: float = 1.00e0
    LBFGS_HISTORY_SIZE: int = 100
    LBFGS_LINE_SEARCH_FN: str | None = "strong_wolfe"

    # Estabilização inicial do termo físico (PDE warmup).
    PDE_WARMUP_EPOCHS: int = 1000
    PDE_WARMUP_START_FACTOR: float = 5.00e-1

    # Penalização extra para cauda do residual PDE (reduz picos locais).
    PDE_TOPK_FRACTION: float = 5.00e-2
    PDE_TOPK_WEIGHT: float = 2.50e-1

    # Refinamento adaptativo de pontos de colocação (RAR-lite).
    # Mantido ativo por padrão para concentrar pontos onde o residual físico é maior.
    ENABLE_RAR: bool = True
    RAR_START_EPOCH: int = 1000
    RAR_FREQUENCY: int = 200
    RAR_REPLACE_FRACTION: float = 1.00e-1
    RAR_CANDIDATE_MULTIPLIER: int = 6
    RAR_TOP_POOL_MULTIPLIER: int = 4
    RAR_SELECTION_POWER: float = 5.00e-1
    # Reforça pontos de colocação em vértices (singularidades geométricas).
    RAR_CORNER_FRACTION: float = 7.00e-1
    RAR_CORNER_BAND_FRACTION: float = 1.00e-1
    RAR_LEFT_FRACTION: float = 2.50e-1
    RAR_LEFT_BAND_FRACTION: float = 1.00e-1
    RAR_RIGHT_FRACTION: float = 2.50e-1
    RAR_RIGHT_BAND_FRACTION: float = 1.00e-1

    # Polimento final com foco na física após Adam/L-BFGS padrão.
    ENABLE_PHYSICS_POLISH: bool = True
    PHYSICS_POLISH_ITERS: int = 60
    PHYSICS_POLISH_MAX_ROUNDS: int = 1
    PHYSICS_POLISH_TARGET_PDE_MEAN: float = 1.00e-2
    PHYSICS_POLISH_TARGET_PDE_MAX: float = 5.00e-2
    PHYSICS_POLISH_PDE_FACTOR: float = 2.00e1
    PHYSICS_POLISH_DIR_FACTOR: float = 1.10e0
    PHYSICS_POLISH_NEU_FACTOR: float = 9.00e-1
    PHYSICS_POLISH_DATA_FACTOR: float = 2.00e-1
    PHYSICS_POLISH_CURV_FACTOR: float = 6.00e0

    # Critérios mínimos para confiabilidade de resultados (bloqueio de publicação).
    PDE_RESIDUAL_ND_MEAN_THRESHOLD: float = 5.00e-3
    PDE_RESIDUAL_ND_MAX_THRESHOLD: float = 5.00e-2
    RELATIVE_L2_THRESHOLD: float = 1.00e-3
    BOUNDARY_BC_ND_THRESHOLD: float = 1.00e-2
    # Mantido para compatibilidade com versões anteriores.
    BOUNDARY_BC_THRESHOLD: float = BOUNDARY_BC_ND_THRESHOLD
    FALSE_CONV_LOSS_THRESHOLD: float = 1.00e-4
    FALSE_CONV_PDE_ND_THRESHOLD: float = 1.00e-2
    FALSE_CONV_PDE_GROWTH_THRESHOLD: float = 1.00e-1

    # Pesos do objetivo adimensional no tuning (Optuna).
    TUNING_W_REL_L2: float = 1.00e0
    TUNING_W_PDE_RESIDUAL: float = 1.00e0
    TUNING_W_BOUNDARY_BC: float = 1.00e0

    RUNS_DIR: Path = Path("runs")
    PLOTS_DIR: Path = RUNS_DIR / "plots"
    LOGS_DIR: Path = RUNS_DIR / "logs"
    RESULTS_DIR: Path = RUNS_DIR / "results"
    LOG_FILE: Path = LOGS_DIR / "pinn_thermal.log"

    @classmethod
    def setup_torch(cls) -> None:
        """Configura precisão, sementes e backend do PyTorch."""
        dtype = torch.float64 if cls.USE_DOUBLE else torch.float32
        torch.set_default_dtype(dtype)
        torch.manual_seed(cls.SEED)
        np.random.seed(cls.SEED)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False


def setup_logging(
    level: int = logging.INFO, log_file: str | Path | None = None
) -> logging.Logger:
    """Cria logger único do projeto com formatação padronizada."""
    resolved_log = Path(log_file) if log_file is not None else Config.LOG_FILE
    resolved_log.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(resolved_log, encoding="utf-8")
    file_handler.setFormatter(formatter)

    project_logger = logging.getLogger("pinn_thermal")
    project_logger.setLevel(level)
    project_logger.handlers.clear()
    project_logger.addHandler(stream_handler)
    project_logger.addHandler(file_handler)
    return project_logger


def physical_scales() -> tuple[float, float]:
    """
    Retorna escalas físicas globais do problema.

    Centralizar essas escalas evita deriva entre módulos (loss, métricas e
    operadores), mantendo a mesma não-dimensionalização em todo o pipeline.
    """
    temp_scale = max(abs(float(Config.T_LEFT) - float(Config.T_RIGHT)), 1.0)
    length_scale = max(float(Config.LX), float(Config.LY), 1.0e-12)
    return temp_scale, length_scale


def physical_scales_tensor(
    *, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Versão tensorial de :func:physical_scales para uso em autograd."""
    temp_scale_value, length_scale_value = physical_scales()
    temp_scale = torch.tensor(temp_scale_value, device=device, dtype=dtype)
    length_scale = torch.tensor(length_scale_value, device=device, dtype=dtype)
    return temp_scale, length_scale


def residual_nondim_scale(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Retorna fator L_ref^2 / DeltaT_ref do residual não-dimensional.

    O residual de Laplace tem unidade K / L^2; multiplicar por este fator o
    torna adimensional e comparável entre malhas e escalas térmicas distintas.
    """
    temp_scale, length_scale = physical_scales()
    scale_value = (length_scale**2) / max(temp_scale, 1.0e-20)
    return torch.tensor(scale_value, device=device, dtype=dtype)


def _fdm_cache_signature() -> tuple[float, float, float, float, float, int, float]:
    """
    Retorna assinatura estável dos parâmetros físicos/numéricos do FDM.

    Centralizar a assinatura elimina duplicação entre módulos de amostragem e
    análise, reduzindo risco de divergência de cache entre referências FDM.
    """
    return (
        float(Config.LX),
        float(Config.LY),
        float(Config.T_LEFT),
        float(Config.T_RIGHT),
        float(Config.FDM_TOL),
        int(Config.FDM_MAX_ITER),
        float(Config.FDM_OMEGA),
    )


def build_fdm_cache_key(
    *,
    nx: int,
    ny: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Monta chave de cache FDM a partir da assinatura centralizada."""
    return (
        int(nx),
        int(ny),
        *_fdm_cache_signature(),
        str(dtype),
        str(device),
    )


def describe_device(active_device: torch.device | None = None) -> str:
    """Retorna descrição textual do dispositivo ativo, sem efeitos colaterais."""
    target_device = active_device if active_device is not None else get_device()
    if target_device.type == "cuda" and torch.cuda.is_available():
        return f"CUDA Ativado: {torch.cuda.get_device_name()}"
    return "CPU Ativado"


_RUNTIME_INITIALIZED: bool = False


def initialize_runtime(
    *,
    log_level: int = logging.INFO,
    log_file: str | Path | None = None,
    emit_device_banner: bool = True,
    force: bool = False,
) -> tuple[logging.Logger, torch.device]:
    """
    Inicializa runtime científico de forma explícita.

    Essa rotina substitui side effects no import. Ela configura dtype/seeds e
    logging uma única vez por processo, preservando reprodutibilidade e evitando
    trabalho inesperado ao importar módulos.
    """
    global _RUNTIME_INITIALIZED
    global logger
    global device
    if _RUNTIME_INITIALIZED and not force:
        return logger, device

   
    Config.setup_torch()
    logger = setup_logging(level=log_level, log_file=log_file)
    device = get_device()
    if emit_device_banner:
        print(describe_device(device))
    _RUNTIME_INITIALIZED = True
    return logger, device


def runtime_initialized() -> bool:
    """Indica se initialize_runtime já foi executado no processo atual."""
    return bool(_RUNTIME_INITIALIZED)


def _build_default_logger() -> logging.Logger:
    """Cria logger seguro para import sem inicialização explícita."""
    project_logger = logging.getLogger("pinn_thermal")
    if not project_logger.handlers:
        project_logger.addHandler(logging.NullHandler())
    return project_logger


def get_device() -> torch.device:
    """Seleciona o dispositivo disponível para treino e inferência."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


logger = _build_default_logger()
device = get_device()
