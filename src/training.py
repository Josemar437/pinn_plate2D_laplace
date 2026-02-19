"""
Rotinas de treinamento híbrido e estabilização numérica para a PINN térmica 2D.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config import Config, logger
    from .losses import AdaptiveWeightScheduler, enhanced_loss_terms
    from .operators import pde_residual
    from .sampling import TrainingData
except ImportError:  
    from config import Config, logger
    from losses import AdaptiveWeightScheduler, enhanced_loss_terms
    from operators import pde_residual
    from sampling import TrainingData


def _pde_warmup_factor(epoch: int, warmup_epochs: int, start_factor: float) -> float:
    """
    Retorna fator de escalonamento para o termo PDE ao longo do treino.

    O fator evolui linearmente de start_factor até 1.0 durante
    warmup_epochs para reduzir conflito inicial entre termos de perda.
    """
    if warmup_epochs <= 0:
        return 1.0
    clipped_start = min(max(float(start_factor), 0.0), 1.0)
    if epoch >= int(warmup_epochs):
        return 1.0
    progress = float(epoch) / float(max(1, int(warmup_epochs)))
    return clipped_start + (1.0 - clipped_start) * progress


def _loss_term_grad_norms(
    net: nn.Module,
    loss_parts: Dict[str, torch.Tensor],
    *,
    keys: tuple[str, ...] = ("L_PDE", "L_D", "L_N"),
) -> Dict[str, float]:
    """Calcula norma L2 dos gradientes dos termos informados."""
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return {key: 0.0 for key in keys}

    grad_norms: Dict[str, float] = {}
    for key in keys:
        term = loss_parts.get(key)
        if term is None:
            grad_norms[key] = 0.0
            continue
        grads = torch.autograd.grad(
            term,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        sq_sum = torch.zeros((), device=term.device, dtype=term.dtype)
        for g in grads:
            if g is not None:
                sq_sum = sq_sum + torch.sum(g.detach() ** 2)
        grad_norms[key] = float(torch.sqrt(sq_sum + 1.0e-24).item())
    return grad_norms


def _sample_rar_candidates(
    n_candidates: int,
    *,
    corner_fraction: float,
    corner_band_fraction: float,
    left_fraction: float,
    left_band_fraction: float,
    right_fraction: float,
    right_band_fraction: float,
    device_: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Gera candidatos para RAR com mistura de pontos globais e focados em cantos.

    A região de canto é construída perto de x in {0, Lx} e
    y in {0, Ly} para reforçar singularidades geométricas.
    """
    n_candidates = max(1, int(n_candidates))
    corner_fraction = min(max(float(corner_fraction), 0.0), 1.0)
    left_fraction = min(max(float(left_fraction), 0.0), 1.0)
    right_fraction = min(max(float(right_fraction), 0.0), 1.0)
    corner_band_fraction = min(max(float(corner_band_fraction), 1.0e-4), 0.5)
    left_band_fraction = min(max(float(left_band_fraction), 1.0e-4), 1.0)
    right_band_fraction = min(max(float(right_band_fraction), 1.0e-4), 1.0)

    n_corner = int(round(n_candidates * corner_fraction))
    n_left = int(round(n_candidates * left_fraction))
    n_right = int(round(n_candidates * right_fraction))
    n_global = n_candidates - n_corner - n_left - n_right
    if n_global < 0:
        overflow = -n_global
        reduce_right = min(overflow, n_right)
        n_right -= reduce_right
        overflow -= reduce_right
        reduce_left = min(overflow, n_left)
        n_left -= reduce_left
        overflow -= reduce_left
        if overflow > 0:
            n_corner = max(0, n_corner - overflow)
        n_global = n_candidates - n_corner - n_left - n_right
    chunks: list[torch.Tensor] = []

    cpu_device = torch.device("cpu")

    if n_global > 0:
        global_pts = torch.rand(
            (n_global, 2), generator=generator, dtype=dtype, device=cpu_device
        )
        global_pts[:, 0] *= float(Config.LX)
        global_pts[:, 1] *= float(Config.LY)
        chunks.append(global_pts)

    if n_corner > 0:
        corner_pts = torch.rand(
            (n_corner, 2), generator=generator, dtype=dtype, device=cpu_device
        )
        dx = float(Config.LX) * corner_band_fraction
        dy = float(Config.LY) * corner_band_fraction
        corner_pts[:, 0] *= dx
        corner_pts[:, 1] *= dy

        choose_right = (
            torch.rand((n_corner, 1), generator=generator, dtype=dtype, device=cpu_device)
            > 0.5
        )
        choose_top = (
            torch.rand((n_corner, 1), generator=generator, dtype=dtype, device=cpu_device)
            > 0.5
        )
        corner_pts[:, 0:1] = torch.where(
            choose_right, float(Config.LX) - corner_pts[:, 0:1], corner_pts[:, 0:1]
        )
        corner_pts[:, 1:2] = torch.where(
            choose_top, float(Config.LY) - corner_pts[:, 1:2], corner_pts[:, 1:2]
        )
        chunks.append(corner_pts)

    if n_left > 0:
        left_pts = torch.rand(
            (n_left, 2), generator=generator, dtype=dtype, device=cpu_device
        )
        left_pts[:, 0] = float(Config.LX) * left_band_fraction * left_pts[:, 0]
        left_pts[:, 1] *= float(Config.LY)
        chunks.append(left_pts)

    if n_right > 0:
        right_pts = torch.rand(
            (n_right, 2), generator=generator, dtype=dtype, device=cpu_device
        )
        right_pts[:, 0] = float(Config.LX) - (
            float(Config.LX) * right_band_fraction * right_pts[:, 0]
        )
        right_pts[:, 1] *= float(Config.LY)
        chunks.append(right_pts)

    all_pts = torch.cat(chunks, dim=0)
    if all_pts.shape[0] <= 1:
        return all_pts.to(device_)

    perm_cpu = torch.randperm(int(all_pts.shape[0]), generator=generator)
    return all_pts[perm_cpu].to(device_)


def _rar_refresh_collocation(
    net: nn.Module,
    X_f: torch.Tensor,
    *,
    replace_fraction: float,
    candidate_multiplier: int,
    top_pool_multiplier: int,
    selection_power: float,
    corner_fraction: float,
    corner_band_fraction: float,
    left_fraction: float,
    left_band_fraction: float,
    right_fraction: float,
    right_band_fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int, float]:
    """
    Atualiza parte de X_f com pontos de maior residual PDE (RAR-lite).

    O ranking usa apenas magnitude do residual; por isso create_graph=False
    é suficiente e evita retenção de grafo de alta ordem desnecessário.
    """
    n_current = int(X_f.shape[0])
    if n_current <= 0:
        return X_f, 0, float("nan")

    replace_fraction = min(max(float(replace_fraction), 0.0), 1.0)
    n_replace = int(round(n_current * replace_fraction))
    if n_replace <= 0:
        return X_f, 0, float("nan")

    n_candidates = max(
        n_replace * 2,
        int(candidate_multiplier) * n_current,
    )
    candidates = _sample_rar_candidates(
        n_candidates,
        corner_fraction=corner_fraction,
        corner_band_fraction=corner_band_fraction,
        left_fraction=left_fraction,
        left_band_fraction=left_band_fraction,
        right_fraction=right_fraction,
        right_band_fraction=right_band_fraction,
        device_=X_f.device,
        dtype=X_f.dtype,
        generator=generator,
    )
    # RAR usa o residual apenas para ranquear pontos. Não há atualização de
    # parâmetros nesta etapa, então manter grafo de alta ordem só consome
    # memória/tempo sem benefício numérico.
    residual_abs = torch.abs(
        pde_residual(
            candidates,
            net,
            nondimensional=True,
            create_graph=False,
        )
    ).reshape(-1)
    if residual_abs.numel() == 0:
        return X_f, 0, float("nan")

    k = min(n_replace, int(residual_abs.numel()))
    top_pool_multiplier = max(1, int(top_pool_multiplier))
    pool_k = min(int(residual_abs.numel()), max(k, k * top_pool_multiplier))
    pool_idx = torch.topk(residual_abs, k=pool_k, largest=True).indices

    if pool_k > k:
        selection_power = max(float(selection_power), 1.0e-6)
        pool_vals_cpu = residual_abs[pool_idx].detach().cpu()
        weights_cpu = (pool_vals_cpu + 1.0e-12) ** selection_power
        probs_cpu = weights_cpu / torch.clamp(weights_cpu.sum(), min=1.0e-12)
        local_idx_cpu = torch.multinomial(
            probs_cpu,
            num_samples=k,
            replacement=False,
            generator=generator,
        )
        selected_idx = pool_idx[local_idx_cpu.to(pool_idx.device)]
    else:
        selected_idx = pool_idx

    selected_pts = candidates[selected_idx].detach()
    selected_residual_mean = float(residual_abs[selected_idx].mean().detach().item())

    replace_idx_cpu = torch.randperm(n_current, generator=generator)[:k]
    replace_idx = replace_idx_cpu.to(X_f.device)
    updated = X_f.clone()
    updated[replace_idx] = selected_pts
    return updated, int(k), selected_residual_mean


def train_pinn_enhanced(
    net: nn.Module,
    data: TrainingData,
    epochs_adam: int = 4000,
    epochs_lbfgs: int = 1500,
    lr: float = 1e-3,
    weight_params: Optional[Dict[str, Any]] = None,
    w_data: float | None = None,
    use_adaptive_weights: bool | None = None,
    use_scheduler: bool = True,
    verbose: bool = True,
) -> Dict[str, List[Any]]:
    """
    Treina a rede com perdas físicas e supervisão por dados FDM.

    O histórico retornado contém perdas, pesos adaptativos, taxa de
    aprendizado e norma de gradiente por época.
    """
    net.train()
    hist: Dict[str, List[Any]] = {
        "total_loss": [],
        "optimizer_phase": [],
        "adam_loss": [],
        "L_PDE": [],
        "L_D": [],
        "L_D_left": [],
        "L_D_right": [],
        "L_N": [],
        "L_data": [],
        "L_curv": [],
        "grad_L_PDE": [],
        "grad_L_D": [],
        "grad_L_N": [],
        "w_pde": [],
        "w_pde_effective": [],
        "w_dir": [],
        "w_neu": [],
        "w_data": [],
        "w_curv": [],
        "w_dir_left_loss": [],
        "w_dir_right_loss": [],
        "dirichlet_side_left_mae_nd": [],
        "dirichlet_side_right_mae_nd": [],
        "dirichlet_side_ratio_right_left": [],
        "dirichlet_side_ratio_control_right_left": [],
        "dirichlet_side_ntk_ratio_left_right": [],
        "lr": [],
        "grad_norm": [],
        "rar_points_replaced": [],
        "rar_selected_residual_mean": [],
        "rar_global_residual_rms": [],
        "lbfgs_loss": [],
        "lbfgs_pde_loss": [],
        "physics_polish_loss": [],
        "physics_polish_pde_loss": [],
        "physics_polish_pde_mean": [],
        "physics_polish_pde_max": [],
    }

    if w_data is None:
        w_data = float(Config.DEFAULT_W_DATA)
    left_dirichlet_weight_multiplier_base = float(
        getattr(Config, "LEFT_DIRICHLET_WEIGHT_MULTIPLIER", 1.0)
    )
    left_dirichlet_weight_multiplier = float(left_dirichlet_weight_multiplier_base)
    right_dirichlet_weight_multiplier_base = float(
        getattr(Config, "RIGHT_DIRICHLET_LOSS_MULTIPLIER", 1.0)
    )
    right_dirichlet_weight_multiplier = float(right_dirichlet_weight_multiplier_base)
    auto_balance_dirichlet_sides = bool(
        getattr(Config, "AUTO_BALANCE_DIRICHLET_SIDES", False)
    )
    hard_dirichlet_enabled = bool(
        getattr(Config, "ENABLE_HARD_DIRICHLET_CONSTRAINT", False)
    )
    if hard_dirichlet_enabled:
        # Com Dirichlet imposto por construção, o controle adaptativo de lados
        # deixa de ser necessário e pode introduzir ruído numérico.
        auto_balance_dirichlet_sides = False
    dirichlet_side_target_ratio = max(
        1.0e-6, float(getattr(Config, "DIRICHLET_SIDE_TARGET_RATIO", 1.0))
    )
    dirichlet_side_ratio_tolerance = max(
        0.0, float(getattr(Config, "DIRICHLET_SIDE_RATIO_TOLERANCE", 0.05))
    )
    dirichlet_side_adapt_rate = min(
        max(float(getattr(Config, "RIGHT_DIRICHLET_LOSS_ADAPT_RATE", 0.1)), 0.0),
        1.0,
    )
    enable_dirichlet_ntk_balance = bool(
        getattr(Config, "ENABLE_DIRICHLET_NTK_BALANCE", False)
    )
    if hard_dirichlet_enabled:
        enable_dirichlet_ntk_balance = False
    dirichlet_ntk_balance_frequency = max(
        1, int(getattr(Config, "DIRICHLET_NTK_BALANCE_FREQUENCY", 25))
    )
    dirichlet_ntk_blend = min(
        max(float(getattr(Config, "DIRICHLET_NTK_BLEND", 0.5)), 0.0),
        1.0,
    )
    left_dirichlet_min_multiplier = max(
        1.0e-6, float(getattr(Config, "LEFT_DIRICHLET_LOSS_MIN_MULTIPLIER", 0.5))
    )
    left_dirichlet_max_multiplier = max(
        left_dirichlet_min_multiplier,
        float(getattr(Config, "LEFT_DIRICHLET_LOSS_MAX_MULTIPLIER", 8.0)),
    )
    right_dirichlet_min_multiplier = max(
        1.0e-6, float(getattr(Config, "RIGHT_DIRICHLET_LOSS_MIN_MULTIPLIER", 0.5))
    )
    right_dirichlet_max_multiplier = max(
        right_dirichlet_min_multiplier,
        float(getattr(Config, "RIGHT_DIRICHLET_LOSS_MAX_MULTIPLIER", 8.0)),
    )
    curv_weight = float(getattr(Config, "CURVATURE_REG_WEIGHT", 0.0))
    if use_adaptive_weights is None:
        use_adaptive_weights = bool(Config.USE_ADAPTIVE_WEIGHTING)

    weight_params = weight_params or Config.DEFAULT_WEIGHT_PARAMS
    static_w_pde = float(weight_params.get("w_pde", 1.0))
    static_w_dir = float(weight_params.get("w_dir", 100.0))
    static_w_neu = float(weight_params.get("w_neu", 10.0))
    weight_scheduler = (
        AdaptiveWeightScheduler(**weight_params) if use_adaptive_weights else None
    )
    X_f = data.X_f
    pde_warmup_epochs = int(getattr(Config, "PDE_WARMUP_EPOCHS", 0))
    pde_warmup_start = float(getattr(Config, "PDE_WARMUP_START_FACTOR", 1.0))

    rar_enabled = bool(getattr(Config, "ENABLE_RAR", False))
    rar_start_epoch = int(getattr(Config, "RAR_START_EPOCH", 0))
    rar_frequency = max(1, int(getattr(Config, "RAR_FREQUENCY", 200)))
    rar_replace_fraction = float(getattr(Config, "RAR_REPLACE_FRACTION", 0.0))
    rar_candidate_multiplier = max(
        2, int(getattr(Config, "RAR_CANDIDATE_MULTIPLIER", 4))
    )
    rar_top_pool_multiplier = max(
        1, int(getattr(Config, "RAR_TOP_POOL_MULTIPLIER", 1))
    )
    rar_selection_power = float(getattr(Config, "RAR_SELECTION_POWER", 1.0))
    rar_corner_fraction = float(getattr(Config, "RAR_CORNER_FRACTION", 0.5))
    rar_corner_band_fraction = float(
        getattr(Config, "RAR_CORNER_BAND_FRACTION", 0.1)
    )
    rar_left_fraction = float(getattr(Config, "RAR_LEFT_FRACTION", 0.0))
    rar_left_band_fraction = float(
        getattr(Config, "RAR_LEFT_BAND_FRACTION", 0.1)
    )
    rar_right_fraction = float(getattr(Config, "RAR_RIGHT_FRACTION", 0.0))
    rar_right_band_fraction = float(
        getattr(Config, "RAR_RIGHT_BAND_FRACTION", 0.1)
    )
    rar_generator = torch.Generator(device="cpu")
    rar_generator.manual_seed(int(Config.SEED) + 12345)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler_mode = str(getattr(Config, "LR_SCHEDULER_MODE", "plateau")).strip().lower()
    valid_scheduler_modes = {"plateau", "cosine", "none"}
    if scheduler_mode not in valid_scheduler_modes:
        raise ValueError(
            f"LR_SCHEDULER_MODE inválido: {scheduler_mode!r}. "
            f"Use uma de {sorted(valid_scheduler_modes)}."
        )

    lr_scheduler_plateau: optim.lr_scheduler.ReduceLROnPlateau | None = None
    lr_scheduler_cosine: optim.lr_scheduler.CosineAnnealingLR | None = None
    if use_scheduler and scheduler_mode == "plateau":
        lr_scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=int(Config.LR_SCHED_PATIENCE),
            factor=float(Config.LR_SCHED_FACTOR),
            cooldown=int(Config.LR_SCHED_COOLDOWN),
            min_lr=float(Config.LR_SCHED_MIN_LR),
            threshold=5.00e-3,
            threshold_mode="rel",
        )
    elif use_scheduler and scheduler_mode == "cosine":
        cosine_t_max = max(1, int(getattr(Config, "LR_COSINE_T_MAX", epochs_adam)))
        cosine_min_lr = float(getattr(Config, "LR_COSINE_MIN_LR", Config.LR_SCHED_MIN_LR))
        lr_scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=cosine_min_lr,
        )

    logger.info("Iniciando Fase Adam (%d épocas)...", epochs_adam)
    logger.info("Scheduler LR ativo: %s", scheduler_mode if use_scheduler else "none")
    start_time = time.time()
    if weight_scheduler is None:
        current_w_pde, current_w_dir, current_w_neu = (
            static_w_pde,
            static_w_dir,
            static_w_neu,
        )
        logger.info("Modo de pesos fixos ativo (sem rebalanceamento adaptativo).")
    else:
        current_w_pde, current_w_dir, current_w_neu = weight_scheduler.get_weights()
    ema_loss_for_scheduler: float | None = None

    def _compute_weighted_losses_common(
        *,
        stage_w_pde: float,
        stage_w_dir: float,
        stage_w_neu: float,
        stage_w_data: float,
        stage_w_curv: float,
        create_graph: bool,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calcula perda total e componentes com pesos informados."""
        return enhanced_loss_terms(
            net=net,
            X_f=X_f,
            X_dir=data.X_dir,
            T_dir_target=data.T_dir_target,
            X_bottom=data.X_bottom,
            X_top=data.X_top,
            X_data=data.X_data,
            T_data_target=data.T_data_target,
            X_midline=data.X_midline,
            w_pde=stage_w_pde,
            w_dir=stage_w_dir,
            left_dirichlet_weight_multiplier=left_dirichlet_weight_multiplier,
            right_dirichlet_weight_multiplier=right_dirichlet_weight_multiplier,
            w_neu=stage_w_neu,
            w_data=stage_w_data,
            w_curv=stage_w_curv,
            create_graph=create_graph,
        )

    for epoch in range(int(epochs_adam)):
        optimizer.zero_grad(set_to_none=True)

        if weight_scheduler is not None:
            current_w_pde, current_w_dir, current_w_neu = weight_scheduler.get_weights()
        pde_factor = _pde_warmup_factor(epoch, pde_warmup_epochs, pde_warmup_start)
        effective_w_pde = float(current_w_pde) * float(pde_factor)
        loss_total, loss_parts = _compute_weighted_losses_common(
            stage_w_pde=effective_w_pde,
            stage_w_dir=current_w_dir,
            stage_w_neu=current_w_neu,
            stage_w_data=w_data,
            stage_w_curv=curv_weight,
            create_graph=True,
        )

        dir_left = max(float(loss_parts["L_D_left_mae_nd"].detach().item()), 1.0e-12)
        dir_right = max(float(loss_parts["L_D_right_mae_nd"].detach().item()), 1.0e-12)
        dirichlet_side_ratio = dir_right / dir_left
        dirichlet_side_ratio_control = float(dirichlet_side_ratio)
        dirichlet_side_ntk_ratio = float("nan")

        need_scheduler_grad = bool(
            weight_scheduler is not None and weight_scheduler.needs_gradient_stats()
        )
        need_dirichlet_ntk_grad = bool(
            auto_balance_dirichlet_sides
            and enable_dirichlet_ntk_balance
            and ((epoch + 1) % dirichlet_ntk_balance_frequency == 0)
        )
        grad_stats: Dict[str, float] | None = None
        if need_scheduler_grad or need_dirichlet_ntk_grad:
            grad_keys: list[str] = []
            if need_scheduler_grad:
                grad_keys.extend(["L_PDE", "L_D", "L_N"])
            if need_dirichlet_ntk_grad:
                grad_keys.extend(["L_D_left", "L_D_right"])
            grad_all = _loss_term_grad_norms(
                net,
                loss_parts,
                keys=tuple(dict.fromkeys(grad_keys)),
            )
            if need_scheduler_grad:
                grad_stats = {
                    "L_PDE": float(grad_all.get("L_PDE", 0.0)),
                    "L_D": float(grad_all.get("L_D", 0.0)),
                    "L_N": float(grad_all.get("L_N", 0.0)),
                }
            if need_dirichlet_ntk_grad:
                g_left = max(abs(float(grad_all.get("L_D_left", 0.0))), 1.0e-12)
                g_right = max(abs(float(grad_all.get("L_D_right", 0.0))), 1.0e-12)
                dirichlet_side_ntk_ratio = g_left / g_right
                if math.isfinite(dirichlet_side_ntk_ratio):
                    ratio_base = max(float(dirichlet_side_ratio), 1.0e-12)
                    ratio_ntk = max(float(dirichlet_side_ntk_ratio), 1.0e-12)
                    # Híbrido MAE+NTK para priorizar lado mais rígido no espaço de parâmetros.
                    dirichlet_side_ratio_control = (ratio_base ** (1.0 - dirichlet_ntk_blend)) * (
                        ratio_ntk**dirichlet_ntk_blend
                    )

        loss_total.backward()

        grad_norm = 0.0
        for param in net.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm**0.5

        torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=float(Config.GRAD_CLIP_NORM)
        )
        optimizer.step()

        if lr_scheduler_plateau is not None:
            current_loss = float(loss_total.detach().item())
            if ema_loss_for_scheduler is None:
                ema_loss_for_scheduler = current_loss
            else:
                beta = float(Config.LR_SCHED_EMA_BETA)
                ema_loss_for_scheduler = beta * ema_loss_for_scheduler + (1.0 - beta) * current_loss
            lr_scheduler_plateau.step(float(ema_loss_for_scheduler))
        elif lr_scheduler_cosine is not None:
            lr_scheduler_cosine.step()
        if weight_scheduler is not None:
            weight_scheduler.step(loss_parts, grad_stats=grad_stats)

        # Controle de assimetria por MAE adimensional (mais interpretável que MSE).
        if (
            auto_balance_dirichlet_sides
            and dirichlet_side_adapt_rate > 0.0
            and math.isfinite(dirichlet_side_ratio_control)
        ):
            ratio_vs_target = max(
                dirichlet_side_ratio_control
                / max(dirichlet_side_target_ratio, 1.0e-12),
                1.0e-6,
            )
            upper_band = 1.0 + dirichlet_side_ratio_tolerance
            lower_band = 1.0 / upper_band
            if ratio_vs_target > upper_band:
                target_right_mult = right_dirichlet_weight_multiplier * (
                    ratio_vs_target**0.5
                )
                target_right_mult = min(
                    max(target_right_mult, right_dirichlet_min_multiplier),
                    right_dirichlet_max_multiplier,
                )
                right_dirichlet_weight_multiplier = (
                    (1.0 - dirichlet_side_adapt_rate) * right_dirichlet_weight_multiplier
                    + dirichlet_side_adapt_rate * target_right_mult
                )
            elif ratio_vs_target < lower_band:
                target_left_mult = left_dirichlet_weight_multiplier * (
                    (1.0 / ratio_vs_target) ** 0.5
                )
                target_left_mult = min(
                    max(target_left_mult, left_dirichlet_min_multiplier),
                    left_dirichlet_max_multiplier,
                )
                left_dirichlet_weight_multiplier = (
                    (1.0 - dirichlet_side_adapt_rate) * left_dirichlet_weight_multiplier
                    + dirichlet_side_adapt_rate * target_left_mult
                )
            else:
                relax_rate = 0.25 * dirichlet_side_adapt_rate
                left_dirichlet_weight_multiplier = (
                    (1.0 - relax_rate) * left_dirichlet_weight_multiplier
                    + relax_rate * left_dirichlet_weight_multiplier_base
                )
                right_dirichlet_weight_multiplier = (
                    (1.0 - relax_rate) * right_dirichlet_weight_multiplier
                    + relax_rate * right_dirichlet_weight_multiplier_base
                )
            left_dirichlet_weight_multiplier = min(
                max(left_dirichlet_weight_multiplier, left_dirichlet_min_multiplier),
                left_dirichlet_max_multiplier,
            )
            right_dirichlet_weight_multiplier = min(
                max(right_dirichlet_weight_multiplier, right_dirichlet_min_multiplier),
                right_dirichlet_max_multiplier,
            )

        hist["adam_loss"].append(float(loss_total.detach().item()))
        hist["total_loss"].append(float(loss_total.detach().item()))
        hist["optimizer_phase"].append("adam")
        hist["L_PDE"].append(float(loss_parts["L_PDE"].detach().item()))
        hist["L_D"].append(float(loss_parts["L_D"].detach().item()))
        hist["L_D_left"].append(float(loss_parts["L_D_left"].detach().item()))
        hist["L_D_right"].append(float(loss_parts["L_D_right"].detach().item()))
        hist["L_N"].append(float(loss_parts["L_N"].detach().item()))
        hist["L_data"].append(float(loss_parts["L_data"].detach().item()))
        hist["L_curv"].append(float(loss_parts["L_curv"].detach().item()))
        hist["grad_L_PDE"].append(
            float(grad_stats["L_PDE"]) if grad_stats is not None else float("nan")
        )
        hist["grad_L_D"].append(
            float(grad_stats["L_D"]) if grad_stats is not None else float("nan")
        )
        hist["grad_L_N"].append(
            float(grad_stats["L_N"]) if grad_stats is not None else float("nan")
        )
        hist["w_pde"].append(float(current_w_pde))
        hist["w_pde_effective"].append(float(effective_w_pde))
        hist["w_dir"].append(float(current_w_dir))
        hist["w_neu"].append(float(current_w_neu))
        hist["w_data"].append(float(w_data))
        hist["w_curv"].append(float(curv_weight))
        hist["w_dir_left_loss"].append(float(left_dirichlet_weight_multiplier))
        hist["w_dir_right_loss"].append(float(right_dirichlet_weight_multiplier))
        hist["dirichlet_side_left_mae_nd"].append(float(dir_left))
        hist["dirichlet_side_right_mae_nd"].append(float(dir_right))
        hist["dirichlet_side_ratio_right_left"].append(float(dirichlet_side_ratio))
        hist["dirichlet_side_ratio_control_right_left"].append(
            float(dirichlet_side_ratio_control)
        )
        hist["dirichlet_side_ntk_ratio_left_right"].append(
            float(dirichlet_side_ntk_ratio)
        )
        hist["lr"].append(float(optimizer.param_groups[0]["lr"]))
        hist["grad_norm"].append(float(grad_norm))
        hist["rar_points_replaced"].append(0.0)
        hist["rar_selected_residual_mean"].append(float("nan"))
        hist["rar_global_residual_rms"].append(float("nan"))

        should_refresh_rar = (
            rar_enabled
            and (epoch + 1) >= rar_start_epoch
            and ((epoch + 1 - rar_start_epoch) % rar_frequency == 0)
        )
        if should_refresh_rar:
            X_f, n_replaced, selected_mean = _rar_refresh_collocation(
                net=net,
                X_f=X_f,
                replace_fraction=rar_replace_fraction,
                candidate_multiplier=rar_candidate_multiplier,
                top_pool_multiplier=rar_top_pool_multiplier,
                selection_power=rar_selection_power,
                corner_fraction=rar_corner_fraction,
                corner_band_fraction=rar_corner_band_fraction,
                left_fraction=rar_left_fraction,
                left_band_fraction=rar_left_band_fraction,
                right_fraction=rar_right_fraction,
                right_band_fraction=rar_right_band_fraction,
                generator=rar_generator,
            )
            hist["rar_points_replaced"][-1] = float(n_replaced)
            hist["rar_selected_residual_mean"][-1] = float(selected_mean)
            hist["rar_global_residual_rms"][-1] = float(
                torch.sqrt(torch.clamp(loss_parts["L_PDE"].detach(), min=0.0)).item()
            )
            if verbose:
                logger.info(
                    "RAR @ epoch %04d | pontos substituídos: %d | residual médio selecionado: %.2e",
                    epoch + 1,
                    n_replaced,
                    selected_mean,
                )

        if verbose and epoch % 500 == 0:
            logger.info(
                "Epoch %04d | Total: %.2e | PDE: %.2e | Data: %.2e | LR: %.2e | w_pde_eff: %.2e | Dir R/L: %.3f | w_dir_left: %.2f | w_dir_right: %.2f",
                epoch,
                loss_total.item(),
                loss_parts["L_PDE"].item(),
                loss_parts["L_data"].item(),
                optimizer.param_groups[0]["lr"],
                effective_w_pde,
                dirichlet_side_ratio,
                left_dirichlet_weight_multiplier,
                right_dirichlet_weight_multiplier,
            )

    if weight_scheduler is not None:
        current_w_pde, current_w_dir, current_w_neu = weight_scheduler.get_weights()

    line_search = getattr(Config, "LBFGS_LINE_SEARCH_FN", None)
    line_search_fn = str(line_search) if line_search is not None else None

    def _create_lbfgs(max_iter: int) -> optim.LBFGS:
        """Cria instância configurada de L-BFGS para uma fase de refinamento."""
        return optim.LBFGS(
            net.parameters(),
            lr=float(getattr(Config, "LBFGS_LR", 1.0)),
            max_iter=int(max_iter),
            history_size=int(getattr(Config, "LBFGS_HISTORY_SIZE", 50)),
            line_search_fn=line_search_fn,
        )

    def _compute_weighted_losses(
        stage_w_pde: float,
        stage_w_dir: float,
        stage_w_neu: float,
        stage_w_data: float,
        stage_w_curv: float,
        *,
        create_graph: bool,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Backward-compat helper para fases de refinamento (L-BFGS/polish).

        Delega ao mesmo caminho usado na fase Adam para manter consistência e
        evitar duplicação de argumentos/contrato da enhanced_loss_terms.
        """
        return _compute_weighted_losses_common(
            stage_w_pde=stage_w_pde,
            stage_w_dir=stage_w_dir,
            stage_w_neu=stage_w_neu,
            stage_w_data=stage_w_data,
            stage_w_curv=stage_w_curv,
            create_graph=create_graph,
        )

    def _evaluate_weighted_losses(
        stage_w_pde: float,
        stage_w_dir: float,
        stage_w_neu: float,
        stage_w_data: float,
        stage_w_curv: float,
    ) -> tuple[float, Dict[str, float]]:
        l_tot_eval, loss_parts_eval = _compute_weighted_losses(
            stage_w_pde=stage_w_pde,
            stage_w_dir=stage_w_dir,
            stage_w_neu=stage_w_neu,
            stage_w_data=stage_w_data,
            stage_w_curv=stage_w_curv,
            create_graph=False,
        )
        total_value = float(l_tot_eval.detach().item())
        scalar_parts = {
            "L_PDE": float(loss_parts_eval["L_PDE"].detach().item()),
            "L_D": float(loss_parts_eval["L_D"].detach().item()),
            "L_D_left": float(loss_parts_eval["L_D_left"].detach().item()),
            "L_D_right": float(loss_parts_eval["L_D_right"].detach().item()),
            "L_D_left_mae_nd": float(loss_parts_eval["L_D_left_mae_nd"].detach().item()),
            "L_D_right_mae_nd": float(loss_parts_eval["L_D_right_mae_nd"].detach().item()),
            "L_N": float(loss_parts_eval["L_N"].detach().item()),
            "L_data": float(loss_parts_eval["L_data"].detach().item()),
            "L_curv": float(loss_parts_eval["L_curv"].detach().item()),
        }
        return total_value, scalar_parts

    def _evaluate_collocation_pde_abs_stats() -> tuple[float, float]:
        pde_abs = torch.abs(
            pde_residual(
                X_f,
                net,
                nondimensional=False,
                create_graph=False,
            )
        )
        return float(torch.mean(pde_abs.detach()).item()), float(torch.max(pde_abs.detach()).item())

    def _append_phase_history(
        *,
        phase_name: str,
        total_loss_value: float,
        parts: Dict[str, float],
        dirichlet_ratio: float,
        dirichlet_ratio_control: float | None = None,
        dirichlet_ntk_ratio: float | None = None,
    ) -> None:
        """
        Registra histórico de uma fase de refinamento sem duplicar lógica.

        Isso garante consistência entre fases (lbfgs e physics_polish) e
        reduz risco de divergência silenciosa em métricas registradas.
        """
        ratio_control = (
            float(dirichlet_ratio)
            if dirichlet_ratio_control is None
            else float(dirichlet_ratio_control)
        )
        ntk_ratio = float("nan") if dirichlet_ntk_ratio is None else float(dirichlet_ntk_ratio)
        hist["total_loss"].append(float(total_loss_value))
        hist["optimizer_phase"].append(str(phase_name))
        hist["L_PDE"].append(float(parts["L_PDE"]))
        hist["L_D"].append(float(parts["L_D"]))
        hist["L_D_left"].append(float(parts["L_D_left"]))
        hist["L_D_right"].append(float(parts["L_D_right"]))
        hist["L_N"].append(float(parts["L_N"]))
        hist["L_data"].append(float(parts["L_data"]))
        hist["L_curv"].append(float(parts["L_curv"]))
        hist["dirichlet_side_left_mae_nd"].append(float(parts["L_D_left_mae_nd"]))
        hist["dirichlet_side_right_mae_nd"].append(float(parts["L_D_right_mae_nd"]))
        hist["dirichlet_side_ratio_right_left"].append(float(dirichlet_ratio))
        hist["dirichlet_side_ratio_control_right_left"].append(ratio_control)
        hist["dirichlet_side_ntk_ratio_left_right"].append(ntk_ratio)

    def _make_lbfgs_closure(
        optimizer_obj: optim.Optimizer,
        stage_w_pde: float,
        stage_w_dir: float,
        stage_w_neu: float,
        stage_w_data: float,
        stage_w_curv: float,
    ):
        """Monta closure de L-BFGS para o conjunto de pesos informado."""

        def closure() -> torch.Tensor:
            optimizer_obj.zero_grad()
            l_tot, _ = _compute_weighted_losses(
                stage_w_pde=stage_w_pde,
                stage_w_dir=stage_w_dir,
                stage_w_neu=stage_w_neu,
                stage_w_data=stage_w_data,
                stage_w_curv=stage_w_curv,
                create_graph=True,
            )
            l_tot.backward()
            return l_tot

        return closure

    if int(epochs_lbfgs) > 0:
        logger.info("Iniciando Refinamento L-BFGS (%d iterações)...", epochs_lbfgs)
        lbfgs = _create_lbfgs(max_iter=int(epochs_lbfgs))
        lbfgs.step(
            _make_lbfgs_closure(
                lbfgs,
                current_w_pde,
                current_w_dir,
                current_w_neu,
                float(w_data),
                curv_weight,
            )
        )
        lbfgs_total, lbfgs_parts = _evaluate_weighted_losses(
            current_w_pde, current_w_dir, current_w_neu, float(w_data), curv_weight
        )
        lbfgs_pde = float(lbfgs_parts["L_PDE"])
        lbfgs_ratio = lbfgs_parts["L_D_right_mae_nd"] / max(
            lbfgs_parts["L_D_left_mae_nd"], 1.0e-12
        )
        hist["lbfgs_loss"].append(lbfgs_total)
        hist["lbfgs_pde_loss"].append(lbfgs_pde)
        _append_phase_history(
            phase_name="lbfgs",
            total_loss_value=lbfgs_total,
            parts=lbfgs_parts,
            dirichlet_ratio=lbfgs_ratio,
            dirichlet_ratio_control=lbfgs_ratio,
        )

    physics_polish_enabled = bool(getattr(Config, "ENABLE_PHYSICS_POLISH", False))
    physics_polish_iters = int(getattr(Config, "PHYSICS_POLISH_ITERS", 0))
    physics_polish_max_rounds = max(
        1, int(getattr(Config, "PHYSICS_POLISH_MAX_ROUNDS", 1))
    )
    physics_polish_target_pde_mean = float(
        getattr(Config, "PHYSICS_POLISH_TARGET_PDE_MEAN", -1.0)
    )
    physics_polish_target_pde_max = float(
        getattr(Config, "PHYSICS_POLISH_TARGET_PDE_MAX", -1.0)
    )
    if physics_polish_enabled and physics_polish_iters > 0:
        polish_w_pde = float(current_w_pde) * float(
            getattr(Config, "PHYSICS_POLISH_PDE_FACTOR", 1.0)
        )
        polish_w_dir = float(current_w_dir) * float(
            getattr(Config, "PHYSICS_POLISH_DIR_FACTOR", 1.0)
        )
        polish_w_neu = float(current_w_neu) * float(
            getattr(Config, "PHYSICS_POLISH_NEU_FACTOR", 1.0)
        )
        polish_w_data = float(w_data) * float(
            getattr(Config, "PHYSICS_POLISH_DATA_FACTOR", 1.0)
        )
        polish_w_curv = float(curv_weight) * float(
            getattr(Config, "PHYSICS_POLISH_CURV_FACTOR", 1.0)
        )
        logger.info(
            "Iniciando Physics-Polish L-BFGS (%d rodada(s) x %d iterações, w_pde=%.2e, w_dir=%.2e, w_neu=%.2e, w_data=%.2e, w_curv=%.2e, alvos mean|res|=%.2e e max|res|=%.2e)...",
            physics_polish_max_rounds,
            physics_polish_iters,
            polish_w_pde,
            polish_w_dir,
            polish_w_neu,
            polish_w_data,
            polish_w_curv,
            physics_polish_target_pde_mean,
            physics_polish_target_pde_max,
        )
        for polish_round in range(physics_polish_max_rounds):
            lbfgs_polish = _create_lbfgs(max_iter=physics_polish_iters)
            lbfgs_polish.step(
                _make_lbfgs_closure(
                    lbfgs_polish,
                    polish_w_pde,
                    polish_w_dir,
                    polish_w_neu,
                    polish_w_data,
                    polish_w_curv,
                )
            )
            polish_total, polish_parts = _evaluate_weighted_losses(
                polish_w_pde,
                polish_w_dir,
                polish_w_neu,
                polish_w_data,
                polish_w_curv,
            )
            polish_pde = float(polish_parts["L_PDE"])
            polish_ratio = polish_parts["L_D_right_mae_nd"] / max(
                polish_parts["L_D_left_mae_nd"], 1.0e-12
            )
            polish_pde_mean, polish_pde_max = _evaluate_collocation_pde_abs_stats()
            hist["physics_polish_loss"].append(polish_total)
            hist["physics_polish_pde_loss"].append(polish_pde)
            hist["physics_polish_pde_mean"].append(polish_pde_mean)
            hist["physics_polish_pde_max"].append(polish_pde_max)
            _append_phase_history(
                phase_name="physics_polish",
                total_loss_value=polish_total,
                parts=polish_parts,
                dirichlet_ratio=polish_ratio,
                dirichlet_ratio_control=polish_ratio,
            )
            logger.info(
                "Physics-Polish %d/%d | total: %.2e | L_PDE: %.2e | mean|res|_colloc: %.2e | max|res|_colloc: %.2e",
                polish_round + 1,
                physics_polish_max_rounds,
                polish_total,
                polish_pde,
                polish_pde_mean,
                polish_pde_max,
            )
            mean_target_hit = (
                physics_polish_target_pde_mean > 0.0
                and polish_pde_mean <= physics_polish_target_pde_mean
            )
            max_target_hit = (
                physics_polish_target_pde_max > 0.0
                and polish_pde_max <= physics_polish_target_pde_max
            )
            if mean_target_hit or max_target_hit:
                logger.info(
                    "Meta de residual PDE atingida no polimento (mean: %.2e <= %.2e | max: %.2e <= %.2e).",
                    polish_pde_mean,
                    physics_polish_target_pde_mean,
                    polish_pde_max,
                    physics_polish_target_pde_max,
                )
                break

    logger.info("Treinamento concluído em %.1fs", time.time() - start_time)
    return hist

