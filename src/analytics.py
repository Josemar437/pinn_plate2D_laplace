"""
Métricas de desempenho do PINN e validação contra solução de referência por FDM.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

try:
    from .config import (
        build_fdm_cache_key,
        Config,
        device,
        logger,
        physical_scales,
        residual_nondim_scale,
    )
    from .fdm import generate_reference_solution
    from .operators import dT_dy_on, pde_residual
except ImportError:  # pragma: no cover - compatibilidade com execução via PYTHONPATH=src
    from config import (
        build_fdm_cache_key,
        Config,
        device,
        logger,
        physical_scales,
        residual_nondim_scale,
    )
    from fdm import generate_reference_solution
    from operators import dT_dy_on, pde_residual


def _physical_scales() -> tuple[float, float]:
    """
    Escalas físicas para métricas adimensionais de erro de contorno.

    - temp_scale: amplitude térmica de referência |T_left - T_right|.
    - length_scale: comprimento característico max(Lx, Ly).
    """
    return physical_scales()


def pde_residual_l2_reference(
    *,
    domain_size: int,
    temp_scale: float | None = None,
    length_scale: float | None = None,
) -> float:
    """
    Referência dimensional para normalizar ||residual||_2.

    A referência considera unidade de residual DeltaT/L^2 e escala com
    sqrt(N) para remover dependência espúria da resolução da malha.
    """
    default_temp, default_len = _physical_scales()
    resolved_temp = float(temp_scale) if temp_scale is not None else default_temp
    resolved_len = float(length_scale) if length_scale is not None else default_len
    n_points = max(1, int(domain_size) * int(domain_size))
    residual_unit = resolved_temp / max(resolved_len**2, 1.0e-20)
    return max(np.sqrt(float(n_points)) * residual_unit, 1.0e-20)


def pde_residual_l2_normalized(
    residual_l2: float,
    *,
    domain_size: int,
    temp_scale: float | None = None,
    length_scale: float | None = None,
) -> float:
    """Retorna residual_L2 / residual_reference em forma adimensional."""
    reference = pde_residual_l2_reference(
        domain_size=domain_size,
        temp_scale=temp_scale,
        length_scale=length_scale,
    )
    return float(residual_l2) / reference


def stable_mape_percent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    temp_scale: float,
    floor_fraction: float = 1.0e-3,
) -> float:
    """
    MAPE robusto com piso relativo à escala térmica.

    Para temperaturas próximas de zero, o MAPE clássico explode por divisão por
    números muito pequenos. O piso floor_fraction * temp_scale preserva
    interpretação física do erro percentual e evita artefatos numéricos.
    """
    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    denom_floor = max(float(floor_fraction) * float(temp_scale), 1.0e-20)
    den = np.maximum(np.abs(np.asarray(y_true)), denom_floor)
    return float(np.mean(abs_err / den) * 100.0)


class EnhancedThermalAnalyzer:
    """Avalia o campo previsto pela rede frente ao campo numérico de referência."""

    _fdm_cache: Dict[tuple, tuple[np.ndarray, int]] = {}

    def __init__(
        self,
        net: nn.Module,
        domain_size: int = 100,
        device_: torch.device = device,
        residual_chunk_size: int = 8192,
    ) -> None:
        """Inicializa malha de avaliação e executa análise imediata do modelo."""
        self.net = net.eval()
        self.device = device_
        self.domain_size = int(domain_size)
        self.residual_chunk_size = max(256, int(residual_chunk_size))

        x = np.linspace(0.0, Config.LX, self.domain_size)
        y = np.linspace(0.0, Config.LY, self.domain_size)
        self.X, self.Y = np.meshgrid(x, y)
        self.pts = torch.tensor(
            np.stack([self.X.flatten(), self.Y.flatten()], axis=1),
            dtype=torch.get_default_dtype(),
            device=device_,
        )

        self.evaluate()

    def _fdm_cache_key(self) -> tuple:
        """Monta chave de cache para referência FDM."""
        return build_fdm_cache_key(
            nx=self.domain_size,
            ny=self.domain_size,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )

    def _reference_field_fdm(self) -> np.ndarray:
        """Gera campo FDM com a mesma resolução espacial da malha de avaliação."""
        key = self._fdm_cache_key()
        if key in self._fdm_cache:
            cached_field, cached_iters = self._fdm_cache[key]
            self.fdm_iterations = int(cached_iters)
            return cached_field.copy()

        t_ref, iters = generate_reference_solution(
            nx=self.domain_size,
            ny=self.domain_size,
            lx=Config.LX,
            ly=Config.LY,
            t_left=Config.T_LEFT,
            t_right=Config.T_RIGHT,
            tol=Config.FDM_TOL,
            max_iter=Config.FDM_MAX_ITER,
            omega=Config.FDM_OMEGA,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        ref_np = t_ref.detach().cpu().numpy()
        self._fdm_cache[key] = (ref_np.copy(), int(iters))
        self.fdm_iterations = int(iters)
        return ref_np

    def _compute_pde_residual_chunked(self) -> np.ndarray:
        """Calcula residual PDE em blocos para reduzir uso de memória."""
        chunks: list[np.ndarray] = []
        total = int(self.pts.shape[0])
        for start in range(0, total, self.residual_chunk_size):
            stop = min(start + self.residual_chunk_size, total)
            chunk = self.pts[start:stop]
            chunk_res = pde_residual(
                chunk,
                self.net,
                nondimensional=False,
                create_graph=False,
            ).detach().cpu().numpy()
            chunks.append(chunk_res)
        residual_flat = np.concatenate(chunks, axis=0)
        return residual_flat.reshape(self.domain_size, self.domain_size)

    def _compute_neumann_bc_residual(self, *, nondimensional: bool = False) -> np.ndarray:
        """
        Calcula residual Neumann nos contornos y=0 e y=Ly.

        Os vértices são excluídos por consistência física: nos cantos o problema
        já é imposto por Dirichlet lateral, então incluir também Neumann cria
        sobreposição artificial de condições de contorno.

        Quando nondimensional=True, aplica escala L_ref/DeltaT_ref para tornar
        o erro de fluxo compatível com métricas de temperatura adimensionais.
        """
        x_full = torch.linspace(
            0.0,
            float(Config.LX),
            self.domain_size,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        # Em malhas com >= 3 pontos, remove cantos para evitar conflito BC.
        x = x_full[1:-1] if x_full.numel() > 2 else x_full
        n_pts = int(x.shape[0])
        y_bottom = torch.zeros_like(x)
        y_top = torch.full_like(x, float(Config.LY))
        pts_bottom = torch.stack([x, y_bottom], dim=1)
        pts_top = torch.stack([x, y_top], dim=1)

        # Uma única chamada ao autograd reduz overhead sem alterar o cálculo
        # físico da condição de fluxo nulo em topo/base.
        pts_neu = torch.cat([pts_bottom, pts_top], dim=0)
        dTdy_all = (
            dT_dy_on(pts_neu, self.net, create_graph=False)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        dTdy_bottom = dTdy_all[:n_pts]
        dTdy_top = dTdy_all[n_pts:]
        neumann_abs = np.concatenate([np.abs(dTdy_bottom), np.abs(dTdy_top)])
        if not nondimensional:
            return neumann_abs
        temp_scale, length_scale = _physical_scales()
        return neumann_abs * (length_scale / temp_scale)

    def evaluate(self) -> None:
        """Calcula métricas globais e residual da EDP no domínio completo."""
        with torch.no_grad():
            self.T_pred = (
                self.net(self.pts)
                .detach()
                .cpu()
                .numpy()
                .reshape(self.domain_size, self.domain_size)
            )

        self.T_ref = self._reference_field_fdm()
        err = self.T_ref - self.T_pred
        abs_err = np.abs(err)
        r2_den = float(np.sum((self.T_ref - np.mean(self.T_ref)) ** 2))
        r2 = float(1.0 - np.sum(err**2) / r2_den) if r2_den > 1e-20 else float("nan")
        eps = 1.0e-12

        boundary_field_err = np.concatenate(
            [abs_err[:, 0], abs_err[:, -1], abs_err[0, :], abs_err[-1, :]]
        )
        temp_scale, length_scale = _physical_scales()
        dirichlet_left_bc_err = np.abs(self.T_pred[:, 0] - float(Config.T_LEFT))
        dirichlet_right_bc_err = np.abs(self.T_pred[:, -1] - float(Config.T_RIGHT))
        dirichlet_left_bc_err_nd = dirichlet_left_bc_err / temp_scale
        dirichlet_right_bc_err_nd = dirichlet_right_bc_err / temp_scale
        dir_lr_ratio = float(
            np.mean(dirichlet_right_bc_err_nd)
            / max(np.mean(dirichlet_left_bc_err_nd), 1.0e-12)
        )
        dirichlet_bc_err = np.concatenate([dirichlet_left_bc_err, dirichlet_right_bc_err])
        dirichlet_bc_err_nd = np.concatenate([dirichlet_left_bc_err_nd, dirichlet_right_bc_err_nd])
        neumann_bc_err = self._compute_neumann_bc_residual(nondimensional=False)
        neumann_bc_err_nd = neumann_bc_err * (length_scale / temp_scale)
        boundary_bc_total_err_nd = np.concatenate([dirichlet_bc_err_nd, neumann_bc_err_nd])
        # Mantido apenas para rastreio retroativo; fisicamente mistura unidades.
        boundary_bc_total_err_legacy_mixed = np.concatenate([dirichlet_bc_err, neumann_bc_err])
        boundary_bc_nd_mean = float(np.mean(boundary_bc_total_err_nd))
        boundary_dirichlet_nd_mean = float(np.mean(dirichlet_bc_err_nd))
        boundary_neumann_nd_mean = float(np.mean(neumann_bc_err_nd))

        relative_l2 = float(np.linalg.norm(err) / (np.linalg.norm(self.T_ref) + eps))
        y_axis = np.linspace(0.0, float(Config.LY), self.domain_size)
        target_y = 0.5 * float(Config.LY)
        mid_idx = int(np.argmin(np.abs(y_axis - target_y)))
        mid_ref = self.T_ref[mid_idx, :]
        mid_pred = self.T_pred[mid_idx, :]
        mid_err = mid_pred - mid_ref
        mid_abs_err = np.abs(mid_err)
        if self.domain_size >= 3:
            dx = float(Config.LX) / float(max(1, self.domain_size - 1))
            mid_curv_ref = np.gradient(np.gradient(mid_ref, dx), dx)
            mid_curv_pred = np.gradient(np.gradient(mid_pred, dx), dx)
            mid_curv_diff = np.abs(mid_curv_pred - mid_curv_ref)
            mid_curv_rmse = float(np.sqrt(np.mean((mid_curv_pred - mid_curv_ref) ** 2)))
            mid_curv_max = float(np.max(mid_curv_diff))
        else:
            mid_curv_rmse = float("nan")
            mid_curv_max = float("nan")

        self.metrics = {
            "MAE": float(np.mean(abs_err)),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "MAPE": stable_mape_percent(
                self.T_ref,
                self.T_pred,
                temp_scale=temp_scale,
                floor_fraction=1.0e-3,
            ),
            "max_error": float(np.max(abs_err)),
            "R2": r2,
            "relative_l2_error": relative_l2,
            "relative_l2_error_vs_fdm": relative_l2,
            # Não é métrica BC "pura": mede discrepância PINN-FDM nas bordas,
            # incluindo possíveis diferenças numéricas do solver de referência.
            "boundary_field_error": float(np.mean(boundary_field_err)),
            "boundary_dirichlet_error_nd": boundary_dirichlet_nd_mean,
            "boundary_neumann_error_nd": boundary_neumann_nd_mean,
            "boundary_bc_error_nd": boundary_bc_nd_mean,
            "boundary_bc_error_dirichlet": boundary_dirichlet_nd_mean,
            "boundary_bc_error_dirichlet_left": float(np.mean(dirichlet_left_bc_err_nd)),
            "boundary_bc_error_dirichlet_right": float(np.mean(dirichlet_right_bc_err_nd)),
            "boundary_bc_error_dirichlet_ratio_right_left": dir_lr_ratio,
            "boundary_bc_error_neumann": boundary_neumann_nd_mean,
            "boundary_bc_error_legacy_mixed_units": float(np.mean(boundary_bc_total_err_legacy_mixed)),
            "boundary_bc_error_legacy_dirichlet_dimensional": float(np.mean(dirichlet_bc_err)),
            "boundary_bc_error_legacy_neumann_dimensional": float(np.mean(neumann_bc_err)),
            # Compatibilidade retroativa.
            "boundary_bc_error": boundary_bc_nd_mean,
            "boundary_error": boundary_bc_nd_mean,
            "boundary_error_dirichlet": boundary_dirichlet_nd_mean,
            "boundary_error_dirichlet_left": float(np.mean(dirichlet_left_bc_err_nd)),
            "boundary_error_dirichlet_right": float(np.mean(dirichlet_right_bc_err_nd)),
            "boundary_error_dirichlet_ratio_right_left": dir_lr_ratio,
            "boundary_error_neumann": boundary_neumann_nd_mean,
            "midline_y_half_mae": float(np.mean(mid_abs_err)),
            "midline_y_half_rmse": float(np.sqrt(np.mean(mid_err**2))),
            "midline_y_half_max_error": float(np.max(mid_abs_err)),
            "midline_y_half_curvature_rmse": mid_curv_rmse,
            "midline_y_half_curvature_max": mid_curv_max,
        }

        self.pde_residual = self._compute_pde_residual_chunked()
        abs_residual = np.abs(self.pde_residual)
        nd_scale = float(
            residual_nondim_scale(device=self.device, dtype=torch.get_default_dtype()).item()
        )
        residual_nd = self.pde_residual * nd_scale
        abs_residual_nd = np.abs(residual_nd)
        self.metrics.update(
            {
                "pde_residual_mean": float(np.mean(abs_residual)),
                "pde_residual_signed_mean": float(np.mean(self.pde_residual)),
                "pde_residual_l2": float(np.linalg.norm(self.pde_residual)),
                "pde_residual_rms": float(
                    np.linalg.norm(self.pde_residual) / np.sqrt(self.pde_residual.size)
                ),
                "pde_residual_max": float(np.max(abs_residual)),
                "pde_residual_nd_mean": float(np.mean(abs_residual_nd)),
                "pde_residual_nd_signed_mean": float(np.mean(residual_nd)),
                "pde_residual_nd_l2": float(np.linalg.norm(residual_nd)),
                "pde_residual_nd_rms": float(
                    np.linalg.norm(residual_nd) / np.sqrt(residual_nd.size)
                ),
                "pde_residual_nd_max": float(np.max(abs_residual_nd)),
                "pde_residual_l2_reference": float(
                    pde_residual_l2_reference(
                        domain_size=self.domain_size,
                        temp_scale=temp_scale,
                        length_scale=length_scale,
                    )
                ),
                "pde_residual_l2_normalized": float(
                    pde_residual_l2_normalized(
                        float(np.linalg.norm(self.pde_residual)),
                        domain_size=self.domain_size,
                        temp_scale=temp_scale,
                        length_scale=length_scale,
                    )
                ),
            }
        )

    def print_comprehensive_analysis(self) -> None:
        """Registra no logger o resumo técnico das métricas calculadas."""
        logger.info("\n" + "=" * 80)
        logger.info("SCIENTIFIC PERFORMANCE MANIFEST")
        logger.info("=" * 80)
        logger.info("FDM (referência) convergiu em %d iterações", self.fdm_iterations)
        logger.info("Mean Absolute Error (MAE):     %.4e K", self.metrics["MAE"])
        logger.info("Root Mean Square Error (RMSE): %.4e K", self.metrics["RMSE"])
        logger.info("Mean Absolute Percentage Error: %.4e %%", self.metrics["MAPE"])
        logger.info("Coefficient of Determination:   %.6f", self.metrics["R2"])
        logger.info("Relative L2 Error:             %.4e", self.metrics["relative_l2_error"])
        logger.info("Global Peak Error:             %.4e K", self.metrics["max_error"])
        logger.info(
            "Boundary Field Error (mean abs): %.4e",
            self.metrics["boundary_field_error"],
        )
        logger.info(
            "Nota: Boundary Field Error compara PINN vs FDM nas bordas; não substitui erro BC puro."
        )
        logger.info(
            "Boundary BC Error ND (mean abs): %.4e",
            self.metrics["boundary_bc_error_nd"],
        )
        logger.info(
            "Dirichlet Error ND L/R (mean abs): %.4e / %.4e",
            self.metrics["boundary_bc_error_dirichlet_left"],
            self.metrics["boundary_bc_error_dirichlet_right"],
        )
        logger.info(
            "Dirichlet Asymmetry (R/L):       %.4f",
            self.metrics["boundary_bc_error_dirichlet_ratio_right_left"],
        )
        logger.info(
            "Neumann Error ND (mean abs):    %.4e",
            self.metrics["boundary_bc_error_neumann"],
        )
        logger.info(
            "Perfil y=0.5 | MAE/RMSE/max:     %.4e / %.4e / %.4e",
            self.metrics["midline_y_half_mae"],
            self.metrics["midline_y_half_rmse"],
            self.metrics["midline_y_half_max_error"],
        )
        logger.info(
            "Suavidade y=0.5 (curv RMSE/max): %.4e / %.4e",
            self.metrics["midline_y_half_curvature_rmse"],
            self.metrics["midline_y_half_curvature_max"],
        )
        logger.info(
            "PDE Residual |mean|/L2/RMS/max: %.4e / %.4e / %.4e / %.4e",
            self.metrics["pde_residual_mean"],
            self.metrics["pde_residual_l2"],
            self.metrics["pde_residual_rms"],
            self.metrics["pde_residual_max"],
        )
        logger.info(
            "PDE Residual ND |mean|/L2/RMS/max: %.4e / %.4e / %.4e / %.4e",
            self.metrics["pde_residual_nd_mean"],
            self.metrics["pde_residual_nd_l2"],
            self.metrics["pde_residual_nd_rms"],
            self.metrics["pde_residual_nd_max"],
        )
        logger.info(
            "PDE Residual L2 Normalized:      %.4e",
            self.metrics["pde_residual_l2_normalized"],
        )
        logger.info(
            "Nota: ||residual||_2 dimensional depende da resolução; prefira RMS e L2 normalizado para comparação entre malhas."
        )
        logger.info("=" * 80)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Retorna média móvel simples de values com janela fixa."""
    if values.size == 0:
        return values
    win = max(1, int(window))
    if win == 1 or values.size < win:
        return values.copy()
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(values, kernel, mode="valid")


def analyze_convergence(history: Dict[str, List[Any]]) -> Dict[str, float]:
    """Extrai estatísticas de convergência usando todas as fases de otimização."""
    total_history = history.get("total_loss", [])
    if total_history:
        losses = np.array(total_history, dtype=np.float64)
    elif history.get("adam_loss"):
        losses = np.array(history["adam_loss"], dtype=np.float64)
    else:
        return {}

    first = float(losses[0])
    reduction = (
        float((first - losses[-1]) / first * 100.0) if abs(first) > 1e-20 else float("nan")
    )
    return {
        "final_loss": float(losses[-1]),
        "total_steps": int(len(losses)),
        "total_epochs": int(len(losses)),
        "loss_reduction": reduction,
    }


def assess_publication_readiness(
    history: Dict[str, List[Any]],
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Verifica critérios mínimos de confiabilidade para uso de resultados em publicação.
    """
    total_loss_hist = np.array(history.get("total_loss", []), dtype=np.float64)
    if total_loss_hist.size == 0:
        total_loss_hist = np.array(history.get("adam_loss", []), dtype=np.float64)
    pde_loss_hist = np.array(history.get("L_PDE", []), dtype=np.float64)

    final_total_loss = float(total_loss_hist[-1]) if total_loss_hist.size > 0 else float("nan")
    final_pde_loss = float(pde_loss_hist[-1]) if pde_loss_hist.size > 0 else float("nan")

    if pde_loss_hist.size > 0:
        edge = max(5, int(np.ceil(0.2 * pde_loss_hist.size)))
        early_ref = float(np.median(pde_loss_hist[:edge]))
        late_ref = float(np.median(pde_loss_hist[-edge:]))
        pde_reduction = (
            float((early_ref - late_ref) / early_ref * 100.0)
            if abs(early_ref) > 1e-20
            else float("nan")
        )
        smooth_win = max(5, min(80, int(pde_loss_hist.size // 20) or 5))
        smoothed = _moving_average(pde_loss_hist, smooth_win)
        if smoothed.size > 1:
            diffs = np.diff(smoothed)
            non_increasing_ratio = float(np.mean(diffs <= 0.0))
            baseline = float(np.median(smoothed[: max(3, smoothed.size // 5)]))
            physical_monotonic = bool(
                non_increasing_ratio >= 0.60 and float(smoothed[-1]) <= baseline
            )
        else:
            physical_monotonic = False
        tail = smoothed[-max(3, min(20, smoothed.size)) :] if smoothed.size > 0 else smoothed
        tail_mean = float(np.mean(tail)) if tail.size > 0 else float("nan")
        tail_std = float(np.std(tail)) if tail.size > 0 else float("nan")
        residual_stability = bool(
            tail.size > 0 and tail_mean > 1e-20 and (tail_std / tail_mean) <= 0.50
        )
    else:
        pde_reduction = float("nan")
        physical_monotonic = False
        residual_stability = False

    if total_loss_hist.size > 0 and pde_loss_hist.size > 0:
        paired = min(total_loss_hist.size, pde_loss_hist.size)
        total_pair = total_loss_hist[-paired:]
        pde_pair = pde_loss_hist[-paired:]
        edge_pair = max(3, int(np.ceil(0.2 * paired)))
        early_total = float(np.median(total_pair[:edge_pair]))
        late_total = float(np.median(total_pair[-edge_pair:]))
        early_pde = float(np.median(pde_pair[:edge_pair]))
        late_pde = float(np.median(pde_pair[-edge_pair:]))
        loss_reduced = late_total < early_total
        pde_growth_ratio = (late_pde - early_pde) / max(abs(early_pde), 1.0e-20)
    else:
        loss_reduced = False
        pde_growth_ratio = float("nan")

    rar_selected = np.array(history.get("rar_selected_residual_mean", []), dtype=np.float64)
    rar_global = np.array(history.get("rar_global_residual_rms", []), dtype=np.float64)
    valid_mask = np.isfinite(rar_selected) & np.isfinite(rar_global) & (rar_global > 1e-20)
    if np.any(valid_mask):
        rar_ratio = rar_selected[valid_mask] / rar_global[valid_mask]
        rar_guidance_ok = bool(float(np.median(rar_ratio)) > 1.0)
    else:
        rar_guidance_ok = True

    checks = {
        "physical_error_reduction": bool(np.isfinite(pde_reduction) and pde_reduction > 0.0),
        "physical_monotonic_trend": physical_monotonic,
        "residual_stability": residual_stability,
        "rar_guidance": rar_guidance_ok,
        "pde_residual_nd_mean_ok": float(metrics.get("pde_residual_nd_mean", np.inf))
        <= float(getattr(Config, "PDE_RESIDUAL_ND_MEAN_THRESHOLD", 5.0e-3)),
        "pde_residual_nd_max_ok": float(metrics.get("pde_residual_nd_max", np.inf))
        <= float(getattr(Config, "PDE_RESIDUAL_ND_MAX_THRESHOLD", 5.0e-2)),
        "relative_l2_ok": float(metrics.get("relative_l2_error", np.inf))
        <= float(getattr(Config, "RELATIVE_L2_THRESHOLD", 1.0e-3)),
        "boundary_bc_ok": float(
            metrics.get("boundary_bc_error_nd", metrics.get("boundary_bc_error", np.inf))
        )
        <= float(
            getattr(
                Config,
                "BOUNDARY_BC_ND_THRESHOLD",
                getattr(Config, "BOUNDARY_BC_THRESHOLD", 1.0e-2),
            )
        ),
    }

    false_convergence = bool(
        np.isfinite(final_total_loss)
        and final_total_loss <= float(getattr(Config, "FALSE_CONV_LOSS_THRESHOLD", 1.0e-4))
        and float(metrics.get("pde_residual_nd_mean", np.inf))
        > float(getattr(Config, "FALSE_CONV_PDE_ND_THRESHOLD", 1.0e-2))
    )
    false_convergence_trend = bool(
        loss_reduced
        and np.isfinite(pde_growth_ratio)
        and pde_growth_ratio
        > float(getattr(Config, "FALSE_CONV_PDE_GROWTH_THRESHOLD", 1.0e-1))
    )
    checks["false_convergence_absent"] = not false_convergence
    checks["false_convergence_trend_absent"] = not false_convergence_trend

    failed_checks = [name for name, ok in checks.items() if not ok]
    status = "Approved" if not failed_checks else "Request Changes"

    return {
        "status": status,
        "publication_ready": status == "Approved",
        "failed_checks": failed_checks,
        "checks": checks,
        "final_total_loss": final_total_loss,
        "final_pde_loss": final_pde_loss,
        "pde_loss_reduction_pct": pde_reduction,
        "relative_l2_error": float(metrics.get("relative_l2_error", float("nan"))),
        "pde_residual_nd_mean": float(metrics.get("pde_residual_nd_mean", float("nan"))),
        "boundary_bc_error_nd": float(
            metrics.get("boundary_bc_error_nd", metrics.get("boundary_bc_error", float("nan")))
        ),
        "boundary_bc_error": float(
            metrics.get("boundary_bc_error_nd", metrics.get("boundary_bc_error", float("nan")))
        ),
        "false_convergence_pde_growth_ratio": pde_growth_ratio,
    }

