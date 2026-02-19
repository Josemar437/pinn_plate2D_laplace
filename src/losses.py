"""
Definição dos termos de perda e do rebalanceamento adaptativo para a PINN 2D.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from .config import Config, physical_scales_tensor
    from .operators import d2T_dx2_on, dT_dy_on, pde_residual, residual_scale
except ImportError:  
    from config import Config, physical_scales_tensor
    from operators import d2T_dx2_on, dT_dy_on, pde_residual, residual_scale


def _physical_scales(*, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Retorna escalas características para não-dimensionalização da perda.

    - temp_scale: variação de temperatura de referência.
    - length_scale: escala espacial característica do domínio.
    """
    return physical_scales_tensor(device=device, dtype=dtype)


def _zero_dirichlet_terms(
    *, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Retorna termos Dirichlet nulos no dtype/dispositivo corretos.

    Usado quando Dirichlet é imposto por arquitetura ou quando o conjunto de
    borda está vazio, evitando duplicação de blocos idênticos.
    """
    zero = torch.zeros((), device=device, dtype=dtype)
    return zero, zero, zero, zero, zero


def enhanced_loss_terms(
    net: nn.Module,
    X_f: torch.Tensor,
    X_dir: torch.Tensor,
    T_dir_target: torch.Tensor,
    X_bottom: torch.Tensor,
    X_top: torch.Tensor,
    X_data: torch.Tensor | None = None,
    T_data_target: torch.Tensor | None = None,
    X_midline: torch.Tensor | None = None,
    w_pde: float = 1.0,
    w_dir: float = 100.0,
    left_dirichlet_weight_multiplier: float | None = None,
    right_dirichlet_weight_multiplier: float | None = None,
    w_neu: float = 10.0,
    w_data: float = 0.0,
    w_curv: float = 0.0,
    create_graph: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Calcula perdas físicas, de contorno e supervisionada em uma única chamada.

    Retorna a perda total ponderada e um dicionário com cada componente.
    Os termos de contorno usam forma quadrática para otimização, mas também
    retornam MAE adimensional por lado para controle robusto de assimetria.
    """
    temp_scale, length_scale = _physical_scales(device=X_f.device, dtype=X_f.dtype)

    residual_nd = pde_residual(
        X_f,
        net,
        nondimensional=True,
        create_graph=create_graph,
    )
    L_PDE = torch.mean(residual_nd**2)
    topk_weight = float(getattr(Config, "PDE_TOPK_WEIGHT", 0.0))
    if topk_weight > 0.0:
        topk_fraction = float(getattr(Config, "PDE_TOPK_FRACTION", 0.0))
        topk_fraction = min(max(topk_fraction, 1.0e-4), 1.0)
        residual_abs = torch.abs(residual_nd).reshape(-1)
        if residual_abs.numel() > 0:
            k = max(1, int(round(float(residual_abs.numel()) * topk_fraction)))
            topk_vals = torch.topk(residual_abs, k=k, largest=True).values
            L_PDE = L_PDE + topk_weight * torch.mean(topk_vals**2)

    hard_dirichlet_enabled = bool(
        getattr(Config, "ENABLE_HARD_DIRICHLET_CONSTRAINT", False)
    )
    if hard_dirichlet_enabled:
        L_D, L_D_left, L_D_right, L_D_left_mae, L_D_right_mae = _zero_dirichlet_terms(
            device=X_f.device,
            dtype=X_f.dtype,
        )
    else:
        T_pred_dir = net(X_dir.detach())
        dirichlet_err_nd = (T_pred_dir - T_dir_target) / temp_scale
        dirichlet_sq_err = dirichlet_err_nd**2
        dirichlet_abs_err = torch.abs(dirichlet_err_nd)

        if dirichlet_sq_err.numel() == 0:
            L_D, L_D_left, L_D_right, L_D_left_mae, L_D_right_mae = _zero_dirichlet_terms(
                device=X_f.device,
                dtype=X_f.dtype,
            )
            left_weight = 1.0
            right_weight = 1.0
            left_mask = torch.zeros_like(X_dir[:, 0:1], dtype=torch.bool)
            right_mask = ~left_mask
        else:
            left_weight = float(
                Config.LEFT_DIRICHLET_WEIGHT_MULTIPLIER
                if left_dirichlet_weight_multiplier is None
                else left_dirichlet_weight_multiplier
            )
            right_weight = float(
                Config.RIGHT_DIRICHLET_LOSS_MULTIPLIER
                if right_dirichlet_weight_multiplier is None
                else right_dirichlet_weight_multiplier
            )
            left_weight = max(left_weight, 1.0e-6)
            right_weight = max(right_weight, 1.0e-6)

            x_coords = X_dir[:, 0:1]
            zero = torch.tensor(0.0, device=x_coords.device, dtype=x_coords.dtype)
            atol = max(float(Config.LX), 1.0) * 1.0e-10
            left_mask = torch.isclose(x_coords, zero, atol=atol, rtol=0.0)
            right_mask = ~left_mask

            weights = torch.ones_like(dirichlet_sq_err)
            if left_weight != 1.0:
                left_weight_tensor = torch.full_like(weights, left_weight)
                weights = torch.where(left_mask, left_weight_tensor, weights)
            if right_weight != 1.0:
                right_weight_tensor = torch.full_like(weights, right_weight)
                weights = torch.where(right_mask, right_weight_tensor, weights)
            L_D = torch.sum(weights * dirichlet_sq_err) / torch.clamp(
                torch.sum(weights), min=1.0e-12
            )

            if torch.any(left_mask):
                L_D_left = torch.mean(dirichlet_sq_err[left_mask])
                L_D_left_mae = torch.mean(dirichlet_abs_err[left_mask])
            else:
                L_D_left = torch.zeros((), device=X_f.device, dtype=X_f.dtype)
                L_D_left_mae = torch.zeros((), device=X_f.device, dtype=X_f.dtype)
            if torch.any(right_mask):
                L_D_right = torch.mean(dirichlet_sq_err[right_mask])
                L_D_right_mae = torch.mean(dirichlet_abs_err[right_mask])
            else:
                L_D_right = torch.zeros((), device=X_f.device, dtype=X_f.dtype)
                L_D_right_mae = torch.zeros((), device=X_f.device, dtype=X_f.dtype)

    # Topo e base compartilham o mesmo operador diferencial; avaliar em lote
    # reduz chamadas de autograd sem alterar a forma física da condição Neumann.
    n_bottom = int(X_bottom.shape[0])
    n_top = int(X_top.shape[0])
    if n_bottom > 0 and n_top > 0:
        X_neu = torch.cat([X_bottom.detach(), X_top.detach()], dim=0)
        dTdy_all = dT_dy_on(X_neu, net, create_graph=create_graph)
        dTdy_bottom = dTdy_all[:n_bottom]
        dTdy_top = dTdy_all[n_bottom:]
    elif n_bottom > 0:
        dTdy_bottom = dT_dy_on(X_bottom.detach(), net, create_graph=create_graph)
        dTdy_top = torch.zeros(
            (0, 1),
            device=X_bottom.device,
            dtype=X_bottom.dtype,
        )
    elif n_top > 0:
        dTdy_bottom = torch.zeros(
            (0, 1),
            device=X_top.device,
            dtype=X_top.dtype,
        )
        dTdy_top = dT_dy_on(X_top.detach(), net, create_graph=create_graph)
    else:
        dTdy_bottom = torch.zeros((0, 1), device=X_f.device, dtype=X_f.dtype)
        dTdy_top = torch.zeros((0, 1), device=X_f.device, dtype=X_f.dtype)
    dTdy_bottom_nd = dTdy_bottom * (length_scale / temp_scale)
    dTdy_top_nd = dTdy_top * (length_scale / temp_scale)
    L_N_bottom = (
        torch.mean(dTdy_bottom_nd**2)
        if dTdy_bottom_nd.numel() > 0
        else torch.zeros((), device=X_f.device, dtype=X_f.dtype)
    )
    L_N_top = (
        torch.mean(dTdy_top_nd**2)
        if dTdy_top_nd.numel() > 0
        else torch.zeros((), device=X_f.device, dtype=X_f.dtype)
    )
    L_N = L_N_bottom + L_N_top

    if (X_data is None) != (T_data_target is None):
        raise ValueError("X_data e T_data_target devem ser fornecidos juntos.")

    if X_data is None:
        L_data = torch.zeros((), device=X_f.device, dtype=X_f.dtype)
    else:
        T_pred_data = net(X_data.detach())
        data_err_nd = (T_pred_data - T_data_target) / temp_scale
        L_data = torch.mean(data_err_nd**2)

    if w_curv > 0.0 and X_midline is not None:
        d2Tdx2_mid = d2T_dx2_on(X_midline, net, create_graph=create_graph)
        curv_scale = residual_scale(device=d2Tdx2_mid.device, dtype=d2Tdx2_mid.dtype)
        d2Tdx2_mid_nd = d2Tdx2_mid * curv_scale
        L_curv = torch.mean(d2Tdx2_mid_nd**2)
    else:
        L_curv = torch.zeros((), device=X_f.device, dtype=X_f.dtype)

    L_total = w_pde * L_PDE + w_dir * L_D + w_neu * L_N + w_data * L_data + w_curv * L_curv

    loss_dict = {
        "L_PDE": L_PDE,
        "L_D": L_D,
        "L_D_left": L_D_left,
        "L_D_right": L_D_right,
        "L_D_left_mae_nd": L_D_left_mae,
        "L_D_right_mae_nd": L_D_right_mae,
        "L_N": L_N,
        "L_data": L_data,
        "L_curv": L_curv,
        "L_total": L_total,
    }

    return L_total, loss_dict


class AdaptiveWeightScheduler:
    """Reequilibra pesos de perda com base na média móvel dos componentes."""
    
    def __init__(
        self,
        w_pde: float = 1.0,
        w_dir: float = 100.0,
        w_neu: float = 10.0,
        balance_frequency: int = 100,
        alpha: float = 0.95,
        max_weight_ratio: float = 1000.0,
        strategy: str = "loss_balance",
        min_balance_step: int = 0,
        window: int = 10,
        max_weight_change: float = 0.30,
        use_median: bool = True,
    ):
        """Inicializa pesos e parâmetros de atualização periódica."""
        self.w_pde = float(w_pde)
        self.w_dir = float(w_dir)
        self.w_neu = float(w_neu)
        self.balance_frequency = int(balance_frequency)
        self.alpha = float(alpha)
        self.max_weight_ratio = float(max_weight_ratio)
        self.strategy = str(strategy).strip().lower()
        valid_strategies = {"loss_balance", "inverse_loss", "grad_norm_balance"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"strategy inválida: {self.strategy!r}. Use uma de {sorted(valid_strategies)}."
            )
        self.min_balance_step = int(min_balance_step)
        self.window = int(window)
        self.max_weight_change = float(max_weight_change)
        self.use_median = bool(use_median)
        self.step_count = 0
        self.loss_history: Dict[str, list[float]] = {"L_PDE": [], "L_D": [], "L_N": []}

    def needs_gradient_stats(self) -> bool:
        """
        Indica se o próximo passo de rebalanceamento exige estatística de gradiente.

        Quando strategy='grad_norm_balance', essa checagem evita custo extra
        de autograd em épocas onde não haverá atualização de pesos.
        """
        if self.strategy != "grad_norm_balance":
            return False
        next_step = self.step_count + 1
        if next_step < self.min_balance_step:
            return False
        if next_step % self.balance_frequency != 0:
            return False
        required = max(3, self.window)
        return (len(self.loss_history["L_PDE"]) + 1) >= required

    def step(
        self,
        loss_dict: Dict[str, torch.Tensor],
        grad_stats: Dict[str, float] | None = None,
    ) -> None:
        """Registra perdas da época e dispara rebalanceamento periódico."""
        self.step_count += 1
        for key in self.loss_history.keys():
            if key in loss_dict:
                self.loss_history[key].append(float(loss_dict[key].item()))

        if self.step_count < self.min_balance_step:
            return

        if (
            self.step_count % self.balance_frequency == 0
            and len(self.loss_history["L_PDE"]) >= max(3, self.window)
        ):
            self._rebalance_weights(grad_stats=grad_stats)

    def _aggregate_recent(self, values: list[float], win: int) -> float:
        """Agrega histórico recente com mediana ou média."""
        recent = torch.tensor(values[-win:])
        if self.use_median:
            return float(torch.median(recent).item())
        return float(torch.mean(recent).item())

    def _bounded_update(self, current: float, target: float) -> float:
        """
        Limita a variação multiplicativa de peso para evitar saltos bruscos.

        A atualização final combina limite multiplicativo local + suavização EMA.
        """
        current = float(max(current, 1e-12))
        target = float(max(target, 1e-12))
        max_up = current * (1.0 + self.max_weight_change)
        max_down = current / (1.0 + self.max_weight_change)
        limited_target = min(max(target, max_down), max_up)
        return self.alpha * current + (1.0 - self.alpha) * limited_target

    def _rebalance_weights(self, grad_stats: Dict[str, float] | None = None) -> None:
        """Atualiza pesos para reduzir discrepância entre termos de perda."""
        win = min(max(3, self.window), len(self.loss_history["L_PDE"]))

        r_pde = max(self._aggregate_recent(self.loss_history["L_PDE"], win), 1e-12)
        r_dir = max(self._aggregate_recent(self.loss_history["L_D"], win), 1e-12)
        r_neu = max(self._aggregate_recent(self.loss_history["L_N"], win), 1e-12)

        if self.strategy == "loss_balance":
            target_w_dir = min(r_pde / r_dir, self.max_weight_ratio)
            target_w_neu = min(r_pde / r_neu, self.max_weight_ratio)
        elif self.strategy == "inverse_loss":
            # Suaviza por raiz para reduzir oscilação.
            target_w_dir = min((r_pde / r_dir) ** 0.5, self.max_weight_ratio)
            target_w_neu = min((r_pde / r_neu) ** 0.5, self.max_weight_ratio)
        else:
            # Combina razão de perdas + razão de normas de gradiente para reduzir
            # competição entre termos de fronteira e PDE.
            if grad_stats is None:
                target_w_dir = min((r_pde / r_dir) ** 0.5, self.max_weight_ratio)
                target_w_neu = min((r_pde / r_neu) ** 0.5, self.max_weight_ratio)
            else:
                g_pde = max(abs(float(grad_stats.get("L_PDE", 0.0))), 1e-12)
                g_dir = max(abs(float(grad_stats.get("L_D", 0.0))), 1e-12)
                g_neu = max(abs(float(grad_stats.get("L_N", 0.0))), 1e-12)
                g_ref = (g_pde + g_dir + g_neu) / 3.0
                target_w_dir = min(
                    ((r_pde / r_dir) ** 0.5) * ((g_ref / g_dir) ** 0.5),
                    self.max_weight_ratio,
                )
                target_w_neu = min(
                    ((r_pde / r_neu) ** 0.5) * ((g_ref / g_neu) ** 0.5),
                    self.max_weight_ratio,
                )

        self.w_dir = self._bounded_update(self.w_dir, target_w_dir)
        self.w_neu = self._bounded_update(self.w_neu, target_w_neu)

    def get_weights(self) -> Tuple[float, float, float]:
        """Retorna pesos atuais dos termos PDE, Dirichlet e Neumann."""
        return float(self.w_pde), float(self.w_dir), float(self.w_neu)

