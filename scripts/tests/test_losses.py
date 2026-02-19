"""
Testes dos termos de perda física e supervisionada do treinamento PINN.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import Config
from losses import enhanced_loss_terms
from operators import pde_residual


class LinearXNet(nn.Module):
    """Modelo analítico com grafo suave para validação de autograd."""

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Calcula temperatura sintética com termo cúbico em y."""
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        return 200.0 - 100.0 * x + y**3


def _build_points(n: int = 64) -> torch.Tensor:
    return torch.rand((n, 2), dtype=torch.get_default_dtype())


def _build_points_y0(n: int = 64) -> torch.Tensor:
    return torch.cat(
        [
            torch.rand((n, 1), dtype=torch.get_default_dtype()),
            torch.zeros((n, 1), dtype=torch.get_default_dtype()),
        ],
        dim=1,
    )


def _build_dirichlet_xy(
    n_side: int = 40,
) -> torch.Tensor:
    """Monta pontos Dirichlet em x=0 e x=1 com amostragem uniforme em y."""
    y_dir = torch.rand((n_side, 1), dtype=torch.get_default_dtype())
    x_left = torch.zeros_like(y_dir)
    x_right = torch.ones_like(y_dir)
    return torch.cat([torch.cat([x_left, y_dir], 1), torch.cat([x_right, y_dir], 1)], dim=0)


def _run_dirichlet_only_loss(
    *,
    net: nn.Module,
    x_f: torch.Tensor,
    x_dir: torch.Tensor,
    t_dir_target: torch.Tensor,
    x_bottom: torch.Tensor,
    x_top: torch.Tensor,
    left_mult: float | None = None,
    right_mult: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Executa loss com foco exclusivo no termo Dirichlet para testes de pesos."""
    return enhanced_loss_terms(
        net=net,
        X_f=x_f,
        X_dir=x_dir,
        T_dir_target=t_dir_target,
        X_bottom=x_bottom,
        X_top=x_top,
        w_pde=0.0,
        w_dir=1.0,
        left_dirichlet_weight_multiplier=left_mult,
        right_dirichlet_weight_multiplier=right_mult,
        w_neu=0.0,
        w_data=0.0,
    )


def _run_pde_only_loss(
    *,
    net: nn.Module,
    x_f: torch.Tensor,
    x_dir: torch.Tensor,
    t_dir_target: torch.Tensor,
    x_bottom: torch.Tensor,
    x_top: torch.Tensor,
    create_graph: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Executa loss com foco no residual PDE para comparar modos de grafo."""
    return enhanced_loss_terms(
        net=net,
        X_f=x_f,
        X_dir=x_dir,
        T_dir_target=t_dir_target,
        X_bottom=x_bottom,
        X_top=x_top,
        w_pde=1.0,
        w_dir=0.0,
        w_neu=0.0,
        w_data=0.0,
        create_graph=create_graph,
    )


def test_enhanced_loss_terms_zero_for_exact_linear_solution():
    net = LinearXNet()
    x_f = _build_points_y0(80)
    x_data = _build_points(80)
    t_data = net(x_data).detach()

    x_dir = _build_dirichlet_xy(40)
    t_dir = net(x_dir).detach()

    x_bottom = _build_points_y0(40)
    x_top = _build_points_y0(40)
    x_midline = torch.cat(
        [
            torch.rand((40, 1), dtype=torch.get_default_dtype()),
            torch.full((40, 1), 0.5, dtype=torch.get_default_dtype()),
        ],
        dim=1,
    )

    total, parts = enhanced_loss_terms(
        net=net,
        X_f=x_f,
        X_dir=x_dir,
        T_dir_target=t_dir,
        X_bottom=x_bottom,
        X_top=x_top,
        X_data=x_data,
        T_data_target=t_data,
        X_midline=x_midline,
        w_pde=1.0,
        w_dir=1.0,
        w_neu=1.0,
        w_data=1.0,
        w_curv=1.0,
    )

    assert float(parts["L_PDE"].item()) < 1e-12
    assert float(parts["L_D"].item()) < 1e-12
    assert float(parts["L_N"].item()) < 1e-12
    assert float(parts["L_data"].item()) < 1e-12
    assert float(parts["L_curv"].item()) < 1e-12
    assert float(total.item()) < 1e-12


def test_enhanced_loss_terms_requires_data_pair():
    net = LinearXNet()
    x = _build_points_y0(16)
    t = net(x).detach()

    try:
        enhanced_loss_terms(
            net=net,
            X_f=x,
            X_dir=x,
            T_dir_target=t,
            X_bottom=x,
            X_top=x,
            X_data=x,
            T_data_target=None,
            w_data=1.0,
        )
    except ValueError as exc:
        assert "fornecidos juntos" in str(exc)
    else:
        raise AssertionError("Era esperado ValueError para par de dados inconsistente.")


def test_enhanced_loss_terms_left_dirichlet_multiplier_prioritizes_left_boundary():
    net = LinearXNet()
    x_f = _build_points_y0(80)

    x_dir = _build_dirichlet_xy(40)
    t_dir = net(x_dir).detach()
    t_dir_left_shifted = t_dir.clone()
    t_dir_left_shifted[:40] += 1.0

    x_bottom = _build_points_y0(40)
    x_top = _build_points_y0(40)

    total_base, parts_base = _run_dirichlet_only_loss(
        net=net,
        x_f=x_f,
        x_dir=x_dir,
        t_dir_target=t_dir_left_shifted,
        x_bottom=x_bottom,
        x_top=x_top,
        left_mult=1.0,
    )
    total_left_boost, parts_left_boost = _run_dirichlet_only_loss(
        net=net,
        x_f=x_f,
        x_dir=x_dir,
        t_dir_target=t_dir_left_shifted,
        x_bottom=x_bottom,
        x_top=x_top,
        left_mult=3.0,
    )

    assert float(parts_base["L_D_right"].item()) < 1e-12
    assert float(parts_left_boost["L_D_right"].item()) < 1e-12
    assert float(parts_base["L_D_left"].item()) > 0.0
    assert float(parts_left_boost["L_D_left"].item()) > 0.0
    assert float(parts_left_boost["L_D"].item()) > float(parts_base["L_D"].item())
    assert float(total_left_boost.item()) > float(total_base.item())


def test_enhanced_loss_terms_right_dirichlet_multiplier_prioritizes_right_boundary():
    net = LinearXNet()
    x_f = _build_points_y0(80)

    x_dir = _build_dirichlet_xy(40)
    t_dir = net(x_dir).detach()
    t_dir_right_shifted = t_dir.clone()
    t_dir_right_shifted[40:] += 1.0

    x_bottom = _build_points_y0(40)
    x_top = _build_points_y0(40)

    total_base, parts_base = _run_dirichlet_only_loss(
        net=net,
        x_f=x_f,
        x_dir=x_dir,
        t_dir_target=t_dir_right_shifted,
        x_bottom=x_bottom,
        x_top=x_top,
        right_mult=1.0,
    )
    total_right_boost, parts_right_boost = _run_dirichlet_only_loss(
        net=net,
        x_f=x_f,
        x_dir=x_dir,
        t_dir_target=t_dir_right_shifted,
        x_bottom=x_bottom,
        x_top=x_top,
        right_mult=3.0,
    )

    assert float(parts_base["L_D_left"].item()) < 1e-12
    assert float(parts_right_boost["L_D_left"].item()) < 1e-12
    assert float(parts_base["L_D_right"].item()) > 0.0
    assert float(parts_right_boost["L_D_right"].item()) > 0.0
    assert float(parts_right_boost["L_D"].item()) > float(parts_base["L_D"].item())
    assert float(total_right_boost.item()) > float(total_base.item())


def test_pde_residual_nondimensional_uses_expected_scale():
    net = LinearXNet()
    pts = _build_points(32)
    res_dim = pde_residual(pts, net, nondimensional=False)
    res_nd = pde_residual(pts, net, nondimensional=True)

    temp_scale = max(abs(float(Config.T_LEFT) - float(Config.T_RIGHT)), 1.0)
    length_scale = max(float(Config.LX), float(Config.LY), 1.0e-12)
    expected = res_dim * (length_scale**2 / temp_scale)
    assert torch.allclose(res_nd, expected, atol=1e-10, rtol=1e-9)


def test_enhanced_loss_terms_supports_create_graph_false_for_evaluation():
    net = LinearXNet()
    x_f = _build_points(48)
    x_dir = _build_points(48)
    t_dir = net(x_dir).detach()
    x_bottom = _build_points_y0(24)
    x_top = _build_points_y0(24)

    total_eval, parts_eval = _run_pde_only_loss(
        net=net,
        x_f=x_f,
        x_dir=x_dir,
        t_dir_target=t_dir,
        x_bottom=x_bottom,
        x_top=x_top,
        create_graph=False,
    )
    total_train, parts_train = _run_pde_only_loss(
        net=net,
        x_f=x_f,
        x_dir=x_dir,
        t_dir_target=t_dir,
        x_bottom=x_bottom,
        x_top=x_top,
        create_graph=True,
    )

    assert not parts_eval["L_PDE"].requires_grad
    assert parts_train["L_PDE"].requires_grad


def test_enhanced_loss_terms_skips_dirichlet_when_hard_constraint_enabled(config_guard):
    config_guard.ENABLE_HARD_DIRICHLET_CONSTRAINT = True
    net = LinearXNet()
    x_f = _build_points(32)
    x_dir = _build_points(32)
    t_dir = net(x_dir).detach()
    x_bottom = _build_points_y0(16)
    x_top = _build_points_y0(16)

    _, parts = enhanced_loss_terms(
        net=net,
        X_f=x_f,
        X_dir=x_dir,
        T_dir_target=t_dir,
        X_bottom=x_bottom,
        X_top=x_top,
        w_pde=0.0,
        w_dir=1.0,
        w_neu=0.0,
        w_data=0.0,
    )

    assert float(parts["L_D"].item()) == 0.0
    assert float(parts["L_D_left"].item()) == 0.0
    assert float(parts["L_D_right"].item()) == 0.0
    assert float(parts["L_D_left_mae_nd"].item()) == 0.0
    assert float(parts["L_D_right_mae_nd"].item()) == 0.0

