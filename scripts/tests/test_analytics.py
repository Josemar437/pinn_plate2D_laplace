"""
Testes das métricas analíticas e de consistência física do módulo analytics.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

import analytics as analytics_module
from _autograd_capture import make_captured_dT_dy_stub
from analytics import (
    EnhancedThermalAnalyzer,
    analyze_convergence,
    assess_publication_readiness,
    stable_mape_percent,
)
from config import Config, device


class LinearXNet(nn.Module):
    """Solução analítica para Laplace com BC mistas do problema."""

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Retorna perfil linear em x, solução exata para o caso de teste."""
        x = xy[:, 0:1]
        return 200.0 - 100.0 * x


class OffsetLinearXNet(nn.Module):
    """Perfil linear com offset para testar escalonamento de contorno ND."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = float(offset)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0:1]
        return 200.0 - 100.0 * x + self.offset


def test_analyzer_reports_extended_metrics(config_guard):
    config_guard.FDM_NX = 41
    config_guard.FDM_NY = 41
    config_guard.FDM_MAX_ITER = 20000
    config_guard.FDM_TOL = 1e-8

    net = LinearXNet().to(device)
    analyzer = EnhancedThermalAnalyzer(net=net, domain_size=41, device_=device)

    expected_keys = {
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
        "max_error",
        "relative_l2_error",
        "relative_l2_error_vs_fdm",
        "boundary_field_error",
        "boundary_dirichlet_error_nd",
        "boundary_neumann_error_nd",
        "boundary_bc_error_nd",
        "boundary_bc_error",
        "boundary_bc_error_dirichlet_left",
        "boundary_bc_error_dirichlet_right",
        "boundary_bc_error_dirichlet_ratio_right_left",
        "boundary_error",
        "boundary_error_dirichlet_left",
        "boundary_error_dirichlet_right",
        "boundary_error_dirichlet_ratio_right_left",
        "midline_y_half_mae",
        "midline_y_half_rmse",
        "midline_y_half_max_error",
        "midline_y_half_curvature_rmse",
        "midline_y_half_curvature_max",
        "pde_residual_mean",
        "pde_residual_l2",
        "pde_residual_rms",
        "pde_residual_max",
        "pde_residual_nd_mean",
        "pde_residual_nd_l2",
        "pde_residual_nd_rms",
        "pde_residual_nd_max",
        "pde_residual_l2_normalized",
    }
    assert expected_keys.issubset(set(analyzer.metrics.keys()))
    assert analyzer.metrics["relative_l2_error"] >= 0.0
    assert analyzer.metrics["pde_residual_l2"] >= 0.0


def test_boundary_metric_is_nondimensional_and_consistent(config_guard):
    config_guard.FDM_NX = 31
    config_guard.FDM_NY = 31
    config_guard.FDM_MAX_ITER = 20000
    config_guard.FDM_TOL = 1e-8
    config_guard.T_LEFT = 200.0
    config_guard.T_RIGHT = 100.0
    config_guard.LX = 1.0
    config_guard.LY = 1.0

    net = OffsetLinearXNet(offset=10.0).to(device)
    analyzer = EnhancedThermalAnalyzer(net=net, domain_size=31, device_=device)

    assert math.isclose(analyzer.metrics["boundary_dirichlet_error_nd"], 0.1, rel_tol=1e-2)
    assert abs(analyzer.metrics["boundary_neumann_error_nd"]) < 1e-9
    n_dir = 2 * analyzer.domain_size
    n_neu = 2 * max(analyzer.domain_size - 2, 0)
    expected_boundary_nd = (0.1 * n_dir) / max(n_dir + n_neu, 1)
    assert math.isclose(
        analyzer.metrics["boundary_bc_error_nd"],
        expected_boundary_nd,
        rel_tol=1e-2,
    )
    # Aliases retroativos devem apontar para versão ND.
    assert math.isclose(
        analyzer.metrics["boundary_bc_error"],
        analyzer.metrics["boundary_bc_error_nd"],
        rel_tol=1e-12,
    )
    assert math.isclose(
        analyzer.metrics["boundary_error"],
        analyzer.metrics["boundary_bc_error_nd"],
        rel_tol=1e-12,
    )


def test_analyze_convergence_prefers_total_loss_series():
    history = {
        "adam_loss": [10.0, 5.0, 1.0],
        "total_loss": [10.0, 5.0, 1.0, 0.1],
    }
    stats = analyze_convergence(history)
    assert stats["final_loss"] == 0.1
    assert stats["total_steps"] == 4


def test_assess_publication_readiness_detects_false_convergence():
    history = {
        "total_loss": [1.0, 1e-5],
        "L_PDE": [1.0, 0.8],
        "rar_selected_residual_mean": [],
        "rar_global_residual_rms": [],
    }
    metrics = {
        "relative_l2_error": 1.0e-5,
        "pde_residual_nd_mean": 5.0e-2,
        "pde_residual_nd_max": 2.0e-1,
        "boundary_bc_error_nd": 1.0e-4,
    }
    report = assess_publication_readiness(history, metrics)
    assert report["status"] == "Request Changes"
    assert "false_convergence_absent" in report["failed_checks"]


def test_stable_mape_percent_is_finite_for_reference_near_zero():
    y_true = torch.tensor([0.0, 1.0e-10, -1.0e-10, 1.0e-6], dtype=torch.float64).numpy()
    y_pred = torch.tensor([1.0e-6, 2.0e-10, -2.0e-10, 3.0e-6], dtype=torch.float64).numpy()
    mape = stable_mape_percent(y_true, y_pred, temp_scale=100.0, floor_fraction=1.0e-3)

    assert math.isfinite(mape)
    assert mape >= 0.0
    assert mape < 1.0e3


def test_neumann_metric_excludes_corners(monkeypatch):
    class DummyNet(nn.Module):
        def forward(self, xy: torch.Tensor) -> torch.Tensor:
            return xy[:, 0:1]

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        analytics_module,
        "dT_dy_on",
        make_captured_dT_dy_stub(captured, fill_value=1.0),
    )

    analyzer = EnhancedThermalAnalyzer.__new__(EnhancedThermalAnalyzer)
    analyzer.net = DummyNet().to(device).eval()
    analyzer.device = device
    analyzer.domain_size = 9

    neumann_abs = analyzer._compute_neumann_bc_residual(nondimensional=False)

    assert neumann_abs.shape == (2 * (analyzer.domain_size - 2),)
    x_values = captured["xy"][:, 0].numpy()
    assert bool(np.all(x_values > 0.0))
    assert bool(np.all(x_values < float(Config.LX)))
    assert captured["create_graph"] is False
