"""
Testes de geração de plots com foco em consistência física de contorno.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import plotting as plotting_module
from _autograd_capture import make_captured_dT_dy_stub
from config import Config, device


class _DummyNet(nn.Module):
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return xy[:, 0:1]


class _DummyAnalyzer:
    def __init__(self, domain_size: int) -> None:
        self.domain_size = int(domain_size)
        self.device = device
        self.net = _DummyNet().to(device).eval()
        shape = (self.domain_size, self.domain_size)
        self.T_ref = np.zeros(shape, dtype=np.float64)
        self.T_pred = np.zeros(shape, dtype=np.float64)
        self.pde_residual = np.zeros(shape, dtype=np.float64)


def test_plot_physics_consistency_uses_autograd_neumann_and_excludes_corners(
    monkeypatch, tmp_path
):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        plotting_module,
        "dT_dy_on",
        make_captured_dT_dy_stub(captured, fill_value=0.0),
    )

    analyzer = _DummyAnalyzer(domain_size=8)
    plotter = plotting_module.PublicationPlotter(
        analyzer=analyzer,
        history={},
        output_dir=str(tmp_path),
    )
    plotter._plot_physics_consistency()

    assert "xy" in captured
    assert captured["create_graph"] is False
    xy_tensor = captured["xy"]
    assert isinstance(xy_tensor, torch.Tensor)
    x_values = xy_tensor[:, 0].numpy()
    assert float(np.min(x_values)) > 0.0
    assert float(np.max(x_values)) < float(Config.LX)
    assert (tmp_path / "physicsConsistency.png").exists()
