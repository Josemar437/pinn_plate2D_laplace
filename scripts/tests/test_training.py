"""
Testes das rotinas de treinamento híbrido (Adam + L-BFGS + polish).

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
from models import create_model
from sampling import TrainingData
import training as training_module
from training import train_pinn_enhanced


def _linear_temperature(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    return float(Config.T_LEFT) + (float(Config.T_RIGHT) - float(Config.T_LEFT)) * (
        x / max(float(Config.LX), 1.0e-12)
    )


def _build_synthetic_training_data() -> TrainingData:
    dtype = torch.get_default_dtype()
    x_f = torch.rand((48, 2), dtype=dtype)

    y_dir = torch.rand((24, 1), dtype=dtype)
    x_left = torch.zeros_like(y_dir)
    x_right = torch.full_like(y_dir, float(Config.LX))
    x_dir = torch.cat([torch.cat([x_left, y_dir], dim=1), torch.cat([x_right, y_dir], dim=1)], dim=0)
    t_dir = _linear_temperature(x_dir)

    x_bottom = torch.cat(
        [torch.rand((24, 1), dtype=dtype), torch.zeros((24, 1), dtype=dtype)],
        dim=1,
    )
    x_top = torch.cat(
        [torch.rand((24, 1), dtype=dtype), torch.full((24, 1), float(Config.LY), dtype=dtype)],
        dim=1,
    )
    x_data = torch.rand((48, 2), dtype=dtype)
    t_data = _linear_temperature(x_data)

    x_mid = torch.linspace(0.0, float(Config.LX), 24, dtype=dtype).unsqueeze(1)
    y_mid = torch.full_like(x_mid, 0.5 * float(Config.LY))
    x_midline = torch.cat([x_mid, y_mid], dim=1)

    return TrainingData(
        X_f=x_f,
        X_dir=x_dir,
        T_dir_target=t_dir,
        X_bottom=x_bottom,
        X_top=x_top,
        X_data=x_data,
        T_data_target=t_data,
        X_midline=x_midline,
        fdm_field=torch.zeros((4, 4), dtype=dtype),
        fdm_iterations=0,
    )


def test_l_pde_history_is_continuous_across_all_phases(config_guard):
    config_guard.ENABLE_RAR = False
    config_guard.ENABLE_PHYSICS_POLISH = True
    config_guard.PHYSICS_POLISH_ITERS = 1
    config_guard.PHYSICS_POLISH_MAX_ROUNDS = 1
    config_guard.CURVATURE_REG_WEIGHT = 0.0

    model = create_model(model_type="enhanced", layers=[2, 12, 12, 1], activation="tanh")
    data = _build_synthetic_training_data()
    history = train_pinn_enhanced(
        net=model,
        data=data,
        epochs_adam=3,
        epochs_lbfgs=1,
        lr=1.0e-3,
        w_data=0.1,
        use_adaptive_weights=False,
        use_scheduler=False,
        verbose=False,
    )

    assert len(history["total_loss"]) == len(history["optimizer_phase"])
    assert len(history["L_PDE"]) == len(history["total_loss"])
    assert len(history["dirichlet_side_ratio_right_left"]) == len(history["L_PDE"])
    assert "adam" in history["optimizer_phase"]
    assert "lbfgs" in history["optimizer_phase"]
    assert "physics_polish" in history["optimizer_phase"]


def test_rar_candidate_ranking_disables_high_order_graph(monkeypatch):
    class DummyNet(nn.Module):
        def forward(self, xy: torch.Tensor) -> torch.Tensor:
            return xy[:, 0:1] + xy[:, 1:2]

    captured_create_graph: list[bool] = []

    def fake_pde_residual(
        xy: torch.Tensor,
        net: nn.Module,
        *,
        nondimensional: bool = True,
        create_graph: bool = True,
    ) -> torch.Tensor:
        captured_create_graph.append(bool(create_graph))
        return torch.ones((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)

    monkeypatch.setattr(training_module, "pde_residual", fake_pde_residual)
    x_f = torch.rand((32, 2), dtype=torch.get_default_dtype())
    gen = torch.Generator(device="cpu").manual_seed(1234)
    _, replaced, _ = training_module._rar_refresh_collocation(
        net=DummyNet(),
        X_f=x_f,
        replace_fraction=0.25,
        candidate_multiplier=4,
        top_pool_multiplier=2,
        selection_power=0.5,
        corner_fraction=0.5,
        corner_band_fraction=0.1,
        left_fraction=0.0,
        left_band_fraction=0.1,
        right_fraction=0.0,
        right_band_fraction=0.1,
        generator=gen,
    )

    assert replaced > 0
    assert captured_create_graph, "pde_residual deveria ser chamado durante o RAR."
    assert all(flag is False for flag in captured_create_graph)
