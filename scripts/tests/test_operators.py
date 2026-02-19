"""
Testes dos operadores diferenciais com foco em custo de autograd.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import torch
import torch.nn as nn

from operators import dT_dy_on, pde_residual


class QuadraticParamNet(nn.Module):
    """Campo quadrático simples para validar flags de create_graph."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.get_default_dtype()))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        return self.scale * (x**2 + y**2)


def test_pde_residual_create_graph_flag_controls_requires_grad():
    net = QuadraticParamNet()
    pts = torch.rand((32, 2), dtype=torch.get_default_dtype())

    res_eval = pde_residual(pts, net, nondimensional=False, create_graph=False)
    res_train = pde_residual(pts, net, nondimensional=False, create_graph=True)

    assert not res_eval.requires_grad
    assert res_train.requires_grad


def test_dt_dy_create_graph_flag_controls_requires_grad():
    net = QuadraticParamNet()
    pts = torch.rand((16, 2), dtype=torch.get_default_dtype())

    dy_eval = dT_dy_on(pts, net, create_graph=False)
    dy_train = dT_dy_on(pts, net, create_graph=True)

    assert not dy_eval.requires_grad
    assert dy_train.requires_grad
