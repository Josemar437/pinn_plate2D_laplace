"""
Operadores diferenciais com autograd para cálculo de derivadas físicas da PINN.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import torch
import torch.autograd as autograd
import torch.nn as nn

try:
    from .config import residual_nondim_scale
except ImportError:  
    from config import residual_nondim_scale


def _first_gradients(
    xy: torch.Tensor,
    net: nn.Module,
    *,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Retorna (xy_req, dT_dxy) para reutilização em operadores diferenciais.

    A entrada é destacada do grafo anterior para evitar acúmulo de histórico
    entre iterações de treino.
    """
    xy_req = xy.detach().requires_grad_(True)
    temperature = net(xy_req)
    dT_dxy = autograd.grad(
        temperature,
        xy_req,
        torch.ones_like(temperature),
        create_graph=bool(create_graph),
    )[0]
    return xy_req, dT_dxy


def laplacian_T(
    xy: torch.Tensor,
    net: nn.Module,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Calcula o Laplaciano d2T/dx2 + d2T/dy2 para um lote de pontos.

    Nota de custo: o primeiro gradiente precisa de create_graph=True para
    habilitar a segunda derivada. O parâmetro create_graph controla apenas se
    o resultado final permanece diferenciável para fases de treino.
    """
    xy_req, dT_dxy = _first_gradients(xy, net, create_graph=True)

    need_d2x = bool(dT_dxy[:, 0:1].requires_grad)
    need_d2y = bool(dT_dxy[:, 1:2].requires_grad)

    if need_d2x:
        d2Tdx2_full = autograd.grad(
            dT_dxy[:, 0:1],
            xy_req,
            torch.ones_like(dT_dxy[:, 0:1]),
            create_graph=bool(create_graph),
            retain_graph=need_d2y,
            allow_unused=True,
        )[0]
    else:
        d2Tdx2_full = None

    if need_d2y:
        d2Tdy2_full = autograd.grad(
            dT_dxy[:, 1:2],
            xy_req,
            torch.ones_like(dT_dxy[:, 1:2]),
            create_graph=bool(create_graph),
            allow_unused=True,
        )[0]
    else:
        d2Tdy2_full = None

    if d2Tdx2_full is None:
        d2Tdx2 = torch.zeros_like(dT_dxy[:, 0:1])
    else:
        d2Tdx2 = d2Tdx2_full[:, 0:1]

    if d2Tdy2_full is None:
        d2Tdy2 = torch.zeros_like(dT_dxy[:, 1:2])
    else:
        d2Tdy2 = d2Tdy2_full[:, 1:2]

    return d2Tdx2 + d2Tdy2


def residual_scale(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Retorna fator L_ref^2 / DeltaT_ref para não-dimensionalizar o residual.

    A implementação delega para config.residual_nondim_scale para garantir que
    loss e análise usem exatamente a mesma convenção de escala.
    """
    return residual_nondim_scale(device=device, dtype=dtype)


def pde_residual(
    xy: torch.Tensor,
    net: nn.Module,
    *,
    nondimensional: bool = True,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Calcula o residual PDE baseado no Laplaciano, dimensional ou não-dimensional.

    Em avaliação/inferência, use create_graph=False para reduzir custo de
    memória, já que não há retropropagação pelos operadores diferenciais.
    """
    residual = laplacian_T(xy, net, create_graph=create_graph)
    if not nondimensional:
        return residual
    scale = residual_scale(device=residual.device, dtype=residual.dtype)
    return residual * scale


def dT_dy_on(xy: torch.Tensor, net: nn.Module, *, create_graph: bool = True) -> torch.Tensor:
    """Calcula dT/dy nos pontos de contorno informados."""
    _, dT_dxy = _first_gradients(xy, net, create_graph=create_graph)
    return dT_dxy[:, 1:2]


def dT_dx_on(xy: torch.Tensor, net: nn.Module, *, create_graph: bool = True) -> torch.Tensor:
    """Calcula dT/dx nos pontos de contorno informados."""
    _, dT_dxy = _first_gradients(xy, net, create_graph=create_graph)
    return dT_dxy[:, 0:1]


def d2T_dx2_on(
    xy: torch.Tensor,
    net: nn.Module,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Calcula d2T/dx2 nos pontos informados.

    Assim como no Laplaciano, o primeiro gradiente permanece com grafo ativo
    para permitir a segunda derivada; create_graph controla o grafo final.
    """
    xy_req, dT_dxy = _first_gradients(xy, net, create_graph=True)
    dTdx = dT_dxy[:, 0:1]
    if not dTdx.requires_grad:
        return torch.zeros_like(dTdx)
    d2_full = autograd.grad(
        dTdx,
        xy_req,
        torch.ones_like(dTdx),
        create_graph=bool(create_graph),
        allow_unused=True,
    )[0]
    if d2_full is None:
        return torch.zeros_like(dTdx)
    return d2_full[:, 0:1]
