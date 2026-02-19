# -*- coding: utf-8 -*-
"""
Solver de referência por diferenças finitas para o problema de Laplace em 2D.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

from typing import Tuple

import torch


def _apply_mixed_boundaries(field: torch.Tensor, boundary_field: torch.Tensor) -> None:
    """
    Atualiza bordas para o problema misto:

    - Dirichlet em x=0 e x=Lx (esquerda/direita).
    - Neumann homogêneo em y=0 e y=Ly (topo/base), imposto por espelhamento.
    """
    field[:, 0] = boundary_field[:, 0]
    field[:, -1] = boundary_field[:, -1]

    if field.shape[1] > 2:
        field[0, 1:-1] = field[1, 1:-1]
        field[-1, 1:-1] = field[-2, 1:-1]

    field[0, 0] = boundary_field[0, 0]
    field[-1, 0] = boundary_field[-1, 0]
    field[0, -1] = boundary_field[0, -1]
    field[-1, -1] = boundary_field[-1, -1]


def solve_laplace_dirichlet(
    boundary_field: torch.Tensor,
    *,
    lx: float,
    ly: float,
    tol: float = 1e-6,
    max_iter: int = 20000,
    omega: float = 1.7,
    initial_guess: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, int]:
    """
    Resolve ∇²T = 0 em malha retangular com condições mistas.

    O solver aplica:
    - Dirichlet em x=0 e x=Lx;
    - Neumann homogêneo em y=0 e y=Ly.

    O método usa SOR com varredura Red-Black e retorna o campo convergido e o
    número de iterações executadas.
    """
    if boundary_field.ndim != 2:
        raise ValueError("boundary_field deve ser um tensor 2D [ny, nx].")

    ny, nx = boundary_field.shape
    if nx < 3 or ny < 3:
        raise ValueError("A malha deve ter pelo menos nx >= 3 e ny >= 3.")

    hx = float(lx) / float(nx - 1)
    hy = float(ly) / float(ny - 1)
    hx2 = hx * hx
    hy2 = hy * hy
    denom = 2.0 * (hx2 + hy2)
    omega = float(omega)
    if not (0.0 < omega < 2.0):
        raise ValueError("omega deve satisfazer 0 < omega < 2 para convergência SOR.")

    if initial_guess is None:
        t = boundary_field.clone()
    else:
        if initial_guess.shape != boundary_field.shape:
            raise ValueError(
                "initial_guess deve ter a mesma forma que boundary_field."
            )
        t = initial_guess.clone()
        _apply_mixed_boundaries(t, boundary_field)

    _apply_mixed_boundaries(t, boundary_field)

    ii = torch.arange(1, ny - 1, device=t.device).view(-1, 1)
    jj = torch.arange(1, nx - 1, device=t.device).view(1, -1)
    red_mask = (ii + jj) % 2 == 0
    black_mask = ~red_mask

    for it in range(1, int(max_iter) + 1):
        max_update = 0.0

        for mask in (red_mask, black_mask):
            interior_view = t[1:-1, 1:-1]
            old_vals = interior_view[mask].clone()

            gs_update = (
                (t[1:-1, 2:] + t[1:-1, :-2]) * hy2
                + (t[2:, 1:-1] + t[:-2, 1:-1]) * hx2
            ) / denom
            new_vals = (1.0 - omega) * old_vals + omega * gs_update[mask]
            interior_view[mask] = new_vals

            if old_vals.numel() > 0:
                step_update = torch.max(torch.abs(new_vals - old_vals)).item()
                max_update = max(max_update, float(step_update))

        _apply_mixed_boundaries(t, boundary_field)
        if max_update < float(tol):
            return t, it

    return t, int(max_iter)


def build_linear_boundary_field(
    *,
    nx: int,
    ny: int,
    t_left: float,
    t_right: float,
    lx: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Monta o campo de contorno para problema misto.

    A função fixa Dirichlet à esquerda/direita e usa um perfil linear em
    topo/base apenas como condição inicial para as bordas Neumann.
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx e ny devem ser maiores ou iguais a 2.")
    if lx <= 0.0:
        raise ValueError("lx deve ser positivo.")

    resolved_device = device or torch.device("cpu")
    resolved_dtype = dtype or torch.get_default_dtype()
    x_coords = torch.linspace(
        0.0, float(lx), int(nx), device=resolved_device, dtype=resolved_dtype
    )
    top_bottom_profile = float(t_left) + (float(t_right) - float(t_left)) * (
        x_coords / float(lx)
    )

    boundary_field = torch.empty(
        (int(ny), int(nx)), device=resolved_device, dtype=resolved_dtype
    )
    boundary_field[:, 0] = float(t_left)
    boundary_field[:, -1] = float(t_right)
    boundary_field[0, :] = top_bottom_profile.clone()
    boundary_field[-1, :] = top_bottom_profile.clone()
    return boundary_field


def generate_reference_solution(
    *,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    t_left: float,
    t_right: float,
    tol: float = 1e-8,
    max_iter: int = 20000,
    omega: float = 1.7,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, int]:
    """Gera campo térmico de referência a partir do contorno e do solver SOR."""
    boundary_field = build_linear_boundary_field(
        nx=nx,
        ny=ny,
        t_left=t_left,
        t_right=t_right,
        lx=lx,
        device=device,
        dtype=dtype,
    )
    return solve_laplace_dirichlet(
        boundary_field=boundary_field,
        lx=lx,
        ly=ly,
        tol=tol,
        max_iter=max_iter,
        omega=omega,
    )
