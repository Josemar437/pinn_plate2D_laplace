"""
Testes unitários do solver FDM e validações de contorno para Laplace 2D.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import torch

from fdm import (
    build_linear_boundary_field,
    generate_reference_solution,
    solve_laplace_dirichlet,
)


def test_build_linear_boundary_field_sets_expected_edges():
    nx, ny = 9, 7
    field = build_linear_boundary_field(
        nx=nx, ny=ny, t_left=200.0, t_right=100.0, lx=1.0
    )

    assert field.shape == (ny, nx)
    assert torch.allclose(field[:, 0], torch.full((ny,), 200.0, dtype=field.dtype))
    assert torch.allclose(field[:, -1], torch.full((ny,), 100.0, dtype=field.dtype))
    assert torch.isclose(field[0, nx // 2], field[-1, nx // 2])


def test_generate_reference_solution_converges_to_linear_profile():
    nx, ny = 31, 31
    solution, iters = generate_reference_solution(
        nx=nx,
        ny=ny,
        lx=1.0,
        ly=1.0,
        t_left=200.0,
        t_right=100.0,
        tol=1e-8,
        max_iter=20000,
        omega=1.7,
    )

    assert solution.shape == (ny, nx)
    assert iters > 0
    assert torch.max(torch.std(solution, dim=0)).item() < 5e-4


def test_solve_laplace_dirichlet_validates_omega():
    boundary = build_linear_boundary_field(nx=11, ny=11, t_left=200.0, t_right=100.0, lx=1.0)
    try:
        solve_laplace_dirichlet(boundary, lx=1.0, ly=1.0, omega=2.1)
    except ValueError as exc:
        assert "omega" in str(exc)
    else:
        raise AssertionError("Era esperado ValueError para omega inválido.")

