"""
Testes da geração de dados de treino e estratégias de amostragem espacial.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import torch

from config import device
from sampling import _sample_rows, create_training_data


def test_create_training_data_returns_consistent_shapes(config_guard):
    config_guard.FDM_NX = 31
    config_guard.FDM_NY = 21
    config_guard.FDM_MAX_ITER = 5000
    config_guard.FDM_TOL = 1e-7
    config_guard.DEFAULT_N_INTERIOR = 120
    config_guard.DEFAULT_N_BOUNDARY = 25
    config_guard.DEFAULT_N_DATA = 140
    config_guard.DEFAULT_N_MIDLINE = 33
    config_guard.RIGHT_DIRICHLET_MULTIPLIER = 1.5

    data = create_training_data(
        N_interior=config_guard.DEFAULT_N_INTERIOR,
        N_boundary=config_guard.DEFAULT_N_BOUNDARY,
        N_data=config_guard.DEFAULT_N_DATA,
        device_=device,
    )

    left_count = config_guard.DEFAULT_N_BOUNDARY
    right_count = max(1, int(round(left_count * config_guard.RIGHT_DIRICHLET_MULTIPLIER)))
    assert data.X_f.shape == (120, 2)
    assert data.X_dir.shape == (left_count + right_count, 2)
    assert data.T_dir_target.shape == (left_count + right_count, 1)
    assert data.X_bottom.shape == (25, 2)
    assert data.X_top.shape == (25, 2)
    assert data.X_data.shape == (140, 2)
    assert data.T_data_target.shape == (140, 1)
    assert data.X_midline.shape == (33, 2)
    assert data.fdm_field.shape == (21, 31)
    assert data.fdm_iterations > 0

    atol = 1e-10
    assert torch.allclose(
        data.X_dir[:left_count, 0],
        torch.zeros(left_count, dtype=data.X_dir.dtype, device=device),
        atol=atol,
    )
    assert torch.allclose(
        data.X_dir[left_count:, 0],
        torch.full((right_count,), config_guard.LX, dtype=data.X_dir.dtype, device=device),
        atol=atol,
    )
    assert torch.allclose(data.X_bottom[:, 1], torch.zeros(25, dtype=data.X_bottom.dtype, device=device), atol=atol)
    assert torch.allclose(
        data.X_top[:, 1],
        torch.full((25,), config_guard.LY, dtype=data.X_top.dtype, device=device),
        atol=atol,
    )


def test_sample_rows_rejects_non_positive_requests():
    rows = torch.rand(5, 2)
    vals = torch.rand(5, 1)
    gen = torch.Generator(device="cpu").manual_seed(123)

    try:
        _sample_rows(rows, vals, 0, generator=gen)
    except ValueError as exc:
        assert "positivo" in str(exc)
    else:
        raise AssertionError("Era esperado ValueError para n_rows <= 0.")


def test_create_training_data_supports_lhs_hybrid_and_sobol(config_guard):
    config_guard.FDM_NX = 25
    config_guard.FDM_NY = 25
    config_guard.FDM_MAX_ITER = 5000
    config_guard.FDM_TOL = 1e-7

    lhs_data = create_training_data(
        N_interior=90,
        N_boundary=20,
        N_data=120,
        interior_strategy="lhs",
        device_=device,
    )
    assert lhs_data.X_f.shape == (90, 2)
    assert torch.all(lhs_data.X_f[:, 0] >= 0.0)
    assert torch.all(lhs_data.X_f[:, 0] <= config_guard.LX)
    assert torch.all(lhs_data.X_f[:, 1] >= 0.0)
    assert torch.all(lhs_data.X_f[:, 1] <= config_guard.LY)

    hybrid_data = create_training_data(
        N_interior=91,
        N_boundary=20,
        N_data=120,
        interior_strategy="hybrid",
        device_=device,
    )
    assert hybrid_data.X_f.shape == (91, 2)

    sobol_data = create_training_data(
        N_interior=92,
        N_boundary=20,
        N_data=120,
        interior_strategy="sobol",
        device_=device,
    )
    assert sobol_data.X_f.shape == (92, 2)
    assert torch.all(sobol_data.X_f[:, 0] >= 0.0)
    assert torch.all(sobol_data.X_f[:, 0] <= config_guard.LX)
    assert torch.all(sobol_data.X_f[:, 1] >= 0.0)
    assert torch.all(sobol_data.X_f[:, 1] <= config_guard.LY)


def test_create_training_data_rejects_invalid_strategy():
    try:
        create_training_data(
            N_interior=50,
            N_boundary=20,
            N_data=60,
            interior_strategy="invalid",
            device_=device,
        )
    except ValueError as exc:
        assert "interior_strategy inválida" in str(exc)
    else:
        raise AssertionError("Era esperado ValueError para interior_strategy inválida.")


def test_create_training_data_right_importance_sampling_focuses_x_near_lx(config_guard):
    config_guard.FDM_NX = 25
    config_guard.FDM_NY = 25
    config_guard.FDM_MAX_ITER = 5000
    config_guard.FDM_TOL = 1e-7
    config_guard.ENABLE_LEFT_IMPORTANCE_SAMPLING = False
    config_guard.ENABLE_RIGHT_IMPORTANCE_SAMPLING = True
    config_guard.RIGHT_IMPORTANCE_FRACTION = 1.0
    config_guard.RIGHT_IMPORTANCE_BAND_FRACTION = 0.1

    data = create_training_data(
        N_interior=120,
        N_boundary=20,
        N_data=120,
        interior_strategy="lhs",
        device_=device,
    )

    x_vals = data.X_f[:, 0]
    right_threshold = config_guard.LX * (1.0 - config_guard.RIGHT_IMPORTANCE_BAND_FRACTION)
    assert torch.all(x_vals >= right_threshold - 1e-10)
    assert torch.all(x_vals <= config_guard.LX + 1e-10)


def test_create_training_data_left_importance_sampling_focuses_x_near_zero(config_guard):
    config_guard.FDM_NX = 25
    config_guard.FDM_NY = 25
    config_guard.FDM_MAX_ITER = 5000
    config_guard.FDM_TOL = 1e-7
    config_guard.ENABLE_LEFT_IMPORTANCE_SAMPLING = True
    config_guard.LEFT_IMPORTANCE_FRACTION = 1.0
    config_guard.LEFT_IMPORTANCE_BAND_FRACTION = 0.1
    config_guard.ENABLE_RIGHT_IMPORTANCE_SAMPLING = False

    data = create_training_data(
        N_interior=120,
        N_boundary=20,
        N_data=120,
        interior_strategy="lhs",
        device_=device,
    )

    x_vals = data.X_f[:, 0]
    left_threshold = config_guard.LX * config_guard.LEFT_IMPORTANCE_BAND_FRACTION
    assert torch.all(x_vals >= -1e-10)
    assert torch.all(x_vals <= left_threshold + 1e-10)


def test_neumann_sets_exclude_corners(config_guard):
    config_guard.FDM_NX = 31
    config_guard.FDM_NY = 21
    config_guard.FDM_MAX_ITER = 5000
    config_guard.FDM_TOL = 1e-7

    data = create_training_data(
        N_interior=120,
        N_boundary=40,
        N_data=140,
        interior_strategy="hybrid",
        device_=device,
    )

    atol = 1e-10
    has_left_corner_bottom = torch.any(
        torch.isclose(data.X_bottom[:, 0], torch.tensor(0.0, dtype=data.X_bottom.dtype, device=device), atol=atol)
    )
    has_right_corner_bottom = torch.any(
        torch.isclose(
            data.X_bottom[:, 0],
            torch.tensor(float(config_guard.LX), dtype=data.X_bottom.dtype, device=device),
            atol=atol,
        )
    )
    has_left_corner_top = torch.any(
        torch.isclose(data.X_top[:, 0], torch.tensor(0.0, dtype=data.X_top.dtype, device=device), atol=atol)
    )
    has_right_corner_top = torch.any(
        torch.isclose(
            data.X_top[:, 0],
            torch.tensor(float(config_guard.LX), dtype=data.X_top.dtype, device=device),
            atol=atol,
        )
    )

    assert not bool(has_left_corner_bottom)
    assert not bool(has_right_corner_bottom)
    assert not bool(has_left_corner_top)
    assert not bool(has_right_corner_top)

