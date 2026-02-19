"""
Testes de criação de modelos e validação das ativações configuráveis da rede.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import torch

from models import AdaptiveTanh, create_model


def test_create_model_supports_adaptive_tanh():
    model = create_model(
        model_type="enhanced",
        layers=[2, 16, 16, 1],
        activation="adaptive_tanh",
    )
    assert isinstance(model.activation_func, AdaptiveTanh)

    x = torch.rand(8, 2, dtype=torch.get_default_dtype())
    y = model(x)
    assert y.shape == (8, 1)

    slopes = [
        param for name, param in model.named_parameters() if "raw_slope" in name
    ]
    assert len(slopes) == 1
    assert slopes[0].requires_grad


def test_create_model_rejects_invalid_activation():
    try:
        create_model(model_type="enhanced", activation="relu")
    except ValueError as exc:
        assert "activation inválida" in str(exc)
    else:
        raise AssertionError("Era esperado ValueError para activation inválida.")


def test_create_model_supports_input_normalization_modes(config_guard):
    config_guard.INPUT_NORMALIZATION = "zero_one"
    model_zero_one = create_model(model_type="enhanced", layers=[2, 8, 1], activation="tanh")
    out_zero_one = model_zero_one(torch.rand(4, 2, dtype=torch.get_default_dtype()))
    assert out_zero_one.shape == (4, 1)

    config_guard.INPUT_NORMALIZATION = "minus_one_one"
    model_minus = create_model(model_type="enhanced", layers=[2, 8, 1], activation="tanh")
    out_minus = model_minus(torch.rand(4, 2, dtype=torch.get_default_dtype()))
    assert out_minus.shape == (4, 1)

    config_guard.INPUT_NORMALIZATION = "none"
    model_none = create_model(model_type="enhanced", layers=[2, 8, 1], activation="tanh")
    out_none = model_none(torch.rand(4, 2, dtype=torch.get_default_dtype()))
    assert out_none.shape == (4, 1)


def test_create_model_rejects_invalid_input_normalization(config_guard):
    config_guard.INPUT_NORMALIZATION = "invalid_mode"
    try:
        create_model(model_type="enhanced", layers=[2, 8, 1], activation="tanh")
    except ValueError as exc:
        assert "INPUT_NORMALIZATION inválido" in str(exc)
    else:
        raise AssertionError("Era esperado ValueError para INPUT_NORMALIZATION inválido.")


def test_default_input_normalization_is_minus_one_one(config_guard):
    config_guard.INPUT_NORMALIZATION = "minus_one_one"
    model = create_model(model_type="enhanced", layers=[2, 8, 1], activation="tanh")
    assert model.input_normalization == "minus_one_one"


def test_hard_dirichlet_constraint_matches_boundary_by_construction(config_guard):
    config_guard.ENABLE_HARD_DIRICHLET_CONSTRAINT = True
    config_guard.LX = 1.0
    config_guard.T_LEFT = 200.0
    config_guard.T_RIGHT = 100.0
    model = create_model(model_type="enhanced", layers=[2, 16, 16, 1], activation="tanh")
    assert model.enable_hard_dirichlet_constraint is True

    y = torch.rand((32, 1), dtype=torch.get_default_dtype())
    x_left = torch.zeros_like(y)
    x_right = torch.full_like(y, float(config_guard.LX))
    pts_left = torch.cat([x_left, y], dim=1)
    pts_right = torch.cat([x_right, y], dim=1)

    t_left_pred = model(pts_left)
    t_right_pred = model(pts_right)

    assert torch.allclose(
        t_left_pred,
        torch.full_like(t_left_pred, float(config_guard.T_LEFT)),
        atol=1.0e-10,
        rtol=0.0,
    )
    assert torch.allclose(
        t_right_pred,
        torch.full_like(t_right_pred, float(config_guard.T_RIGHT)),
        atol=1.0e-10,
        rtol=0.0,
    )
