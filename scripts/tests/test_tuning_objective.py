"""
Testes da função objetivo adimensional usada no tuning com Optuna.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import math

from analytics import pde_residual_l2_reference
from tuning import nondimensional_tuning_objective


def test_nondimensional_objective_is_mesh_invariant():
    rel_l2 = 3.0e-3
    boundary_nd = 4.0e-3
    target_pde_norm = 2.5e-2

    domain_coarse = 32
    domain_fine = 96
    ref_coarse = pde_residual_l2_reference(domain_size=domain_coarse)
    ref_fine = pde_residual_l2_reference(domain_size=domain_fine)

    metrics_coarse = {
        "relative_l2_error": rel_l2,
        "pde_residual_l2": target_pde_norm * ref_coarse,
        "boundary_bc_error_nd": boundary_nd,
    }
    metrics_fine = {
        "relative_l2_error": rel_l2,
        "pde_residual_l2": target_pde_norm * ref_fine,
        "boundary_bc_error_nd": boundary_nd,
    }

    objective_coarse = nondimensional_tuning_objective(
        metrics_coarse,
        domain_size=domain_coarse,
        w1=1.0,
        w2=1.0,
        w3=1.0,
    )
    objective_fine = nondimensional_tuning_objective(
        metrics_fine,
        domain_size=domain_fine,
        w1=1.0,
        w2=1.0,
        w3=1.0,
    )

    assert math.isclose(objective_coarse, objective_fine, rel_tol=1e-12)
