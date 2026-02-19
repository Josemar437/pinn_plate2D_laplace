"""
Helpers de teste para capturar chamadas de operadores autograd.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


def make_captured_dT_dy_stub(
    captured: dict[str, Any],
    *,
    fill_value: float,
) -> Callable[..., torch.Tensor]:
    """
    Cria stub para dT_dy_on que registra entrada e flag create_graph.

    O retorno usa um valor constante para simplificar validação de chamadas sem
    depender do comportamento de um modelo real.
    """

    def fake_dT_dy_on(
        xy: torch.Tensor, net: nn.Module, *, create_graph: bool = True
    ) -> torch.Tensor:
        captured["xy"] = xy.detach().cpu()
        captured["create_graph"] = bool(create_graph)
        return torch.full(
            (xy.shape[0], 1),
            float(fill_value),
            dtype=xy.dtype,
            device=xy.device,
        )

    return fake_dT_dy_on

