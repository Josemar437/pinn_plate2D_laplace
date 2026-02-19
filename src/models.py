"""
Arquiteturas de rede neural para aproximação do campo de temperatura em 2D.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

try:
    from .config import Config, logger
except ImportError:  
    from config import Config, logger


class AdaptiveTanh(nn.Module):
    """
    Ativação tanh(a*x) com inclinação treinável e limitada inferiormente.

    A parametrização em softplus mantém a inclinação sempre positiva.
    """

    def __init__(self, init_slope: float, min_slope: float) -> None:
        """Inicializa parâmetros da ativação com inclinação mínima garantida."""
        super().__init__()
        self.min_slope = float(max(min_slope, 1.0e-6))
        init_slope = float(max(init_slope, self.min_slope + 1.0e-6))
        raw_init = math.log(math.expm1(init_slope - self.min_slope))
        self.raw_slope = nn.Parameter(torch.tensor(raw_init, dtype=torch.get_default_dtype()))

    def slope(self) -> torch.Tensor:
        """Retorna inclinação positiva efetiva."""
        return self.min_slope + torch.nn.functional.softplus(self.raw_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica transformação adaptativa tanh(a*x)."""
        return torch.tanh(self.slope() * x)


def _build_activation(activation: str) -> nn.Module:
    """Resolve nome de ativação para módulo PyTorch."""
    act_name = str(activation).strip().lower()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name in {"adaptive_tanh", "adapt_tanh", "atanh"}:
        return AdaptiveTanh(
            init_slope=float(Config.ADAPTIVE_TANH_INIT),
            min_slope=float(Config.ADAPTIVE_TANH_MIN_SLOPE),
        )
    if act_name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(
        f"activation inválida: {activation!r}. Use 'tanh', 'adaptive_tanh' ou 'silu'."
    )


class EnhancedThermalPINN(nn.Module):
    """MLP para mapear coordenadas (x, y) em temperatura T(x, y)."""

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        activation: str | None = None,
    ) -> None:
        """Inicializa camadas lineares, ativação e viés físico da saída."""
        super().__init__()

        if layers is None:
            layers = [2, 64, 64, 1]
        self.activation_name = (
            str(activation).strip().lower()
            if activation is not None
            else str(Config.DEFAULT_ACTIVATION).strip().lower()
        )
        self.input_normalization = str(
            getattr(Config, "INPUT_NORMALIZATION", "minus_one_one")
        ).strip().lower()
        valid_norm_modes = {"zero_one", "minus_one_one", "none"}
        if self.input_normalization not in valid_norm_modes:
            raise ValueError(
                f"INPUT_NORMALIZATION inválido: {self.input_normalization!r}. "
                f"Use uma de {sorted(valid_norm_modes)}."
            )
        self.activation_func = _build_activation(self.activation_name)
        self.enable_hard_dirichlet_constraint = bool(
            getattr(Config, "ENABLE_HARD_DIRICHLET_CONSTRAINT", False)
        )

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        self._init_weights()

        mean_temp = (Config.T_LEFT + Config.T_RIGHT) / 2.0
        coord_scale = torch.tensor(
            [
                max(float(Config.LX), 1.0e-12),
                max(float(Config.LY), 1.0e-12),
            ],
            dtype=torch.get_default_dtype(),
        )
        coord_center = 0.5 * coord_scale
        self.register_buffer("_coord_scale", coord_scale)
        self.register_buffer("_coord_center", coord_center)
        self.register_buffer(
            "_boundary_temps",
            torch.tensor(
                [float(Config.T_LEFT), float(Config.T_RIGHT)],
                dtype=torch.get_default_dtype(),
            ),
        )
        with torch.no_grad():
            self.layers[-1].bias.fill_(mean_temp)

    def _apply_hard_dirichlet(self, xy: torch.Tensor, nn_out: torch.Tensor) -> torch.Tensor:
        """
        Impõe Dirichlet por construção para eliminar erro de contorno em x=0 e x=Lx.

        Transformação:
            T(x,y) = T_lin(x) + x*(Lx-x)*N(x,y)
        onde:
            T_lin(x) = T_left + (T_right - T_left) * (x/Lx)
        """
        lx = torch.clamp(self._coord_scale[0], min=1.0e-12)
        x_coord = xy[:, 0:1]
        x_hat = x_coord / lx
        t_left = self._boundary_temps[0]
        t_right = self._boundary_temps[1]
        linear_profile = t_left + (t_right - t_left) * x_hat
        envelope = x_coord * (lx - x_coord)
        return linear_profile + envelope * nn_out

    def _init_weights(self) -> None:
        """Aplica inicialização Xavier em pesos e zera vieses."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Executa propagação direta da rede para o lote de coordenadas."""
        if self.input_normalization == "zero_one":
            x = xy / self._coord_scale
        elif self.input_normalization == "minus_one_one":
            x = 2.0 * (xy - self._coord_center) / self._coord_scale
        else:
            x = xy
        for layer in self.layers[:-1]:
            x = self.activation_func(layer(x))
        raw_out = self.layers[-1](x)
        if self.enable_hard_dirichlet_constraint:
            return self._apply_hard_dirichlet(xy, raw_out)
        return raw_out


def create_model(model_type: str = "enhanced", **kwargs) -> nn.Module:
    """Cria e retorna a arquitetura de modelo solicitada."""
    if model_type == "enhanced":
        return EnhancedThermalPINN(**kwargs)

    layers = kwargs.get("layers", [2, 64, 64, 1])
    activation_name = kwargs.get("activation", "tanh")
    seq_layers: list[nn.Module] = []
    for i in range(len(layers) - 1):
        seq_layers.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            seq_layers.append(_build_activation(activation_name))
    return nn.Sequential(*seq_layers)


def count_parameters(model: nn.Module) -> int:
    """Conta parâmetros treináveis da arquitetura."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module) -> None:
    """Registra no logger nome da arquitetura e número de parâmetros."""
    total_params = count_parameters(model)
    logger.info("Model Architecture: %s", model.__class__.__name__)
    logger.info("Trainable Parameters: %s", f"{total_params:,}")

