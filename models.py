"""
Neural network architectures for PINN Thermal 2D models.
Implements MLP architectures with specialized weight initialization for heat transfer.
"""
import torch
import torch.nn as nn
from typing import List, Optional
from config import Config, logger

class EnhancedThermalPINN(nn.Module):
    """
    MLP architecture for thermal PINN solvers with optional hard constraint support.
    Includes robust initialization and modular layering.
    Hard constraints enforce exact Dirichlet BCs on lateral boundaries via network design.
    """
    
    def __init__(self, layers: Optional[List[int]] = None, 
                 activation: str = 'tanh',
                 hard_constraint_bc: bool = True) -> None:
        super().__init__()
        
        if layers is None:
            layers = [2, 64, 64, 1]
            
        self.activation_func = nn.Tanh() if activation == 'tanh' else nn.SiLU()
        self.hard_constraint_bc = hard_constraint_bc
        
        # Construct layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            
        self._init_weights()
        
        # Physical bias initialization (Mean temperature)
        mean_temp = (Config.T_LEFT + Config.T_RIGHT) / 2.0
        with torch.no_grad():
            self.layers[-1].bias.fill_(mean_temp)

    def _init_weights(self) -> None:
        """Xavier/Glorot weight initialization for Tanh networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Forward pass through the coordinate-to-temperature mapping."""
        x = xy
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_func(layer(x))
        return self.layers[-1](x)
    
    def forward_with_hard_constraint(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hard constraint enforcement for lateral Dirichlet BCs.
        Ensures exact satisfaction of T(x=0)=T_LEFT and T(x=LX)=T_RIGHT.
        Interior and top/bottom use unconstrained network output.
        
        Implementation:
            T_constrained = T_boundary + penalty(x) * (T_unconstrained - T_boundary)
            where:
            - T_boundary = T_LEFT + (T_RIGHT - T_LEFT) * (x / LX)
            - penalty(x) = x_norm * (1 - x_norm)  [zero at boundaries, max at center]
        """
        if not self.hard_constraint_bc:
            return self.forward(xy)
        
        # Extract coordinates
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        
        # Get unconstrained network output
        T_unconstrained = self.forward(xy)
        
        # Normalized x coordinate [0, 1]
        x_norm = x / Config.LX
        
        # Boundary values (linear interpolation between T_LEFT and T_RIGHT)
        T_boundary = Config.T_LEFT + (Config.T_RIGHT - Config.T_LEFT) * x_norm
        
        # Distance from boundaries (bell-shaped penalty function)
        # Zero at x=0 and x=LX (boundaries), maximum at x=LX/2 (interior)
        penalty = x_norm * (1.0 - x_norm)
        
        # Apply hard constraint: interpolate from boundary to unconstrained interior
        T_constrained = T_boundary + penalty * (T_unconstrained - T_boundary)
        
        return T_constrained

def create_model(model_type: str = "enhanced", hard_constraint_bc: bool = True, **kwargs) -> nn.Module:
    """Factory function for model creation."""
    if model_type == "enhanced":
        return EnhancedThermalPINN(hard_constraint_bc=hard_constraint_bc, **kwargs)
    else:
        # Fallback to standard Sequential for debugging
        layers = kwargs.get('layers', [2, 64, 64, 1])
        seq_layers = []
        for i in range(len(layers) - 1):
            seq_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                seq_layers.append(nn.Tanh())
        return nn.Sequential(*seq_layers)

def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: nn.Module) -> None:
    """Logs the model architecture and parameter count."""
    total_params = count_parameters(model)
    logger.info(f"Model Architecture: {model.__class__.__name__}")
    logger.info(f"Trainable Parameters: {total_params:,}")
