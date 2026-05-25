"""
Data sampling strategies for PINN Thermal 2D models.
Supports Uniform, Sobol, and Latin Hypercube sampling for space-filling designs.
"""
import torch
import numpy as np
from typing import Tuple
from config import Config, device, logger

class SamplingStrategy:
    """Provides high-density and space-filling sampling designs for coordinate-based PINNs."""
    
    @staticmethod
    def sample_interior(N: int, device_: torch.device = device, 
                       strategy: str = "uniform") -> torch.Tensor:
        """Samples collocation points within the rectangular domain."""
        if strategy == "uniform":
            scale = torch.tensor([Config.LX, Config.LY], device=device_)
            return torch.rand((N, 2), device=device_) * scale
        
        elif strategy == "sobol":
            # Sobol sequence for quasi-random low-discrepancy sampling
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=2, scramble=True)
            points = sobol.draw(N).to(device_)
            scale = torch.tensor([Config.LX, Config.LY], device=device_)
            return points * scale
        
        elif strategy == "latin_hypercube":
            # Latin Hypercube Sampling (LHS) for improved variance reduction
            points = torch.rand((N, 2), device=device_)
            for i in range(2):
                points[:, i] = torch.randperm(N, device=device_).float() / (N - 1)
            scale = torch.tensor([Config.LX, Config.LY], device=device_)
            return points * scale
        else:
            raise ValueError(f"Estratégia de amostragem desconhecida: {strategy}")

    @staticmethod
    def sample_dirichlet(N_each: int, device_: torch.device = device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples manifolds for Dirichlet boundary conditions (Fixed Temperature)."""
        # Left (x=0) and Right (x=Lx) boundaries
        yL = torch.rand(N_each, 1, device=device_) * Config.LY
        xL = torch.zeros_like(yL)
        yR = torch.rand(N_each, 1, device=device_) * Config.LY
        xR = torch.full_like(yR, Config.LX)

        X_dir = torch.cat([torch.cat([xL, yL], 1), torch.cat([xR, yR], 1)], 0)
        
        T_L = torch.full((N_each, 1), Config.T_LEFT, device=device_)
        T_R = torch.full((N_each, 1), Config.T_RIGHT, device=device_)
        T_dir_target = torch.cat([T_L, T_R], 0)
        
        return X_dir, T_dir_target

    @staticmethod
    def sample_neumann(N_each: int, device_: torch.device = device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples manifolds for Neumann boundary conditions (Adiabatic)."""
        # Bottom (y=0) and Top (y=Ly) boundaries
        xB = torch.rand(N_each, 1, device=device_) * Config.LX
        yB = torch.zeros_like(xB)
        xT = torch.rand(N_each, 1, device=device_) * Config.LX
        yT = torch.full_like(xT, Config.LY)

        return torch.cat([xB, yB], 1), torch.cat([xT, yT], 1)

def create_training_data(N_interior: int = Config.DEFAULT_N_INTERIOR,
                        N_boundary: int = Config.DEFAULT_N_BOUNDARY,
                        interior_strategy: str = "sobol") -> Tuple[torch.Tensor, ...]:
    """Generates a complete dataset for steady-state thermal analysis."""
    sampling = SamplingStrategy()
    
    X_f = sampling.sample_interior(N_interior, device, strategy=interior_strategy)
    X_dir, T_dir_target = sampling.sample_dirichlet(N_boundary, device)
    X_bottom, X_top = sampling.sample_neumann(N_boundary, device)
    
    return X_f, X_dir, T_dir_target, X_bottom, X_top
