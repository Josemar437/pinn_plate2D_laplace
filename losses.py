"""
Loss functions and adaptive weighting strategies for PINN Thermal 2D.
Implementation of multi-objective optimization balancing PDE and Boundary Conditions.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from operators import laplacian_T, dT_dy_on
from config import logger

def enhanced_loss_terms(net: nn.Module, X_f: torch.Tensor, 
                       X_dir: torch.Tensor, T_dir_target: torch.Tensor,
                       X_bottom: torch.Tensor, X_top: torch.Tensor,
                       w_pde: float = 1.0, w_dir: float = 100.0, 
                       w_neu: float = 10.0, hard_constraint_bc: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Calculates all loss components for the PINN solver.
    
    Args:
        net: Neural network model
        X_f: Interior collocation points
        X_dir: Dirichlet boundary points (lateral sides)
        T_dir_target: Target temperature values at Dirichlet boundaries
        X_bottom: Bottom boundary points (Neumann: adiabatic)
        X_top: Top boundary points (Neumann: adiabatic)
        w_pde: Weight for PDE residual loss
        w_dir: Weight for Dirichlet BC loss
        w_neu: Weight for Neumann BC loss
        hard_constraint_bc: If True, lateral Dirichlet BCs are enforced via network architecture.
                          If False, all BCs are enforced via loss terms (soft constraints).
    """
    
    # 1. Physics Residual (Laplace PDE)
    laplacian_res = laplacian_T(X_f, net, hard_constraint_bc=hard_constraint_bc)
    L_PDE = torch.mean(laplacian_res**2)
    
    # 2. Dirichlet Boundary Conditions (Fixed Temperature on lateral sides)
    if hard_constraint_bc:
        # With hard constraints, Dirichlet BC loss is reduced since it's automatically satisfied
        # by network architecture. Still computed for monitoring.
        if hasattr(net, 'forward_with_hard_constraint'):
            T_pred_dir = net.forward_with_hard_constraint(X_dir.detach())
        else:
            T_pred_dir = net(X_dir.detach())
        L_D = torch.mean((T_pred_dir - T_dir_target)**2)
    else:
        # Soft constraints: standard loss enforcement for all boundaries
        T_pred_dir = net(X_dir.detach())
        L_D = torch.mean((T_pred_dir - T_dir_target)**2)
    
    # 3. Neumann Boundary Conditions (Adiabatic / Isolated) - Top and Bottom
    dTdy_bottom = dT_dy_on(X_bottom, net, hard_constraint_bc=hard_constraint_bc)
    dTdy_top = dT_dy_on(X_top, net, hard_constraint_bc=hard_constraint_bc)
    L_N = torch.mean(dTdy_bottom**2) + torch.mean(dTdy_top**2)
    
    # Total Weighted Loss
    L_total = w_pde * L_PDE + w_dir * L_D + w_neu * L_N
    
    loss_dict = {
        'L_PDE': L_PDE,
        'L_D': L_D,
        'L_N': L_N,
        'L_total': L_total
    }
    
    return L_total, loss_dict

class AdaptiveWeightScheduler:
    """
    Manages dynamic loss weighing to balance gradients 
    during training, preventing spectral bias.
    """
    
    def __init__(self, w_pde: float = 1.0, w_dir: float = 100.0, w_neu: float = 10.0,
                 balance_frequency: int = 100, alpha: float = 0.95, 
                 max_weight_ratio: float = 1000.0, strategy: str = "gradnorm"):
        
        self.w_pde = float(w_pde)
        self.w_dir = float(w_dir)
        self.w_neu = float(w_neu)
        self.balance_frequency = int(balance_frequency)
        self.alpha = float(alpha)
        self.max_weight_ratio = float(max_weight_ratio)
        self.strategy = strategy
        self.step_count = 0
        self.loss_history: Dict[str, list] = {'L_PDE': [], 'L_D': [], 'L_N': []}

    def step(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        """Records current losses and updates weights periodically."""
        self.step_count += 1
        for key in self.loss_history.keys():
            if key in loss_dict:
                self.loss_history[key].append(float(loss_dict[key].item()))

        if self.step_count % self.balance_frequency == 0 and len(self.loss_history['L_PDE']) >= 10:
            self._rebalance_weights()

    def _rebalance_weights(self) -> None:
        """Dynamic balancing based on loss magnitudes."""
        win = min(10, len(self.loss_history['L_PDE']))
        
        r_pde = max(torch.tensor(self.loss_history['L_PDE'][-win:]).mean().item(), 1e-12)
        r_dir = max(torch.tensor(self.loss_history['L_D'][-win:]).mean().item(), 1e-12)
        r_neu = max(torch.tensor(self.loss_history['L_N'][-win:]).mean().item(), 1e-12)

        # Update weights to maintain balance
        new_w_dir = min(r_pde / r_dir, self.max_weight_ratio)
        new_w_neu = min(r_pde / r_neu, self.max_weight_ratio)

        # Exponential Moving Average for smooth transitions
        self.w_dir = self.alpha * self.w_dir + (1.0 - self.alpha) * new_w_dir
        self.w_neu = self.alpha * self.w_neu + (1.0 - self.alpha) * new_w_neu

    def get_weights(self) -> Tuple[float, float, float]:
        """Returns the current weight coefficients."""
        return float(self.w_pde), float(self.w_dir), float(self.w_neu)
