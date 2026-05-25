"""
Scientific analysis and metrics module for PINN Thermal 2D models.
Provides exact solution verification and statistical error quantification.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from config import Config, device, logger
from operators import laplacian_T

def exact_analytical_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact linear solution for the Steady-State Conduction on a Flat Plate."""
    return Config.T_LEFT + (Config.T_RIGHT - Config.T_LEFT) * (x / Config.LX)

class EnhancedThermalAnalyzer:
    """Rigorous analyzer for evaluating PINN reconstructions against physical benchmarks."""
    
    def __init__(self, net: nn.Module, domain_size: int = 100, 
                 device_: torch.device = device) -> None:
        self.net = net.eval()
        self.device = device_
        self.domain_size = domain_size
        
        # Grid Setup
        x = np.linspace(0, Config.LX, domain_size)
        y = np.linspace(0, Config.LY, domain_size)
        self.X, self.Y = np.meshgrid(x, y)
        self.pts = torch.tensor(np.stack([self.X.flatten(), self.Y.flatten()], 1), 
                               dtype=torch.get_default_dtype(), device=device_)
        
        self.evaluate()

    def evaluate(self):
        """Quantifies error metrics and PDE residuals across the domain."""
        with torch.no_grad():
            self.T_pred = self.net(self.pts).cpu().numpy().reshape(self.domain_size, self.domain_size)
            self.T_exact = exact_analytical_solution(self.X, self.Y)
            
        # Error metrics
        err = self.T_exact - self.T_pred
        self.metrics = {
            'MAE': float(np.mean(np.abs(err))),
            'RMSE': float(np.sqrt(np.mean(err**2))),
            'Max_Error': float(np.max(np.abs(err))),
            'R2': float(1.0 - np.sum(err**2) / np.sum((self.T_exact - np.mean(self.T_exact))**2))
        }
        
        # PDE Residuals
        self.pts.requires_grad_(True)
        res = laplacian_T(self.pts, self.net).detach().cpu().numpy()
        self.pde_residual = res.reshape(self.domain_size, self.domain_size)

    def print_comprehensive_analysis(self) -> None:
        """Logs a scientific summary of the solver performance."""
        logger.info("\n" + "="*80)
        logger.info("SCIENTIFIC PERFORMANCE MANIFEST")
        logger.info("="*80)
        logger.info(f"Mean Absolute Error (MAE):     {self.metrics['MAE']:.4e} K")
        logger.info(f"Root Mean Square Error (RMSE): {self.metrics['RMSE']:.4e} K")
        logger.info(f"Coefficient of Determination:   {self.metrics['R2']:.6f}")
        logger.info(f"Global Peak Error:             {self.metrics['Max_Error']:.4e} K")
        logger.info("="*80)

def analyze_convergence(history: Dict[str, List[float]]) -> Dict[str, float]:
    """Evaluates the numerical stability and asymptotic behavior of the solver."""
    if not history.get('adam_loss'):
        return {}
    
    losses = np.array(history['adam_loss'])
    return {
        'final_loss': float(losses[-1]),
        'total_epochs': len(losses),
        'loss_reduction': float((losses[0] - losses[-1]) / losses[0] * 100)
    }
