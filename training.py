"""
Training module for PINN Thermal 2D solvers.
Implements hybrid Adam/L-BFGS optimization with adaptive weighting and diagnostic tracking.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from config import Config, device, logger
from losses import enhanced_loss_terms, AdaptiveWeightScheduler

def train_pinn_enhanced(net: nn.Module, X_f: torch.Tensor, 
                       X_dir: torch.Tensor, T_dir_target: torch.Tensor,
                       X_bottom: torch.Tensor, X_top: torch.Tensor,
                       epochs_adam: int = 4000, 
                       epochs_lbfgs: int = 1500,
                       lr: float = 1e-3, 
                       weight_params: Optional[Dict] = None,
                       use_scheduler: bool = True,
                       hard_constraint_bc: bool = True,
                       verbose: bool = True) -> Dict[str, List[float]]:
    """
    Performs hybrid training (Adam followed by L-BFGS) with adaptive weighting and diagnostic tracking.
    
    Args:
        hard_constraint_bc: If True, use hard constraints for lateral Dirichlet BCs
    """
    net.train()
    hist = {
        'adam_loss': [], 
        'L_PDE': [], 'L_D': [], 'L_N': [],
        'w_pde': [], 'w_dir': [], 'w_neu': [],
        'lr': [], 'grad_norm': []
    }

    # Setup Schedulers
    weight_params = weight_params or Config.DEFAULT_WEIGHT_PARAMS
    weight_scheduler = AdaptiveWeightScheduler(**weight_params)
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)

    logger.info(f"Iniciando Fase de Treinamento Adam ({epochs_adam} épocas)...")
    logger.info(f"Hard Constraint BC: {'ATIVADO' if hard_constraint_bc else 'DESATIVADO'}")
    start_time = time.time()

    for epoch in range(epochs_adam):
        optimizer.zero_grad(set_to_none=True)
        
        # Get current weights and calculate loss
        w_pde, w_dir, w_neu = weight_scheduler.get_weights()
        loss_total, loss_parts = enhanced_loss_terms(
            net, X_f, X_dir, T_dir_target, X_bottom, X_top,
            w_pde=w_pde, w_dir=w_dir, w_neu=w_neu,
            hard_constraint_bc=hard_constraint_bc
        )
        
        loss_total.backward()
        
        # Capture gradient norm for diagnostics
        grad_norm = 0.0
        for p in net.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item()**2
        grad_norm = grad_norm**0.5
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update Schedulers
        if use_scheduler:
            lr_scheduler.step(loss_total)
        weight_scheduler.step(loss_parts)
        
        # Logging
        hist['adam_loss'].append(loss_total.item())
        hist['L_PDE'].append(loss_parts['L_PDE'].item())
        hist['L_D'].append(loss_parts['L_D'].item())
        hist['L_N'].append(loss_parts['L_N'].item())
        hist['w_pde'].append(float(w_pde))
        hist['w_dir'].append(float(w_dir))
        hist['w_neu'].append(float(w_neu))
        hist['lr'].append(optimizer.param_groups[0]['lr'])
        hist['grad_norm'].append(grad_norm)

        if verbose and epoch % 500 == 0:
            logger.info(f"Epoch {epoch:04d} | Total Loss: {loss_total.item():.2e} | PDE: {loss_parts['L_PDE'].item():.2e} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # L-BFGS Fine-tuning
    if epochs_lbfgs > 0:
        logger.info(f"Iniciando Refinamento L-BFGS ({epochs_lbfgs} iterações)...")
        lbfgs = optim.LBFGS(net.parameters(), lr=1.0, max_iter=epochs_lbfgs, history_size=50)
        
        def closure():
            lbfgs.zero_grad()
            l_tot, _ = enhanced_loss_terms(net, X_f, X_dir, T_dir_target, X_bottom, X_top,
                                          hard_constraint_bc=hard_constraint_bc)
            l_tot.backward()
            return l_tot
            
        lbfgs.step(closure)

    logger.info(f"Treinamento concluído em {time.time() - start_time:.1f}s")
    return hist
