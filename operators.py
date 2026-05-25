"""
Differential operators module for PINN Thermal 2D calculations.
Provides optimized computation of Laplacian and Neumann boundary conditions using Autograd.
"""
import torch
import torch.autograd as autograd
import torch.nn as nn

def laplacian_T(xy: torch.Tensor, net: nn.Module, hard_constraint_bc: bool = True) -> torch.Tensor:
    """
    Computes the Laplacian (d2T/dx2 + d2T/dy2) of temperature T.
    Ensures input is isolated by detaching to avoid graph accumulation errors.
    
    Args:
        xy: Input coordinates
        net: Neural network model
        hard_constraint_bc: If True, uses forward_with_hard_constraint method if available
    """
    xy_req = xy.detach().requires_grad_(True)
    
    # Use appropriate forward method based on hard constraint setting
    if hard_constraint_bc and hasattr(net, 'forward_with_hard_constraint'):
        T = net.forward_with_hard_constraint(xy_req)
    else:
        T = net(xy_req)
    
    # First derivatives
    dT = autograd.grad(T, xy_req, torch.ones_like(T), create_graph=True)[0]
    
    # Second derivatives
    d2Tdx2 = autograd.grad(dT[:, 0:1], xy_req, torch.ones_like(dT[:, 0:1]), create_graph=True)[0][:, 0:1]
    d2Tdy2 = autograd.grad(dT[:, 1:2], xy_req, torch.ones_like(dT[:, 1:2]), create_graph=True)[0][:, 1:2]
    
    return d2Tdx2 + d2Tdy2

def dT_dy_on(xy: torch.Tensor, net: nn.Module, hard_constraint_bc: bool = True) -> torch.Tensor:
    """Computes the normal derivative dT/dy on boundaries.
    
    Args:
        xy: Boundary coordinates
        net: Neural network model
        hard_constraint_bc: If True, uses forward_with_hard_constraint method if available
    """
    xy_req = xy.detach().requires_grad_(True)
    
    # Use appropriate forward method based on hard constraint setting
    if hard_constraint_bc and hasattr(net, 'forward_with_hard_constraint'):
        T = net.forward_with_hard_constraint(xy_req)
    else:
        T = net(xy_req)
    
    dT = autograd.grad(T, xy_req, torch.ones_like(T), create_graph=True)[0]
    return dT[:, 1:2]

def dT_dx_on(xy: torch.Tensor, net: nn.Module) -> torch.Tensor:
    """Computes the horizontal derivative dT/dx on boundaries."""
    xy_req = xy.detach().requires_grad_(True)
    T = net(xy_req)
    dT = autograd.grad(T, xy_req, torch.ones_like(T), create_graph=True)[0]
    return dT[:, 0:1]
