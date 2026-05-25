"""
High-Precision configurations for PINN Thermal 2D calculations.
Standardized for academic journal quality.
"""
import torch
import numpy as np
import logging
from typing import Dict, Any

class Config:
    """Project-wide settings for high-performance computation."""
    
    # Precision Settings (Essential for <1e-4 accuracy)
    USE_DOUBLE: bool = True
    
    # Physical Geometry & Conditions
    T_LEFT: float = 200.0
    T_RIGHT: float = 100.0
    LX: float = 1.0
    LY: float = 1.0
    
    # Numerical Reproducibility
    SEED: int = 42
    
    # Global Training Hyperparameters (Balanced for demonstration)
    DEFAULT_EPOCHS_ADAM: int = 4000
    DEFAULT_EPOCHS_LBFGS: int = 400
    DEFAULT_LR: float = 1e-3
    
    # Collocation Grid Resolution (High density)
    DEFAULT_N_INTERIOR: int = 10000
    DEFAULT_N_BOUNDARY: int = 600
    
    # Visualization Settings
    DOMAIN_SIZE: int = 150
    DPI: int = 600
    
    # Neural Network Defaults (Deeper/Wider)
    DEFAULT_LAYERS: list = [2, 64, 64, 64, 1]
    DEFAULT_ACTIVATION: str = 'tanh'
    
    # Adaptive Weighting Config
    DEFAULT_WEIGHT_PARAMS: Dict[str, Any] = {
        'w_pde': 1.0,
        'w_dir': 400.0,
        'w_neu': 40.0,
        'balance_frequency': 100,
        'alpha': 0.9,
        'max_weight_ratio': 1000.0,
        'strategy': 'loss_balance'
    }
    
    # Hard Constraint BC Settings
    HARD_CONSTRAINT_BC: bool = True  # Use hard constraints for Dirichlet BCs on lateral sides
    ENFORCE_SYMMETRY: bool = False
    
    @classmethod
    def setup_torch(cls) -> None:
        """Initializes PyTorch backend with double precision and reproducibility."""
        dtype = torch.float64 if cls.USE_DOUBLE else torch.float32
        torch.set_default_dtype(dtype)
        torch.manual_seed(cls.SEED)
        np.random.seed(cls.SEED)
        
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False # Disable TF32 for double precision
            torch.backends.cudnn.allow_tf32 = False

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures scientific logging for technical output."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger("pinn_thermal")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger

def get_device() -> torch.device:
    """Detects and returns the optimal compute backend."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA Ativado: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("CPU Ativado")
    return device

# Framework Initialization
Config.setup_torch()
logger = setup_logging()
device = get_device()
