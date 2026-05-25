"""
Scientific visualization module for PINN Thermal 2D models.
Generates publication-quality figures using matplotlib and seaborn.
Comprehensive diagnostics for loss components, weights, and optimization health.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib

# Headless configuration for scientific clusters
matplotlib.use('Agg')
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex': False  
})

class PublicationPlotter:
    """Orchestrates the generation of scientific-grade visualizations and diagnostics."""
    
    def __init__(self, analyzer, history: Dict[str, List[float]], 
                 output_dir: str = "plots",
                 training_data: Optional[Dict[str, torch.Tensor]] = None):
        self.analyzer = analyzer
        self.history = history
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_data = training_data
        sns.set_theme(style="white")

    def run_all_enhanced_plots(self):
        """Generates the full suite of diagnostic and solution figures."""
        logger_pinn = logging.getLogger("pinn_thermal")
        logger_pinn.info("Generating expanded dashboard...")
        
        self._plot_scientific_comparison()
        self._plot_convergence_spectrum()
        self._plot_profile_y05()
        self._plot_point_distribution()
        self._plot_loss_components()
        self._plot_adaptive_weights()
        self._plot_optimization_diagnostics()

    def _plot_scientific_comparison(self):
        """Plots Primary Solution Comparison (Exact vs PINN vs Error)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = [r'Solução Analítica ($T_{exata}$)', 
                  r'Reconstrução PINN ($\hat{T}$)', 
                  r'Erro L2 Pontual ($|T - \hat{T}|$)']
        
        datas = [self.analyzer.T_exact, self.analyzer.T_pred, np.abs(self.analyzer.T_exact - self.analyzer.T_pred)]
        cmaps = ['plasma', 'plasma', 'magma']
        
        for ax, title, data, cmap in zip(axes, titles, datas, cmaps):
            im = ax.imshow(data, extent=[0, 1, 0, 1], origin='lower', cmap=cmap)
            ax.set_title(title, weight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_scientific_comparison.png", dpi=600)
        plt.close()

    def _plot_convergence_spectrum(self):
        """Plots the total loss convergence trajectory."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.history['adam_loss'], label='Perda Total Ponderada', color='navy', lw=2)
        plt.yscale('log')
        plt.xlabel('Iterações de Otimização (Épocas)')
        plt.ylabel('Nível de Perda')
        plt.title('Espectro de Convergência Numérica (Total)', weight='bold')
        plt.grid(True, which="both", ls="-", alpha=0.15)
        plt.legend()
        plt.savefig(self.output_dir / "02_total_loss_convergence.png", dpi=600)
        plt.close()

    def _plot_loss_components(self):
        """Plots the evolution of individual loss terms (PDE, Dirichlet, Neumann)."""
        plt.figure(figsize=(10, 6))
        
        components = {
            'L_PDE': ('Resíduo da EDP', 'crimson'),
            'L_D': ('Contorno de Dirichlet', 'forestgreen'),
            'L_N': ('Contorno de Neumann', 'royalblue')
        }
        
        for key, (label, color) in components.items():
            if key in self.history:
                plt.plot(self.history[key], label=label, color=color, alpha=0.8, lw=1.5)
                
        plt.yscale('log')
        plt.xlabel('Épocas')
        plt.ylabel('Valor da Perda do Componente')
        plt.title('Evolução dos Componentes de Perda (Física/Contorno)', weight='bold')
        plt.grid(True, which="both", ls="-", alpha=0.15)
        plt.legend(loc='best', frameon=True)
        plt.savefig(self.output_dir / "03_loss_components.png", dpi=600)
        plt.close()

    def _plot_point_distribution(self):
        """Visualizes the spatial distribution of collocation and boundary points."""
        if self.training_data is None:
            return
            
        plt.figure(figsize=(8, 8))
        
        X_f = self.training_data['X_f'].cpu().numpy()
        X_dir = self.training_data['X_dir'].cpu().numpy()
        X_bot = self.training_data['X_bottom'].cpu().numpy()
        X_top = self.training_data['X_top'].cpu().numpy()
        
        plt.scatter(X_f[:, 0], X_f[:, 1], s=2, color='gray', alpha=0.3, label='Colocação (Interior)')
        plt.scatter(X_dir[:, 0], X_dir[:, 1], s=10, color='red', marker='x', label='Contorno de Dirichlet')
        plt.scatter(X_bot[:, 0], X_bot[:, 1], s=10, color='blue', marker='+', label='Neumann Inferior')
        plt.scatter(X_top[:, 0], X_top[:, 1], s=10, color='cyan', marker='+', label='Neumann Superior')
        
        plt.xlabel('Coordenada x')
        plt.ylabel('Coordenada y')
        plt.title('Estratégia de Colocação Espacial', weight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
        plt.axis('equal')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(self.output_dir / "04_point_distribution.png", dpi=600)
        plt.close()

    def _plot_adaptive_weights(self):
        """Plots the trajectory of adaptive weight coefficients."""
        if 'w_dir' not in self.history:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.get('w_pde', [1.0]*len(self.history['adam_loss'])), label=r'$w_{PDE}$', color='black')
        plt.plot(self.history['w_dir'], label=r'$w_{Dirichlet}$', color='forestgreen')
        plt.plot(self.history['w_neu'], label=r'$w_{Neumann}$', color='royalblue')
        
        plt.yscale('log')
        plt.xlabel('Épocas')
        plt.ylabel('Magnitude do Peso')
        plt.title('Dinâmica do Balanceamento Adaptativo de Pesos', weight='bold')
        plt.grid(True, which="both", ls="-", alpha=0.15)
        plt.legend()
        plt.savefig(self.output_dir / "05_adaptive_weights.png", dpi=600)
        plt.close()

    def _plot_optimization_diagnostics(self):
        """Plots Learning Rate decay and Gradient Norm evolution."""
        if 'lr' not in self.history:
            return
            
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Learning Rate on left axis
        color = 'tab:orange'
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Taxa de Aprendizado (LR)', color=color)
        ax1.plot(self.history['lr'], color=color, lw=2, label='Taxa de Aprendizado')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log')
        
        # Gradient Norm on right axis
        ax2 = ax1.twinx()
        color = 'tab:purple'
        ax2.set_ylabel('Norma Global do Gradiente', color=color)
        ax2.plot(self.history['grad_norm'], color=color, lw=1.5, alpha=0.6, label='Grad Norm')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yscale('log')
        
        plt.title('Saúde da Otimização: Trajetórias de LR e Gradiente', weight='bold')
        fig.tight_layout()
        plt.savefig(self.output_dir / "06_optimization_diagnostics.png", dpi=600)
        plt.close()

    def _plot_profile_y05(self):
        """Vertical cut along the centerline."""
        mid = self.analyzer.domain_size // 2
        x = np.linspace(0, 1, self.analyzer.domain_size)
        
        plt.figure(figsize=(8, 5))
        plt.plot(x, self.analyzer.T_exact[mid, :], 'k-', lw=2, label='Referência Exata')
        plt.plot(x, self.analyzer.T_pred[mid, :], 'r--', lw=2, label='Predição PINN')
        
        plt.xlabel('Coordenada x')
        plt.ylabel('Temperatura [K]')
        plt.title('Perfil Horizontal de Temperatura Meio-Domínio ($y = 0.5$)', weight='bold')
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.savefig(self.output_dir / "07_profile_validation.png", dpi=600)
        plt.close()

import logging
