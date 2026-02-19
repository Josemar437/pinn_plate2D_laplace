"""
Geração de figuras técnicas para análise de desempenho numérico e físico do PINN.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

try:
    from .config import Config
    from .operators import dT_dy_on
except ImportError:  
    from config import Config
    from operators import dT_dy_on

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "text.usetex": False,
    }
)


class PublicationPlotter:
    """Produz o conjunto de gráficos de solução, perda e diagnóstico numérico."""

    def __init__(
        self,
        analyzer,
        history: Dict[str, List[float]],
        output_dir: str = "runs/plots",
        training_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Recebe resultados do experimento e prepara diretório de saída."""
        self.analyzer = analyzer
        self.history = history
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_data = training_data
        sns.set_theme(style="white")

    def run_all_enhanced_plots(self) -> None:
        """Executa a geração de todas as figuras previstas no experimento."""
        logger_pinn = logging.getLogger("pinn_thermal")
        logger_pinn.info("Gerando dashboard de plots padronizados...")

        self._plot_ensemble_predictions()
        self._plot_field_comparison()
        self._plot_midline_error_difference()
        self._plot_physics_consistency()
        self._plot_training_curves()
        self._plot_training_history()
        self._plot_uncertainty_analysis()
        self._plot_gan_quality_metrics_if_available()

    def _plot_ensemble_predictions(self) -> None:
        """Salva ensemblePredictions.png com ref/pred/erro absoluto."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ["Campo FDM", "Predição PINN", "Erro Absoluto"]
        datas = [
            self.analyzer.T_ref,
            self.analyzer.T_pred,
            np.abs(self.analyzer.T_ref - self.analyzer.T_pred),
        ]
        cmaps = ["plasma", "plasma", "magma"]

        for ax, title, data, cmap in zip(axes, titles, datas, cmaps):
            im = ax.imshow(
                data, extent=[0, Config.LX, 0, Config.LY], origin="lower", cmap=cmap
            )
            ax.set_title(title, weight="bold")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(self.output_dir / "ensemblePredictions.png", dpi=600)
        plt.close()

    def _plot_field_comparison(self) -> None:
        """Salva fieldComparison.png com perfil central e dispersão ref vs pred."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        y_axis = np.linspace(0.0, Config.LY, self.analyzer.domain_size)
        target_y = 0.5 * float(Config.LY)
        mid = int(np.argmin(np.abs(y_axis - target_y)))
        x = np.linspace(0.0, Config.LX, self.analyzer.domain_size)
        axes[0].plot(x, self.analyzer.T_ref[mid, :], "k-", lw=2, label="FDM")
        axes[0].plot(x, self.analyzer.T_pred[mid, :], "r--", lw=2, label="PINN")
        axes[0].set_title(f"Perfil em y = {y_axis[mid]:.3f}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("Temperatura [K]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        t_ref_flat = self.analyzer.T_ref.flatten()
        t_pred_flat = self.analyzer.T_pred.flatten()
        axes[1].scatter(t_ref_flat, t_pred_flat, s=6, alpha=0.25, color="navy")
        low = float(min(np.min(t_ref_flat), np.min(t_pred_flat)))
        high = float(max(np.max(t_ref_flat), np.max(t_pred_flat)))
        axes[1].plot([low, high], [low, high], "r--", lw=1.5, label="Ideal")
        axes[1].set_title("Comparação ponto a ponto")
        axes[1].set_xlabel("FDM [K]")
        axes[1].set_ylabel("PINN [K]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "fieldComparison.png", dpi=600)
        plt.close()

    def _plot_midline_error_difference(self) -> None:
        """Salva midlineErrorDifference.png com T_PINN - T_FDM em y=0.5."""
        y_axis = np.linspace(0.0, Config.LY, self.analyzer.domain_size)
        target_y = 0.5 * float(Config.LY)
        mid = int(np.argmin(np.abs(y_axis - target_y)))
        x = np.linspace(0.0, Config.LX, self.analyzer.domain_size)
        diff = self.analyzer.T_pred[mid, :] - self.analyzer.T_ref[mid, :]

        plt.figure(figsize=(10, 5))
        plt.plot(x, diff, color="darkred", lw=2, label=r"$T_{PINN} - T_{FDM}$")
        plt.axhline(0.0, color="black", ls="--", lw=1)
        plt.title(f"Diferenca de erro em y = {y_axis[mid]:.3f}")
        plt.xlabel("x")
        plt.ylabel("Erro [K]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "midlineErrorDifference.png", dpi=600)
        plt.close()

    def _plot_physics_consistency(self) -> None:
        """Salva physicsConsistency.png com residual PDE e coerência de contorno."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        residual = self.analyzer.pde_residual
        im = axes[0].imshow(
            np.abs(residual),
            extent=[0, Config.LX, 0, Config.LY],
            origin="lower",
            cmap="viridis",
        )
        axes[0].set_title("Magnitude do residual PDE")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        y = np.linspace(0.0, Config.LY, self.analyzer.domain_size)
        dir_left = np.abs(self.analyzer.T_pred[:, 0] - float(Config.T_LEFT))
        dir_right = np.abs(self.analyzer.T_pred[:, -1] - float(Config.T_RIGHT))
        axes[1].plot(
            y,
            dir_left,
            color="forestgreen",
            lw=2,
            label=r"$|T-T_{left}|$ em x=0",
        )
        axes[1].plot(
            y,
            dir_right,
            color="crimson",
            lw=2,
            label=r"$|T-T_{right}|$ em x=Lx",
        )
        axes[1].set_title("Residual de Dirichlet nas bordas")
        axes[1].set_xlabel("y")
        axes[1].set_ylabel(r"$|T-T_{BC}|$")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        x_full = torch.linspace(
            0.0,
            float(Config.LX),
            self.analyzer.domain_size,
            device=self.analyzer.device,
            dtype=torch.get_default_dtype(),
        )
        # Exclui cantos para manter consistência com o conjunto Neumann do treino.
        x_eval = x_full[1:-1] if x_full.numel() > 2 else x_full
        y_bottom = torch.zeros_like(x_eval)
        y_top = torch.full_like(x_eval, float(Config.LY))
        pts_neu = torch.cat(
            [
                torch.stack([x_eval, y_bottom], dim=1),
                torch.stack([x_eval, y_top], dim=1),
            ],
            dim=0,
        )
        dTdy_all = (
            dT_dy_on(pts_neu, self.analyzer.net, create_graph=False)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        n_pts = int(x_eval.shape[0])
        dTdy_bottom = dTdy_all[:n_pts]
        dTdy_top = dTdy_all[n_pts:]
        x = x_eval.detach().cpu().numpy()
        axes[2].plot(
            x, np.abs(dTdy_bottom), color="darkorange", lw=2, label=r"$|dT/dy|$ y=0"
        )
        axes[2].plot(
            x, np.abs(dTdy_top), color="steelblue", lw=2, label=r"$|dT/dy|$ y=Ly"
        )
        axes[2].set_title("Residual de Neumann nas bordas")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel(r"$|dT/dy|$")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "physicsConsistency.png", dpi=600)
        plt.close()

    def _plot_training_curves(self) -> None:
        """Salva trainingCurves.png com perda total e componentes principais."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if "adam_loss" in self.history:
            axes[0].plot(self.history["adam_loss"], label="Loss total", color="navy", lw=2)
        axes[0].set_yscale("log")
        axes[0].set_title("Curva de treino total")
        axes[0].set_xlabel("Epocas")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, which="both", alpha=0.2)
        axes[0].legend()

        components = {
            "L_PDE": "PDE",
            "L_D": "Dirichlet",
            "L_N": "Neumann",
            "L_data": "Data",
            "L_curv": "Curvature",
        }
        for key, label in components.items():
            if key in self.history:
                axes[1].plot(self.history[key], label=label, lw=1.5)
        axes[1].set_yscale("log")
        axes[1].set_title("Componentes da perda")
        axes[1].set_xlabel("Epocas")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, which="both", alpha=0.2)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "trainingCurves.png", dpi=600)
        plt.close()

    def _plot_training_history(self) -> None:
        """Salva trainingHistory.png com pesos, LR e norma de gradiente."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if "w_pde" in self.history:
            axes[0].plot(self.history["w_pde"], label="w_pde", color="black")
        if "w_dir" in self.history:
            axes[0].plot(self.history["w_dir"], label="w_dir", color="forestgreen")
        if "w_neu" in self.history:
            axes[0].plot(self.history["w_neu"], label="w_neu", color="royalblue")
        if "w_data" in self.history:
            axes[0].plot(self.history["w_data"], label="w_data", color="darkorange")
        if "w_curv" in self.history:
            axes[0].plot(self.history["w_curv"], label="w_curv", color="teal")
        if "w_dir_left_loss" in self.history:
            axes[0].plot(
                self.history["w_dir_left_loss"],
                label="w_dir_left_loss",
                color="darkgreen",
            )
        if "w_dir_right_loss" in self.history:
            axes[0].plot(
                self.history["w_dir_right_loss"],
                label="w_dir_right_loss",
                color="firebrick",
            )
        axes[0].set_yscale("log")
        axes[0].set_title("Historico de pesos")
        axes[0].set_xlabel("Epocas")
        axes[0].set_ylabel("Peso")
        axes[0].grid(True, which="both", alpha=0.2)
        axes[0].legend()

        if "lr" in self.history:
            axes[1].plot(self.history["lr"], label="LR", color="tab:orange", lw=2)
        if "grad_norm" in self.history:
            axes[1].plot(
                self.history["grad_norm"], label="Grad norm", color="tab:purple", lw=1.5
            )
        axes[1].set_yscale("log")
        axes[1].set_title("Historico de otimizacao")
        axes[1].set_xlabel("Epocas")
        axes[1].set_ylabel("Magnitude")
        axes[1].grid(True, which="both", alpha=0.2)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "trainingHistory.png", dpi=600)
        plt.close()

    def _plot_uncertainty_analysis(self) -> None:
        """Salva uncertaintyAnalysis.png com proxies de incerteza do PINN."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        abs_error = np.abs(self.analyzer.T_ref - self.analyzer.T_pred)
        im = axes[0].imshow(
            abs_error,
            extent=[0, Config.LX, 0, Config.LY],
            origin="lower",
            cmap="magma",
        )
        axes[0].set_title("Proxy de incerteza (erro absoluto)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        residual_abs = np.abs(self.analyzer.pde_residual).flatten()
        axes[1].hist(residual_abs, bins=50, color="teal", alpha=0.8)
        axes[1].set_title("Distribuicao do residual PDE")
        axes[1].set_xlabel("|Residual|")
        axes[1].set_ylabel("Frequencia")
        axes[1].set_yscale("log")
        axes[1].grid(True, which="both", alpha=0.2)

        plt.tight_layout()
        plt.savefig(self.output_dir / "uncertaintyAnalysis.png", dpi=600)
        plt.close()

    def _plot_gan_quality_metrics_if_available(self) -> None:
        """
        Salva ganQualityMetrics.png somente quando o histórico possuir métricas GAN.

        Para a PINN térmica pura, esse plot normalmente não se aplica.
        """
        gan_keys = [k for k in ("gan_g_loss", "gan_d_loss", "fid", "is_score") if k in self.history]
        if not gan_keys:
            logging.getLogger("pinn_thermal").info(
                "ganQualityMetrics não aplicável para este experimento (sem métricas GAN)."
            )
            return

        plt.figure(figsize=(10, 6))
        for key in gan_keys:
            plt.plot(self.history[key], label=key, lw=2)
        plt.title("GAN Quality Metrics")
        plt.xlabel("Epocas")
        plt.ylabel("Metrica")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "ganQualityMetrics.png", dpi=600)
        plt.close()

