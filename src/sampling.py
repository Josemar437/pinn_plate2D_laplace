"""
Geração de pontos de treino e pares supervisionados a partir da referência FDM.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.quasirandom import SobolEngine

try:
    from .config import Config, build_fdm_cache_key, device, logger
    from .fdm import generate_reference_solution
except ImportError:  
    from config import Config, build_fdm_cache_key, device, logger
    from fdm import generate_reference_solution


_FDM_REFERENCE_CACHE: dict[tuple, tuple[torch.Tensor, int]] = {}


@dataclass(frozen=True)
class TrainingData:
    """Agrupa os tensores necessários para treino, análise e plotagem."""

    X_f: torch.Tensor
    X_dir: torch.Tensor
    T_dir_target: torch.Tensor
    X_bottom: torch.Tensor
    X_top: torch.Tensor
    X_data: torch.Tensor
    T_data_target: torch.Tensor
    X_midline: torch.Tensor
    fdm_field: torch.Tensor
    fdm_iterations: int


def _sample_rows(
    rows: torch.Tensor, values: torch.Tensor, n_rows: int, *, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Amostra pares (coordenada, valor) da malha de referência.

    Usa amostragem sem reposição quando possível e com reposição quando
    n_rows excede o total disponível.
    """
    if n_rows <= 0:
        raise ValueError("n_rows deve ser positivo.")
    total = int(rows.shape[0])
    if total == 0:
        raise ValueError("Não há linhas disponíveis para amostragem.")

    if n_rows <= total:
        idx_cpu = torch.randperm(total, generator=generator)[:n_rows]
    else:
        idx_cpu = torch.randint(0, total, (n_rows,), generator=generator)

    idx = idx_cpu.to(rows.device)
    return rows[idx], values[idx]


def _latin_hypercube_2d(
    n_samples: int, *, generator: torch.Generator, device_: torch.device
) -> torch.Tensor:
    """Amostra pontos no domínio por Latin Hypercube Sampling em 2D."""
    if n_samples <= 0:
        raise ValueError("n_samples deve ser positivo.")

    base = (
        torch.arange(n_samples, dtype=torch.get_default_dtype()).unsqueeze(1).repeat(1, 2)
    )
    jitter = torch.rand((n_samples, 2), generator=generator, dtype=torch.get_default_dtype())
    lhs = (base + jitter) / float(n_samples)

    for dim in range(2):
        perm = torch.randperm(n_samples, generator=generator)
        lhs[:, dim] = lhs[perm, dim]

    lhs[:, 0] *= float(Config.LX)
    lhs[:, 1] *= float(Config.LY)
    return lhs.to(device_)


def _sobol_2d(
    n_samples: int,
    *,
    scramble: bool,
    seed: int,
    device_: torch.device,
) -> torch.Tensor:
    """Amostra pontos no domínio por sequência de Sobol em 2D."""
    if n_samples <= 0:
        raise ValueError("n_samples deve ser positivo.")

    engine = SobolEngine(dimension=2, scramble=bool(scramble), seed=int(seed))
    sobol = engine.draw(n_samples).to(dtype=torch.get_default_dtype())
    sobol[:, 0] *= float(Config.LX)
    sobol[:, 1] *= float(Config.LY)
    return sobol.to(device_)


def _sample_right_band_2d(
    n_samples: int,
    *,
    band_fraction: float,
    generator: torch.Generator,
    device_: torch.device,
) -> torch.Tensor:
    """Amostra pontos em uma faixa próxima à borda direita x=Lx."""
    if n_samples <= 0:
        raise ValueError("n_samples deve ser positivo.")

    band = float(Config.LX) * min(max(float(band_fraction), 1.0e-4), 1.0)
    # Gera em CPU para manter compatibilidade com gerador CPU e depois move.
    pts = torch.rand(
        (n_samples, 2),
        generator=generator,
        dtype=torch.get_default_dtype(),
    )
    pts[:, 0] = float(Config.LX) - (band * pts[:, 0])
    pts[:, 1] *= float(Config.LY)
    return pts.to(device_)


def _sample_left_band_2d(
    n_samples: int,
    *,
    band_fraction: float,
    generator: torch.Generator,
    device_: torch.device,
) -> torch.Tensor:
    """Amostra pontos em uma faixa próxima à borda esquerda x=0."""
    if n_samples <= 0:
        raise ValueError("n_samples deve ser positivo.")

    band = float(Config.LX) * min(max(float(band_fraction), 1.0e-4), 1.0)
    pts = torch.rand(
        (n_samples, 2),
        generator=generator,
        dtype=torch.get_default_dtype(),
    )
    pts[:, 0] = band * pts[:, 0]
    pts[:, 1] *= float(Config.LY)
    return pts.to(device_)


def _build_grid_coordinates(nx: int, ny: int, device_: torch.device) -> torch.Tensor:
    """Cria grade achatada [ny*nx, 2] alinhada ao campo FDM."""
    x = torch.linspace(0.0, Config.LX, int(nx), device=device_, dtype=torch.get_default_dtype())
    y = torch.linspace(0.0, Config.LY, int(ny), device=device_, dtype=torch.get_default_dtype())
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def _fdm_cache_key(*, device_: torch.device, dtype: torch.dtype) -> tuple:
    """Gera chave estável de cache para a referência FDM de treino."""
    return build_fdm_cache_key(
        nx=int(Config.FDM_NX),
        ny=int(Config.FDM_NY),
        device=device_,
        dtype=dtype,
    )


def create_training_data(
    N_interior: int = Config.DEFAULT_N_INTERIOR,
    N_boundary: int = Config.DEFAULT_N_BOUNDARY,
    N_data: int = Config.DEFAULT_N_DATA,
    N_midline: int | None = None,
    interior_strategy: Literal["fdm", "lhs", "sobol", "hybrid"] = Config.DEFAULT_INTERIOR_STRATEGY,
    device_: torch.device = device,
) -> TrainingData:
    """
    Monta conjunto completo de treino usando a solução FDM como referência.

    O retorno inclui pontos de colocação, contornos, pares supervisionados e
    o campo FDM integral para análise posterior.

    Nota física: os vértices são mantidos exclusivamente em Dirichlet. As
    amostras de Neumann em y=0 e y=Ly excluem cantos para evitar impor
    condições de contorno conflitantes no mesmo ponto.
    """
    valid_strategies = {"fdm", "lhs", "sobol", "hybrid"}
    strategy = str(interior_strategy).strip().lower()
    if strategy not in valid_strategies:
        raise ValueError(
            f"interior_strategy inválida: {interior_strategy!r}. "
            f"Use uma de {sorted(valid_strategies)}."
        )

    dtype = torch.get_default_dtype()
    cache_key = _fdm_cache_key(device_=device_, dtype=dtype)
    if cache_key in _FDM_REFERENCE_CACHE:
        cached_field, cached_iters = _FDM_REFERENCE_CACHE[cache_key]
        # Clone defensivo: evita mutação acidental da referência cacheada.
        fdm_field = cached_field.clone()
        fdm_iterations = int(cached_iters)
    else:
        fdm_field, fdm_iterations = generate_reference_solution(
            nx=Config.FDM_NX,
            ny=Config.FDM_NY,
            lx=Config.LX,
            ly=Config.LY,
            t_left=Config.T_LEFT,
            t_right=Config.T_RIGHT,
            tol=Config.FDM_TOL,
            max_iter=Config.FDM_MAX_ITER,
            omega=Config.FDM_OMEGA,
            device=device_,
            dtype=dtype,
        )
        _FDM_REFERENCE_CACHE[cache_key] = (fdm_field.detach().clone(), int(fdm_iterations))
    logger.info("Solução FDM convergiu em %d iterações.", fdm_iterations)

    pts = _build_grid_coordinates(Config.FDM_NX, Config.FDM_NY, device_=device_)
    temp_values = fdm_field.reshape(-1, 1)

    x = pts[:, 0]
    y = pts[:, 1]
    zero = torch.tensor(0.0, device=device_, dtype=pts.dtype)
    lx = torch.tensor(Config.LX, device=device_, dtype=pts.dtype)
    ly = torch.tensor(Config.LY, device=device_, dtype=pts.dtype)
    atol = 1e-12

    left_mask = torch.isclose(x, zero, atol=atol)
    right_mask = torch.isclose(x, lx, atol=atol)
    bottom_mask = torch.isclose(y, zero, atol=atol)
    top_mask = torch.isclose(y, ly, atol=atol)
    interior_mask = ~(left_mask | right_mask | bottom_mask | top_mask)

    interior_pts = pts[interior_mask]
    interior_temp = temp_values[interior_mask]
    left_pts, left_temp = pts[left_mask], temp_values[left_mask]
    right_pts, right_temp = pts[right_mask], temp_values[right_mask]
    corner_mask = (left_mask | right_mask) & (bottom_mask | top_mask)
    # Cantos pertencem fisicamente às fronteiras de Dirichlet (x=0 e x=Lx).
    # Excluí-los de Neumann evita impor condições conflitantes no mesmo ponto.
    bottom_mask_neu = bottom_mask & ~corner_mask
    top_mask_neu = top_mask & ~corner_mask
    bottom_pts, bottom_temp = pts[bottom_mask_neu], temp_values[bottom_mask_neu]
    top_pts, top_temp = pts[top_mask_neu], temp_values[top_mask_neu]
    if bottom_pts.shape[0] == 0 or top_pts.shape[0] == 0:
        raise ValueError(
            "Malha insuficiente para Neumann sem cantos; use FDM_NY >= 3."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(Config.SEED)

    n_interior = int(N_interior)
    if n_interior <= 0:
        raise ValueError("N_interior deve ser positivo.")

    use_left_importance = bool(getattr(Config, "ENABLE_LEFT_IMPORTANCE_SAMPLING", False))
    left_fraction = float(getattr(Config, "LEFT_IMPORTANCE_FRACTION", 0.0))
    left_fraction = min(max(left_fraction, 0.0), 1.0)
    left_band_fraction = float(getattr(Config, "LEFT_IMPORTANCE_BAND_FRACTION", 0.1))

    use_right_importance = bool(getattr(Config, "ENABLE_RIGHT_IMPORTANCE_SAMPLING", False))
    right_fraction = float(getattr(Config, "RIGHT_IMPORTANCE_FRACTION", 0.0))
    right_fraction = min(max(right_fraction, 0.0), 1.0)
    right_band_fraction = float(getattr(Config, "RIGHT_IMPORTANCE_BAND_FRACTION", 0.1))

    n_left_focus = int(round(n_interior * left_fraction)) if use_left_importance else 0
    n_right_focus = int(round(n_interior * right_fraction)) if use_right_importance else 0
    n_left_focus = min(max(0, n_left_focus), n_interior)
    n_right_focus = min(max(0, n_right_focus), n_interior)

    total_focus = n_left_focus + n_right_focus
    if total_focus > n_interior:
        scale = float(n_interior) / float(total_focus)
        n_left_focus = int(round(n_left_focus * scale))
        n_right_focus = int(round(n_right_focus * scale))
        while (n_left_focus + n_right_focus) > n_interior:
            if n_right_focus >= n_left_focus and n_right_focus > 0:
                n_right_focus -= 1
            elif n_left_focus > 0:
                n_left_focus -= 1

    n_base = n_interior - n_left_focus - n_right_focus

    chunks_xf: list[torch.Tensor] = []
    if n_base > 0:
        if strategy == "fdm":
            X_f_base, _ = _sample_rows(interior_pts, interior_temp, n_base, generator=generator)
            chunks_xf.append(X_f_base)
        elif strategy == "lhs":
            X_f_base = _latin_hypercube_2d(n_base, generator=generator, device_=device_)
            chunks_xf.append(X_f_base)
        elif strategy == "sobol":
            X_f_base = _sobol_2d(
                n_base,
                scramble=True,
                seed=int(Config.SEED),
                device_=device_,
            )
            chunks_xf.append(X_f_base)
        else:
            n_fdm = n_base // 2
            n_sobol = n_base - n_fdm
            if n_fdm > 0:
                X_f_fdm, _ = _sample_rows(
                    interior_pts, interior_temp, n_fdm, generator=generator
                )
                chunks_xf.append(X_f_fdm)
            if n_sobol > 0:
                X_f_sobol = _sobol_2d(
                    n_sobol,
                    scramble=True,
                    seed=int(Config.SEED) + 777,
                    device_=device_,
                )
                chunks_xf.append(X_f_sobol)

    if n_left_focus > 0:
        X_f_left = _sample_left_band_2d(
            n_left_focus,
            band_fraction=left_band_fraction,
            generator=generator,
            device_=device_,
        )
        chunks_xf.append(X_f_left)

    if n_right_focus > 0:
        X_f_right = _sample_right_band_2d(
            n_right_focus,
            band_fraction=right_band_fraction,
            generator=generator,
            device_=device_,
        )
        chunks_xf.append(X_f_right)

    if not chunks_xf:
        raise RuntimeError("Falha ao montar pontos de colocação.")

    X_f = torch.cat(chunks_xf, dim=0)
    if X_f.shape[0] > 1:
        perm_cpu = torch.randperm(int(X_f.shape[0]), generator=generator)
        X_f = X_f[perm_cpu.to(X_f.device)]

    if use_left_importance and n_left_focus > 0:
        logger.info(
            "Importance sampling em x≈0 ativo: %d/%d pontos internos na faixa esquerda (%.2f%% do domínio em x).",
            n_left_focus,
            n_interior,
            100.0 * min(max(left_band_fraction, 1.0e-4), 1.0),
        )
    if use_right_importance and n_right_focus > 0:
        logger.info(
            "Importance sampling em x≈Lx ativo: %d/%d pontos internos na faixa direita (%.2f%% do domínio em x).",
            n_right_focus,
            n_interior,
            100.0 * min(max(right_band_fraction, 1.0e-4), 1.0),
        )
    if bool(getattr(Config, "COLLOCATION_AUDIT_ENABLED", False)):
        audit_band_fraction = min(
            max(float(getattr(Config, "COLLOCATION_AUDIT_BAND_FRACTION", 0.1)), 1.0e-4),
            1.0,
        )
        left_limit = float(Config.LX) * audit_band_fraction
        right_limit = float(Config.LX) * (1.0 - audit_band_fraction)
        x_vals = X_f[:, 0]
        left_density = float(torch.mean((x_vals <= left_limit).to(X_f.dtype)).item())
        right_density = float(torch.mean((x_vals >= right_limit).to(X_f.dtype)).item())
        logger.info(
            "Diagnóstico de colocação (faixa %.2f%% de Lx): esquerda=%.1f%% | direita=%.1f%%.",
            100.0 * audit_band_fraction,
            100.0 * left_density,
            100.0 * right_density,
        )

    n_boundary_left = int(N_boundary)
    right_multiplier = max(
        1.0e-2, float(getattr(Config, "RIGHT_DIRICHLET_MULTIPLIER", 1.0))
    )
    n_boundary_right = max(1, int(round(n_boundary_left * right_multiplier)))
    if n_boundary_right != n_boundary_left:
        logger.info(
            "Reforço Dirichlet na direita: N_left=%d | N_right=%d (x%.2f).",
            n_boundary_left,
            n_boundary_right,
            right_multiplier,
        )

    X_left, T_left = _sample_rows(left_pts, left_temp, n_boundary_left, generator=generator)
    X_right, T_right = _sample_rows(right_pts, right_temp, n_boundary_right, generator=generator)
    X_bottom, _ = _sample_rows(
        bottom_pts, bottom_temp, n_boundary_left, generator=generator
    )
    X_top, _ = _sample_rows(top_pts, top_temp, n_boundary_left, generator=generator)
    X_data, T_data_target = _sample_rows(pts, temp_values, int(N_data), generator=generator)
    resolved_n_midline = (
        int(Config.DEFAULT_N_MIDLINE) if N_midline is None else int(N_midline)
    )
    n_midline = max(4, resolved_n_midline)
    x_mid = torch.linspace(
        0.0,
        float(Config.LX),
        n_midline,
        device=device_,
        dtype=torch.get_default_dtype(),
    ).unsqueeze(1)
    y_mid = torch.full_like(x_mid, 0.5 * float(Config.LY))
    X_midline = torch.cat([x_mid, y_mid], dim=1)

    X_dir = torch.cat([X_left, X_right], dim=0)
    T_dir_target = torch.cat([T_left, T_right], dim=0)

    return TrainingData(
        X_f=X_f,
        X_dir=X_dir,
        T_dir_target=T_dir_target,
        X_bottom=X_bottom,
        X_top=X_top,
        X_data=X_data,
        T_data_target=T_data_target,
        X_midline=X_midline,
        fdm_field=fdm_field,
        fdm_iterations=int(fdm_iterations),
    )

