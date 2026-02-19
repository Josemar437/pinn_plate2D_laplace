"""
Regras para aplicar resultados do Optuna na configuração de treino do projeto.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def round_to_sig_sci(value: float, sig_figs: int = 3) -> float:
    """Arredonda valor para 3 algarismos significativos em notação científica."""
    if sig_figs < 1:
        raise ValueError("sig_figs deve ser >= 1.")
    numeric = float(value)
    if numeric == 0.0:
        return 0.0
    return float(f"{numeric:.{sig_figs - 1}e}")


def parse_optuna_bool(value: Any) -> bool:
    """Normaliza representações comuns de booleano usadas em payloads."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Valor booleano inválido: {value!r}")
    return bool(value)


def extract_best_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extrai best_params do payload do Optuna com validação mínima."""
    if not isinstance(payload, dict):
        raise ValueError("payload inválido: esperado dict.")
    params = payload.get("best_params", payload)
    if not isinstance(params, dict):
        raise ValueError("payload inválido: 'best_params' deve ser dict.")
    return params


def resolve_optuna_layers(
    payload: Dict[str, Any], params: Dict[str, Any]
) -> list[int] | None:
    """Resolve arquitetura da rede a partir dos dados do melhor trial."""
    layers = payload.get("best_user_attrs", {}).get("layers")
    if layers is not None:
        return [int(v) for v in layers]

    hidden_layers = params.get("hidden_layers")
    hidden_width = params.get("hidden_width")
    if hidden_layers is None or hidden_width is None:
        return None
    return [2] + [int(hidden_width)] * int(hidden_layers) + [1]


def apply_optuna_best(
    config_cls: Any,
    payload: Dict[str, Any],
    *,
    sig_figs: int | None = None,
) -> Dict[str, Any]:
    """
    Aplica no config_cls os melhores hiperparâmetros produzidos pelo Optuna.

    Retorna o payload recebido para rastreabilidade no chamador.
    """
    params = extract_best_params(payload)
    resolved_sig = int(
        sig_figs
        if sig_figs is not None
        else getattr(config_cls, "HYPERPARAM_SIG_FIGS", 3)
    )

    if "lr" in params:
        config_cls.DEFAULT_LR = round_to_sig_sci(params["lr"], sig_figs=resolved_sig)
    if "w_data" in params:
        config_cls.DEFAULT_W_DATA = round_to_sig_sci(
            params["w_data"], sig_figs=resolved_sig
        )

    float_weight_keys = (
        "w_pde",
        "w_dir",
        "w_neu",
        "alpha",
        "max_weight_ratio",
        "max_weight_change",
    )
    for key in float_weight_keys:
        if key in params:
            config_cls.DEFAULT_WEIGHT_PARAMS[key] = round_to_sig_sci(
                params[key], sig_figs=resolved_sig
            )

    int_weight_keys = ("balance_frequency", "min_balance_step", "window")
    for key in int_weight_keys:
        if key in params:
            config_cls.DEFAULT_WEIGHT_PARAMS[key] = int(round(float(params[key])))

    if "use_adaptive_weights" in params:
        config_cls.USE_ADAPTIVE_WEIGHTING = parse_optuna_bool(
            params["use_adaptive_weights"]
        )

    layers = resolve_optuna_layers(payload, params)
    if layers is not None:
        config_cls.DEFAULT_LAYERS = layers

    return payload


def load_optuna_payload(best_path: str | Path) -> Dict[str, Any]:
    """Carrega payload JSON do melhor trial do Optuna."""
    resolved = Path(best_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Arquivo de melhores parâmetros não encontrado: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def apply_optuna_best_from_file(
    config_cls: Any, best_path: str | Path, *, sig_figs: int | None = None
) -> Dict[str, Any]:
    """Carrega arquivo JSON do Optuna e aplica no config_cls."""
    payload = load_optuna_payload(best_path)
    return apply_optuna_best(config_cls, payload, sig_figs=sig_figs)
