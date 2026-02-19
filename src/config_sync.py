"""
Sincronização de hiperparâmetros do Optuna para valores padrão em config.py.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

try:
    from .optuna_config import (
        extract_best_params,
        parse_optuna_bool,
        resolve_optuna_layers,
        round_to_sig_sci,
    )
except ImportError:  # pragma: no cover - compatibilidade com execução via PYTHONPATH=src
    from optuna_config import (
        extract_best_params,
        parse_optuna_bool,
        resolve_optuna_layers,
        round_to_sig_sci,
    )


def _format_float_literal(value: float, sig_figs: int) -> str:
    """Formata float arredondado como literal Python."""
    rounded = round_to_sig_sci(value, sig_figs=sig_figs)
    if rounded == 0.0:
        return "0.0"
    mantissa, exponent = f"{rounded:.{sig_figs - 1}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _format_bool_literal(value: Any) -> str:
    """Converte valor em literal booleano Python."""
    return "True" if parse_optuna_bool(value) else "False"


def _extract_sig_figs(config_text: str) -> int:
    """Lê HYPERPARAM_SIG_FIGS diretamente do arquivo de configuração."""
    match = re.search(
        r"^\s*HYPERPARAM_SIG_FIGS:\s*int\s*=\s*(\d+)\s*$",
        config_text,
        flags=re.MULTILINE,
    )
    if not match:
        return 3
    return max(1, int(match.group(1)))


def _replace_assignment_line(config_text: str, assignment: str, literal: str) -> str:
    """Substitui uma linha de atribuição única no Config."""
    pattern = rf"^(\s*{re.escape(assignment)}\s*=\s*)(.+)$"
    regex = re.compile(pattern, flags=re.MULTILINE)
    updated_text, count = regex.subn(
        lambda match: f"{match.group(1)}{literal}",
        config_text,
        count=1,
    )
    if count == 0:
        raise ValueError(f"Atribuição não encontrada no config.py: {assignment}")
    return updated_text


def _replace_weight_param(config_text: str, key: str, literal: str) -> str:
    """Substitui valor de uma chave dentro de DEFAULT_WEIGHT_PARAMS."""
    pattern = rf'^(\s*"{re.escape(key)}":\s*)([^,\n]+)(,\s*)$'
    regex = re.compile(pattern, flags=re.MULTILINE)
    updated_text, count = regex.subn(
        lambda match: f"{match.group(1)}{literal}{match.group(3)}",
        config_text,
        count=1,
    )
    if count == 0:
        raise ValueError(f"Chave não encontrada em DEFAULT_WEIGHT_PARAMS: {key}")
    return updated_text


def _format_int_list(values: list[int]) -> str:
    """Formata lista de inteiros como literal Python estável."""
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def _default_config_path() -> Path:
    """
    Resolve caminho padrão de configuração da aplicação.

    A resolução é ancorada no diretório src/ para evitar dependência acidental
    de módulos de teste e garantir funcionamento do CLI sem --config-path.
    """
    return Path(__file__).resolve().with_name("config.py")


def apply_best_to_config_file(
    *,
    best_path: str | Path = "runs/results/optuna_best.json",
    config_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Atualiza config.py com os melhores hiperparâmetros salvos pelo Optuna.

    O arredondamento segue o valor de HYPERPARAM_SIG_FIGS do próprio arquivo.
    """
    best_file = Path(best_path)
    if not best_file.exists():
        raise FileNotFoundError(f"Arquivo de melhores parâmetros não encontrado: {best_file}")

    target_config = Path(config_path) if config_path is not None else _default_config_path()
    if not target_config.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {target_config}")

    payload = json.loads(best_file.read_text(encoding="utf-8"))
    params = extract_best_params(payload)

    config_text = target_config.read_text(encoding="utf-8")
    sig_figs = _extract_sig_figs(config_text)

    assignment_updates: Dict[str, str] = {}
    if "lr" in params:
        assignment_updates["DEFAULT_LR: float"] = _format_float_literal(
            params["lr"], sig_figs
        )
    if "w_data" in params:
        assignment_updates["DEFAULT_W_DATA: float"] = _format_float_literal(
            params["w_data"], sig_figs
        )
    if "use_adaptive_weights" in params:
        assignment_updates["USE_ADAPTIVE_WEIGHTING: bool"] = _format_bool_literal(
            params["use_adaptive_weights"]
        )

    layers = resolve_optuna_layers(payload, params)
    if layers is not None:
        assignment_updates["DEFAULT_LAYERS: list[int]"] = _format_int_list(layers)

    weight_updates: Dict[str, str] = {}
    float_weight_keys = (
        "w_pde",
        "w_dir",
        "w_neu",
        "alpha",
        "max_weight_ratio",
        "max_weight_change",
    )
    int_weight_keys = ("balance_frequency", "min_balance_step", "window")

    for key in float_weight_keys:
        if key in params:
            weight_updates[key] = _format_float_literal(params[key], sig_figs)
    for key in int_weight_keys:
        if key in params:
            weight_updates[key] = str(int(round(float(params[key]))))

    if not assignment_updates and not weight_updates:
        raise ValueError(
            "Nenhum hiperparâmetro aplicável foi encontrado no arquivo de entrada."
        )

    updated_text = config_text
    for assignment, literal in assignment_updates.items():
        updated_text = _replace_assignment_line(updated_text, assignment, literal)
    for key, literal in weight_updates.items():
        updated_text = _replace_weight_param(updated_text, key, literal)

    changed = updated_text != config_text
    if changed:
        target_config.write_text(updated_text, encoding="utf-8")

    return {
        "best_path": str(best_file),
        "config_path": str(target_config),
        "sig_figs": sig_figs,
        "changed": changed,
        "applied_assignments": assignment_updates,
        "applied_weight_params": weight_updates,
    }

