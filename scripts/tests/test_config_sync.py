"""
Testes de sincronização de parâmetros do Optuna para config.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import src.config_sync as config_sync_module


def test_default_config_path_points_to_src_config():
    resolved = config_sync_module._default_config_path()
    assert resolved.name == "config.py"
    assert resolved.parent.name == "src"
    assert resolved.exists()


def test_apply_best_to_config_file_works_without_explicit_args(tmp_path, monkeypatch):
    project_root = tmp_path
    runs_results = project_root / "runs" / "results"
    runs_results.mkdir(parents=True, exist_ok=True)
    best_payload = {
        "best_params": {
            "lr": 2.5e-3,
            "w_data": 1.5,
            "w_dir": 120.0,
            "w_neu": 25.0,
            "alpha": 0.93,
            "max_weight_ratio": 220.0,
            "max_weight_change": 0.2,
            "balance_frequency": 80,
            "min_balance_step": 300,
            "window": 20,
            "use_adaptive_weights": True,
        }
    }
    (runs_results / "optuna_best.json").write_text(
        json.dumps(best_payload), encoding="utf-8"
    )

    config_file = project_root / "src" / "config.py"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                "from typing import Any, Dict",
                "",
                "class Config:",
                "    HYPERPARAM_SIG_FIGS: int = 3",
                "    DEFAULT_LR: float = 1.00e-3",
                "    DEFAULT_W_DATA: float = 1.00e0",
                "    USE_ADAPTIVE_WEIGHTING: bool = False",
                "    DEFAULT_LAYERS: list[int] = [2, 64, 64, 1]",
                "    DEFAULT_WEIGHT_PARAMS: Dict[str, Any] = {",
                '        "w_pde": 1.00e0,',
                '        "w_dir": 1.00e2,',
                '        "w_neu": 1.00e1,',
                '        "balance_frequency": 60,',
                '        "alpha": 9.50e-1,',
                '        "max_weight_ratio": 3.00e2,',
                '        "min_balance_step": 100,',
                '        "window": 10,',
                '        "max_weight_change": 1.50e-1,',
                "    }",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(project_root)
    monkeypatch.setattr(config_sync_module, "_default_config_path", lambda: config_file)

    result = config_sync_module.apply_best_to_config_file()

    updated_text = config_file.read_text(encoding="utf-8")
    assert result["changed"] is True
    assert Path(result["config_path"]) == config_file
    assert "DEFAULT_LR: float = 2.50e-3" in updated_text
    assert '        "w_dir": 1.20e2,' in updated_text
