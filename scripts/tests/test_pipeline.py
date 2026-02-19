"""
Testes de utilitários de pipeline e snapshot de reprodutibilidade.
"""

from __future__ import annotations

import torch

import pipeline as pipeline_module


def test_reproducibility_snapshot_uses_centralized_physical_scales(monkeypatch):
    monkeypatch.setattr(pipeline_module, "physical_scales", lambda: (321.0, 4.5))

    snapshot = pipeline_module._build_reproducibility_snapshot(
        active_device=torch.device("cpu")
    )

    assert snapshot["scales"]["temp_scale"] == 321.0
    assert snapshot["scales"]["length_scale"] == 4.5
    assert snapshot["reproducibility"]["device"] == "cpu"
