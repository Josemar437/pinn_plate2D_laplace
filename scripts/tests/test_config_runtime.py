"""
Testes da inicialização explícita de runtime em config.
"""

from __future__ import annotations

import logging

import torch

import config as config_module


def test_initialize_runtime_sets_dtype_logging_and_state(tmp_path):
    log_file = tmp_path / "runtime.log"
    logger, resolved_device = config_module.initialize_runtime(
        log_file=log_file,
        emit_device_banner=False,
        force=True,
    )

    expected_dtype = (
        torch.float64 if config_module.Config.USE_DOUBLE else torch.float32
    )
    assert torch.get_default_dtype() == expected_dtype
    assert config_module.runtime_initialized() is True
    assert isinstance(resolved_device, torch.device)
    assert log_file.exists()
    assert any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
