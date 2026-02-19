"""
Atalho para execução da suíte de testes automatizados do projeto.

Projeto: PINN Thermal 2D
Autores:
    - Josemar Rocha Pedroso
    - Camila Borge
    - Nuccia Carla Arruda de Souza
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Executa pytest no diretório raiz do repositório e retorna o código de saída."""
    project_root = Path(__file__).resolve().parents[1]
    return subprocess.call([sys.executable, "-m", "pytest"], cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())
