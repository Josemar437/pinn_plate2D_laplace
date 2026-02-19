# PINN Thermal 2D

Physics-Informed Neural Network (PINN) para condução térmica bidimensional em regime permanente, com validação quantitativa contra uma solução de referência por diferenças finitas (FDM).


## Resumo

Este trabalho resolve a equação de Laplace em uma placa retangular com contornos mistos:
- Dirichlet nas fronteiras verticais.
- Neumann homogêneo nas fronteiras horizontais.

A metodologia combina:
- PINN com perdas físicas não-dimensionalizadas.
- Supervisão por amostras da solução FDM.
- Otimização híbrida (`Adam` + `L-BFGS` + etapa opcional de `physics-polish`).
- Rebalanceamento adaptativo de pesos da loss (`grad_norm_balance`).
- Amostragem adaptativa de colocation points (RAR-lite).
- Métricas de contorno não-dimensionais separadas para Dirichlet e Neumann.
- Diagnóstico de convergência física contínuo em todas as fases de otimização.

## Problema Físico

Domínio:
- $\Omega = (0, L_x) \times (0, L_y)$

Equação governante:
- $\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0, \ \forall (x,y)\in\Omega$

Condições de contorno:
- $T(0,y)=T_{left}$
- $T(L_x,y)=T_{right}$
- $\frac{\partial T}{\partial y}(x,0)=0$
- $\frac{\partial T}{\partial y}(x,L_y)=0$

## Formulação da Loss

Loss total:

```text
L_total = w_pde * L_PDE + w_dir * L_D + w_neu * L_N + w_data * L_data + w_curv * L_curv
```

Componentes:
- `L_PDE`: residual da EDP no interior.
- `L_D`: erro Dirichlet em `x=0` e `x=Lx`.
- `L_N`: erro Neumann em `y=0` e `y=Ly`.
- `L_data`: erro supervisionado versus FDM.
- `L_curv`: regularização de curvatura na linha média (`y=Ly/2`).

Detalhes relevantes para estabilidade:
- Não-dimensionalização por escala de temperatura (`|T_left - T_right|`) e comprimento (`max(Lx, Ly)`).
- Penalização de cauda para hotspots do residual PDE (top-k): `PDE_TOPK_FRACTION`, `PDE_TOPK_WEIGHT`.
- Warmup do peso PDE no início do treino: `PDE_WARMUP_EPOCHS`, `PDE_WARMUP_START_FACTOR`.
- Controle de assimetria Dirichlet esquerda/direita por MAE não-dimensional (`L_D_right_mae_nd / L_D_left_mae_nd`).

## Metodologia Numérica

### Referência FDM
- Implementada em `src/fdm.py` (SOR para Laplace com contornos mistos).
- Usada para gerar:
  - alvo supervisionado (`L_data`);
  - benchmark quantitativo das métricas.

### PINN
- Implementada em `src/models.py` (`EnhancedThermalPINN`).
- Arquitetura default em `src/config.py`:
  - camadas: `[2, 128, 128, 128, 128, 128, 128, 128, 128, 1]`
  - ativação: `adaptive_tanh`
- Normalização de entrada no forward para `[-1,1]` por dimensão.
- Hard Constraint opcional para Dirichlet:
  - `ENABLE_HARD_DIRICHLET_CONSTRAINT=True`
  - transformação arquitetural: `T(x,y)=T_lin(x) + x*(Lx-x)*N(x,y)`
  - garante por construção: `T(0,y)=T_left` e `T(Lx,y)=T_right`.

### Otimização
- Fase 1: `Adam` com clipping de gradiente + scheduler de LR.
- Scheduler configurável (`LR_SCHEDULER_MODE`):
  - `cosine` (default atual)
  - `plateau`
  - `none`
- Fase 2: `L-BFGS`.
- Fase 3 (opcional): `Physics-Polish L-BFGS` com reforço de `w_pde`.
- `L_PDE` é registrado continuamente nas três fases (`Adam`, `L-BFGS`, `physics-polish`) para auditoria física consistente.
- Em avaliação, operadores diferenciais usam `create_graph=False` para reduzir custo de memória/tempo.

### Balanceamento Adaptativo de Pesos
- Classe: `AdaptiveWeightScheduler` em `src/losses.py`.
- Estratégias disponíveis:
  - `loss_balance`
  - `inverse_loss`
  - `grad_norm_balance` (default atual)
- `grad_norm_balance` usa estatísticas de norma de gradiente de `L_PDE`, `L_D`, `L_N` para reduzir competição entre termos.
- O ajuste esquerda/direita de Dirichlet usa MAE não-dimensional e pode incorporar um fator NTK (proxy por norma de gradiente de `L_D_left` e `L_D_right`):
  - `ENABLE_DIRICHLET_NTK_BALANCE=True`
  - `DIRICHLET_NTK_BALANCE_FREQUENCY=25`
  - `DIRICHLET_NTK_BLEND=0.5`

### Amostragem de Pontos
- `src/sampling.py`:
  - interior: `fdm`, `lhs`, `sobol`, `hybrid`
  - importance sampling próximo a `x=Lx`
  - importance sampling opcional próximo a `x=0`
  - reforço de Dirichlet à direita (`RIGHT_DIRICHLET_MULTIPLIER`)
- `src/training.py`:
  - RAR-lite periódico com candidatos focados em domínio global, cantos e faixa direita.
  - seleção suavizada de pontos de alto residual (pool top-k com amostragem probabilística).
  - recomendado manter `ENABLE_RAR=True` durante o polimento com `L-BFGS` para reduzir hotspots físicos locais.
  - reforço de cantos configurado em `RAR_CORNER_FRACTION=0.7` (default atual).

## Configuração Experimental (Default)

Parâmetros centrais em `src/config.py`:
- Física: `LX=1.0`, `LY=1.0`, `T_LEFT=200.0`, `T_RIGHT=100.0`
- FDM: `FDM_NX=121`, `FDM_NY=121`, `FDM_TOL=1e-8`, `FDM_OMEGA=1.7`
- Treino:
  - `DEFAULT_EPOCHS_ADAM=4000`
  - `DEFAULT_EPOCHS_LBFGS=500` (polimento final agressivo após Adam+RAR)
  - `DEFAULT_LR=1.59e-3`
  - `DEFAULT_N_INTERIOR=1024`, `DEFAULT_N_BOUNDARY=240`, `DEFAULT_N_DATA=3000`
  - `RIGHT_DIRICHLET_LOSS_ADAPT_RATE=0.2`
  - `RAR_CORNER_FRACTION=0.7`
- Rede:
  - `DEFAULT_ACTIVATION=adaptive_tanh`
  - `INPUT_NORMALIZATION=minus_one_one` (default atual para melhor condicionamento com `tanh`)
- Reprodutibilidade:
  - `SEED=42`
  - `USE_DOUBLE=True`

## Reprodutibilidade

### Ambiente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Execução principal

```bash
python main.py train
```

Atalho equivalente:

```bash
python main.py
```

Execução com diretórios dedicados para experimento:

```bash
python main.py train --output-dir runs/plots/paper --results-dir runs/results/paper --report-path runs/results/paper/scientific_report.txt
```

Execução com Hard Constraint de Dirichlet:

```bash
python main.py train --enable-hard-dirichlet-constraint --epochs-lbfgs 500
```

### Fluxo de tuning (Optuna)

```bash
python main.py tune --n-trials 40 --epochs-adam 300 --results-path runs/results/optuna_best.json
python main.py apply-best-config --best-path runs/results/optuna_best.json
python main.py train
```

Treino usando melhor configuração carregada em memória:

```bash
python main.py train-best --best-path runs/results/optuna_best.json
```

Objetivo adimensional utilizado no tuning:

```text
objective = w1 * relative_l2_error
          + w2 * pde_residual_l2_normalized
          + w3 * boundary_bc_error_nd
```

com:

```text
pde_residual_l2_normalized = pde_residual_l2 / pde_residual_l2_reference
```

Pesos padrão configuráveis em `src/config.py`:
- `TUNING_W_REL_L2`
- `TUNING_W_PDE_RESIDUAL`
- `TUNING_W_BOUNDARY_BC`

## Métricas Reportadas

Calculadas em `src/analytics.py`:
- Erro de campo: `MAE`, `RMSE`, `MAPE`, `R2`, `max_error`
- Erro relativo: `relative_l2_error`, `relative_l2_error_vs_fdm`
- Erros de contorno (não-dimensionais):
  - `boundary_dirichlet_error_nd`
  - `boundary_neumann_error_nd`
  - `boundary_bc_error_nd`
  - `boundary_bc_error_dirichlet_left`, `boundary_bc_error_dirichlet_right`
  - `boundary_bc_error_dirichlet_ratio_right_left`
- Residual PDE (dimensional): `pde_residual_mean`, `pde_residual_l2`, `pde_residual_max`
- Residual PDE não-dimensional: `pde_residual_nd_mean`, `pde_residual_nd_l2`, `pde_residual_nd_max`
- Residual PDE normalizado para tuning: `pde_residual_l2_reference`, `pde_residual_l2_normalized`

Compatibilidade retroativa:
- `boundary_bc_error` e `boundary_error` são mantidas como aliases da métrica ND combinada.
- Métricas legadas mistas ainda podem aparecer para diagnóstico: `boundary_bc_error_legacy_*`.
- Com `ENABLE_HARD_DIRICHLET_CONSTRAINT=True`, o erro Dirichlet tende ao limite numérico de máquina.

## Breaking Changes (2026-02-18)

- `boundary_bc_error` mudou de semântica: antes combinava grandezas com unidades diferentes (`K` e `K/m`), agora é alias de `boundary_bc_error_nd` (não-dimensional).
- O default de normalização de entrada mudou de `zero_one` para `minus_one_one` em `INPUT_NORMALIZATION`, melhorando o condicionamento com ativações tipo `tanh`.
- O objetivo do tuning com Optuna passou a ser totalmente adimensional (`relative_l2_error`, `pde_residual_l2_normalized`, `boundary_bc_error_nd`), reduzindo dependência de escala e malha.

Impacto de migração:
- Não compare diretamente valores históricos de `boundary_bc_error` (pré e pós 18/02/2026).
- Para reproduções antigas, force explicitamente `INPUT_NORMALIZATION=zero_one` no `Config`.
- Em análises de tuning, priorize `best_value_relative_l2` e os atributos físicos normalizados do trial.

Saídas salvas:
- `runs/results/<experimento>/metrics.json`
- `runs/results/<experimento>/training_history.json`
- `runs/results/<experimento>/scientific_report.txt`
- `runs/results/<experimento>/pde_residual.npy`
- `runs/results/<experimento>/reproducibility_snapshot.json`
- `runs/logs/pinn_thermal.log`

## Resultados Recentes

As métricas de contorno foram reformuladas para forma não-dimensional.  
Comparações históricas com rodadas anteriores a essa mudança devem usar cautela, pois os valores antigos de `boundary_bc_error` combinavam grandezas com unidades diferentes.

Para reporte científico nas versões atuais, priorizar:
- `relative_l2_error`
- `pde_residual_nd_mean` e `pde_residual_nd_max`
- `pde_residual_l2_normalized`
- `boundary_bc_error_nd` e razão de assimetria `boundary_bc_error_dirichlet_ratio_right_left`

## Gráficos Gerados 

Geradas por `src/plotting.py`:
- `fieldComparison.png`
- `physicsConsistency.png`
- `trainingCurves.png`
- `trainingHistory.png`
- `ensemblePredictions.png`
- `uncertaintyAnalysis.png`

Para discussão no paper, priorizar:
- mapa de `|residual PDE|` e histogramas (`physicsConsistency.png`)
- evolução temporal dos termos de perda (`trainingCurves.png`)
- comparação de campo PINN vs FDM (`fieldComparison.png`)

## Estrutura do Repositório

```text
main.py
src/
  analytics.py
  config.py
  fdm.py
  losses.py
  models.py
  operators.py
  pipeline.py
  plotting.py
  sampling.py
  training.py
scripts/
  run_tests.py
  tests/
orphaned/
  legacy_entrypoints/
  legacy_shims/
runs/
  logs/
  plots/
  results/
```

## Testes

```bash
python -m pytest -q
```

ou

```bash
python scripts/run_tests.py
```

## Limitações e Ameaças à Validade

- O problema atual é estacionário e com propriedades homogêneas.
- Métricas globais altas podem mascarar hotspots de residual PDE.
- Resultados podem variar com seed e estratégia de amostragem.
- Em ambiente CPU-only, custo computacional aumenta para malhas e redes maiores.

## Diretrizes de Reporte para Paper

Recomendado reportar:
- versão do código e `src/config.py` usado;
- hardware, versão do PyTorch e disponibilidade CUDA;
- número de épocas por fase (`Adam`, `L-BFGS`, `physics-polish`);
- métricas de erro de campo + residual PDE dimensional/não-dimensional/normalizado;
- snapshot de reprodutibilidade (`reproducibility_snapshot.json`);
- pelo menos 3 execuções com seeds distintas para média e desvio padrão.

## Citação


Template BibTeX:

```bibtex
@software{pinn_thermal_2d_2026,
  title   = {PINN Thermal 2D},
  author  = {Josemar Rocha Pedroso; Camila Borges; Dra. Nuccia Carla Arruda de Souza},
  year    = {2026},
  url     = {https://github.com/Josemar437/pinn_plate2D_laplace.git},
}
```
