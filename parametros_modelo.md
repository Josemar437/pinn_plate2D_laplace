# Parâmetros do Modelo PINN Thermal 2D

## Arquitetura da Rede Neural

| Parâmetro | Valor |
|-----------|-------|
| **Tipo de Modelo** | `EnhancedThermalPINN` (MLP) |
| **Camadas** | `[2, 64, 64, 64, 1]` |
| **Número de camadas ocultas** | 3 |
| **Neurônios por camada oculta** | 64 |
| **Entrada** | 2 (coordenadas x, y) |
| **Saída** | 1 (temperatura) |
| **Função de ativação** | `tanh` |
| **Inicialização de pesos** | Xavier/Glorot Normal |

---

## Hiperparâmetros de Treinamento

| Parâmetro | Valor |
|-----------|-------|
| **Épocas Adam** | 4000 |
| **Iterações L-BFGS** | 400 |
| **Learning Rate inicial** | 1e-3 |
| **Scheduler** | `ReduceLROnPlateau` (patience=200, factor=0.5) |
| **Gradient Clipping** | max_norm=1.0 |
| **L-BFGS history_size** | 50 |

---

## Dados de Treinamento (Pontos de Colocação)

| Parâmetro | Valor |
|-----------|-------|
| **Pontos interiores** | 10.000 |
| **Pontos de contorno** | 600 |
| **Estratégia de amostragem** | Sobol (quasi-random) |

---

## Pesos Adaptativos da Loss

| Parâmetro | Valor |
|-----------|-------|
| **w_pde** (peso EDP) | 1.0 |
| **w_dir** (Dirichlet) | 400.0 |
| **w_neu** (Neumann) | 40.0 |
| **Frequência de balanceamento** | 100 épocas |
| **Alpha (suavização)** | 0.9 |
| **Razão máxima de pesos** | 1000.0 |
| **Estratégia** | `loss_balance` |

---

## Condições Físicas do Problema

| Parâmetro | Valor |
|-----------|-------|
| **T_LEFT** (Dirichlet esquerda) | 200.0 K |
| **T_RIGHT** (Dirichlet direita) | 100.0 K |
| **Lx** (comprimento em x) | 1.0 m |
| **Ly** (comprimento em y) | 1.0 m |

---

## Configurações de Precisão

| Parâmetro | Valor |
|-----------|-------|
| **Precisão** | `float64` (double) |
| **Seed (reprodutibilidade)** | 42 |
| **TF32 (CUDA)** | Desabilitado |

---

## Visualização

| Parâmetro | Valor |
|-----------|-------|
| **Domain size (grid)** | 150×150 |
| **DPI** | 600 |
