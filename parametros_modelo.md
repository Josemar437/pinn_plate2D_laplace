# Parâmetros do Modelo PINN Thermal 2D

## Arquitetura da Rede

| Parâmetro | Valor |
|-----------|-------|
| Tipo | `EnhancedThermalPINN` |
| Camadas | `[2, 144, 144, 144, 144, 1]` |
| Ativação | `tanh` |
| Inicialização | Xavier/Glorot |

## Dados de Treino (Gerados via FDM)

| Parâmetro | Valor |
|-----------|-------|
| Malha FDM | `121 x 121` |
| Pontos interiores (`X_f`) | `10.000` |
| Pontos por fronteira (`N_boundary`) | `600` |
| Pontos supervisionados (`X_data`) | `10.000` |
| Método de referência | SOR Red-Black |

## Hiperparâmetros de Treino

| Parâmetro | Valor |
|-----------|-------|
| Épocas Adam | `4000` |
| Iterações L-BFGS | `400` |
| Learning rate | `1.08e-3` |
| Gradient clipping | `1.0` |

## Pesos da Função de Perda

| Parâmetro | Valor |
|-----------|-------|
| `w_pde` | `1.00e0` |
| `w_dir` | `6.24e1` |
| `w_neu` | `2.24e2` |
| `w_data` | `2.89e0` |

## Configuração Física

| Parâmetro | Valor |
|-----------|-------|
| `T_LEFT` | `200 K` |
| `T_RIGHT` | `100 K` |
| `LX` | `1.0 m` |
| `LY` | `1.0 m` |

## Otimização Automática (Optuna)

| Parâmetro | Valor padrão |
|-----------|--------------|
| Trials | `30` |
| Faixa `lr` | `1e-4` a `5e-3` (log) |
| Faixa `w_data` | `1` a `500` (log) |
| Faixa `w_dir` | `50` a `1500` (log) |
| Faixa `w_neu` | `1` a `300` (log) |
| Largura oculta | `32` a `160` (step 16) |
| Profundidade oculta | `2` a `4` camadas |
