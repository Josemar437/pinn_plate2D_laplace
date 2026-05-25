# PINN 2D - Equação de Laplace com Condições de Contorno Mistas

Solução por Physics-Informed Neural Networks (PINNs) para o problema de difusão térmica 2D em regime permanente, regido pela equação de Laplace com condições de Dirichlet nas laterais e Neumann homogêneo (adiabático) nas bordas superior e inferior.

## Estrutura do Projeto

```
PINN1/
├── config.py              # Configurações físicas e de treinamento
├── sampling.py            # Estratégias de colocation (Sobol, uniforme)
├── models.py              # Arquitetura neural com hard constraints
├── operators.py           # Operadores diferenciais (Laplaciano, derivadas)
├── losses.py              # Funções de perda multi-objetivo
├── training.py            # Pipeline Adam + L-BFGS
├── analytics.py           # Análise de erros e convergência
├── plotting.py            # Visualizações de publicação
├── main.py                # Orquestração principal
└── README.md              # Este arquivo
```

## Problema Físico

**Equação Governante** (Regime Permanente):
$$\nabla^2 T = 0 \quad \text{em } \Omega = [0, L_x] \times [0, L_y]$$

**Condições de Contorno**:
- **Laterais (Dirichlet)**: $T(0, y) = T_{LEFT}$, $T(L_x, y) = T_{RIGHT}$
- **Bordas H/V (Neumann)**: $\frac{\partial T}{\partial n} = 0$ (adiabático)

## Características Implementadas

### Arquitetura Neural Avançada

1. **Hard Constraints Dirichlet**: Satisfação exata via design da rede
   - Penalidade polinomial: $\text{penalty}(x) = x_{norm}(1-x_{norm})$
   - Garante $T(0,y) = T_{LEFT}$ e $T(L_x,y) = T_{RIGHT}$ automaticamente

2. **Imposição Suave Neumann**: Minimização de gradiente normal
   - $L_N = \mathbb{E}[(\partial T/\partial y)^2]$ nas bordas

3. **Treinamento Híbrido**:
   - Fase 1: Adam (4000 épocas) com scheduler adaptativo
   - Fase 2: L-BFGS (400 iterações) para refinamento

### Otimizações de Convergência

- **Ponderação Adaptativa**: Balanceamento automático PDE/BC via histórico
- **Clipping de Gradiente**: Estabilidade numérica
- **LR Scheduler**: ReduceLROnPlateau

## Instalação

### Ambiente Recomendado

```bash
# Criar ambiente virtual
python -m venv .venv
.\.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### Dependências Essenciais

```
torch>=2.0.0
numpy
matplotlib>=3.8
seaborn
```

## Uso Básico

### Execução Simples

```python
from main import main_enhanced

# Executar análise completa
net, analyzer, hist = main_enhanced()
```

### Configuração Personalizada

```python
from config import Config
from models import create_model
from sampling import create_training_data
from training import train_pinn_enhanced

# Modificar configurações
Config.DEFAULT_EPOCHS_ADAM = 5000
Config.DEFAULT_LR = 5e-4

# Criar dados
X_f, X_dir, T_dir_target, X_bottom, X_top = create_training_data(
    N_interior=12000,
    N_boundary=300,
    interior_strategy="sobol"
)

# Criar modelo personalizado
net = create_model(
    model_type="enhanced",
    layers=[2, 128, 128, 128, 1],
    activation='swish',
    use_residual=True
)

# Treinar
hist = train_pinn_enhanced(net, X_f, X_dir, T_dir_target, X_bottom, X_top)
## Uso Básico

### Execução Padrão

```bash
python main.py
```

Executa o pipeline completo:
1. Amostragem de pontos interior e contorno via Sobol
2. Treinamento PINN com hard constraints Dirichlet
3. Análise de convergência e erros
4. Plotagem de soluções e campos

### Configuração Personalizada

```python
from config import Config
Config.HARD_CONSTRAINT_BC = True   # Ativar hard constraints nas laterais
Config.DEFAULT_N_INTERIOR = 10000  # Aumentar densidade de colocation
Config.DEFAULT_EPOCHS_ADAM = 5000  # Mais épocas Adam
```

## Resultados Esperados

### Indicadores de Performance

- **MAE**: < 1e-3 K (erro absoluto médio)
- **RMSE**: < 5e-3 K
- **R²**: > 0.9999
- **Tempo**: ~2 min (GPU), ~10 min (CPU)

### Verificação de Condições de Contorno

- **Dirichlet**: Satisfeito automaticamente (hard constraint)
- **Neumann**: $\max|\partial T/\partial y| < 1e-4$ (bordas isoladas)
