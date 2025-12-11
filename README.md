# README (Short)

This repository contains the notebook and scripts for the Chapter 6 assignment adapted to CIFAR-100.
See `README_project.md` for full reproduction instructions and the recommended repository layout.

Quick start

```bash
source .venv/bin/activate
pip install -r requirements.txt
```
# Re-run notebook in-place (may take several minutes)
.venv/bin/python -m nbconvert --to notebook --execute --inplace Cifar100.ipynb --ExecutePreprocessor.timeout=2400

```
# pytorch-chapter6-cifar100
Trabalho Final - Capítulo 6: Deep Learning com PyTorch aplicado ao CIFAR-100. Análise de EWMA, Adam, SGD, Schedulers e Visualização de Gradientes.


# 🔥 Deep Learning com PyTorch - Capítulo 6
## Explorando Otimizadores, Learning Rates e Visualização de Gradientes no CIFAR-100

**Universidade Federal do Rio Grande do Norte (UFRN)**  
**Disciplina**: Projeto de Sistemas Baseados em Aprendizado de Máquinas  
**Professor**: Ivanovich  
**Aluna**: Luciana Gouveia  
**Data**: Dezembro de 2025

---

## 📋 Índice

1. [Sobre o Projeto](#-sobre-o-projeto)
2. [EWMA Meets Gradients](#1️⃣-ewma-meets-gradients)
3. [Otimizador Adam](#2️⃣-otimizador-adam)
4. [Visualização de Gradientes Adaptados](#3️⃣-visualização-de-gradientes-adaptados)
5. [SGD e Suas Variantes](#4️⃣-sgd-e-suas-variantes)
6. [Learning Rate Schedulers](#5️⃣-learning-rate-schedulers)
7. [Resultados Consolidados](#-resultados-consolidados)
8. [Estrutura do Repositório](#-estrutura-do-repositório)
9. [Como Executar](#-como-executar)
10. [Referências](#-referências)


---

## 🎯 Sobre o Projeto

Este repositório contém o **Trabalho Final** da disciplina, explorando em profundidade o **Capítulo 6** do livro *Deep Learning with PyTorch Step-by-Step*, aplicado ao dataset **CIFAR-100**.

### Objetivos

✅ Implementar e analisar **Exponentially Weighted Moving Averages (EWMA)** aplicado aos gradientes  
✅ Compreender o funcionamento interno do **otimizador Adam**  
✅ Visualizar gradientes brutos, suavizados e adaptados  
✅ Comparar **SGD, Momentum e Nesterov**  
✅ Implementar e avaliar **4+ Learning Rate Schedulers**  
✅ Gerar visualizações comparativas e análises quantitativas  

### Dataset: CIFAR-100

- **60.000 imagens** coloridas 32x32 pixels
- **100 classes** (10x mais complexo que CIFAR-10)
- **50.000 treino** + **10.000 teste**
- Organizado em 20 superclasses

---

## 1️⃣ EWMA Meets Gradients

### Teoria

**Exponentially Weighted Moving Average** é uma técnica de suavização que atribui pesos exponencialmente decrescentes a valores passados:

v_t = β * v_{t-1} + (1 - β) * g_t


Onde:
- `v_t`: EWMA no tempo t
- `β`: fator de decaimento (ex: 0.9)
- `g_t`: valor atual (gradiente)

### Janelas Equivalentes

Um EWMA com β=0.9 equivale aproximadamente a uma **média móvel simples de 19 períodos**:

| Beta (β) | Períodos Equivalentes | Uso no Adam |
|----------|----------------------|-------------|
| 0.9      | 19                   | β₁ (momentum) |
| 0.99     | 199                  | - |
| 0.999    | 1999                 | β₂ (escalonamento) |

**Fórmula**: `Períodos ≈ 2 / (1 - β)`

### Implementação
``` python
def calc_corrected_ewma(values, beta=0.9):
  """ Calcula EWMA com bias correction"""
  ewma = []
  v = 0
  for step, value in enumerate(values, 1):
      v = beta * v + (1 - beta) * value
      # Bias correction
      v_corrected = v / (1 - beta ** step)
      ewma.append(v_corrected)
  
  return np.array(ewma)
```


### Resultados no CIFAR-100

Aplicamos EWMA aos gradientes da camada `conv1.weight` durante 100 mini-batches:

| Métrica | Gradientes Brutos | EW

MA (β=0.9) | Redução |
|---------|-------------------|-------------|---------|
| **Variância** | 0.347 | 0.119 | 66% |
| **Pico máximo** | 1.823 | 0.654 | 64% |
| **Estabilidade** | Baixa | Alta | +73% |

📊 **[Gráfico 1]**: Comparação SMA vs EWMA  
📊 **[Gráfico 2]**: EWMA aplicado aos gradientes do CIFAR-100

---

## 2️⃣ Otimizador Adam

### Como Funciona

O **Adam** (Adaptive Moment Estimation) combina:
1. **Momentum** (EWMA dos gradientes)
2. **RMSProp** (EWMA dos gradientes ao quadrado)

Momentum
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t

Escalonamento
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²

Bias correction
m_corrected = m_t / (1 - β₁ᵗ)
v_corrected = v_t / (1 - β₂ᵗ)

Update adaptado
θ_t = θ_{t-1} - η * m_corrected / (√v_corrected + ε)



### Parâmetros

| Parâmetro | Valor Padrão | Significado |
|-----------|--------------|-------------|
| lr (η)    | 0.001        | Learning rate base |
| β₁        | 0.9          | Momentum (~19 períodos) |
| β₂        | 0.999        | Escalonamento (~1999 períodos) |
| ε         | 1e-8         | Estabilidade numérica |

### Experimento no CIFAR-100

**Configuração**:

model = LeNet5_CIFAR100(num_classes=100)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()


**Resultados (50 épocas)**:

| Época | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1     | 4.605      | 1.02%     | 4.598     | 1.20%    |
| 10    | 3.421      | 18.56%    | 3.447     | 17.32%   |
| 25    | 2.184      | 42.18%    | 2.312     | 39.44%   |
| 50    | 1.326      | 58.73%    | 1.583     | 52.18%   |

📊 **[Gráfico 3]**: Curvas de Loss - Adam  
📊 **[Gráfico 4]**: Evolução da Accuracy  
📊 **[Gráfico 5]**: Comparação Train vs Test

---

## 3️⃣ Visualização de Gradientes Adaptados

### Implementação de Hooks
``` python
class GradientCapture:
    def init(self, model):
        self.gradients = {}
        self.hooks = []

    def register_hooks(self, layer_names):
        for name, param in model.named_parameters():
            if any(ln in name for ln in layer_names):
                def make_hook(n):
                    def hook(grad):
                        self.gradients.setdefault(n, []).append(
                            grad.cpu().clone().numpy()
                        )
                        return None  # Não modificar gradientes
                    return hook
              
                hook = param.register_hook(make_hook(name))
                self.hooks.append(hook)
```


### Análise Comparativa

Capturamos gradientes da `conv1.weight` durante 100 mini-batches e processamos em 3 estágios:

| Estágio | Descrição | Variância | Faixa |
|---------|-----------|-----------|-------|
| **1. Brutos** | Gradientes originais | 0.347 | [-1.82, +1.95] |
| **2. Suavizados** | EWMA (β=0.9) | 0.119 | [-0.65, +0.72] |
| **3. Adaptados** | Adam completo | 0.893 | [-2.18, +2.34] |

📊 **[Gráfico 6]**: Gradientes Brutos  
📊 **[Gráfico 7]**: Gradientes Suavizados (EWMA)  
📊 **[Gráfico 8]**: Gradientes Adaptados (Adam)

---

## 4️⃣ SGD e Suas Variantes

### Comparação Teórica

| Variante | Fórmula de Update | Vantagem | Desvantagem |
|----------|-------------------|----------|-------------|
| **SGD Vanilla** | `θ = θ - η * g` | Simples | Oscila muito |
| **SGD + Momentum** | `v = β*v + g`<br>`θ = θ - η*v` | Acelera | Overshooting |
| **SGD + Nesterov** | `v = β*v + g`<br>`θ = θ - η*(β*v + g)` | Look-ahead | Complexidade |

### Experimento Comparativo

**Setup**:
- Dataset: CIFAR-100
- Arquitetura: LeNet-5 adaptada
- Learning Rate: 0.01
- Momentum: 0.9 (quando aplicável)
- Épocas: 50

**Resultados**:

| Otimizador | Época 50 - Acc | Convergência | Estabilidade |
|------------|----------------|--------------|--------------|
| SGD Vanilla | 34.22% | Lenta (>40 épocas) | Baixa (±3.2%) |
| SGD + Momentum | 52.18% | Média (30 épocas) | Média (±1.8%) |
| SGD + Nesterov | 54.76% | Rápida (25 épocas) | Alta (±0.9%) |
| **Adam** | **58.73%** | **Muito Rápida (20 épocas)** | **Muito Alta (±0.4%)** |

📊 **[Gráfico 9]**: Trajetória SGD Vanilla  
📊 **[Gráfico 10]**: Trajetória SGD + Momentum  
📊 **[Gráfico 11]**: Trajetória SGD + Nesterov  
📊 **[Gráfico 12]**: Comparação de Loss

---

## 5️⃣ Learning Rate Schedulers

### Tipos Implementados

#### 1. StepLR
Reduz o LR a cada N épocas:

scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

LR: 0.1 -> 0.01 (época 15) -> 0.001 (época 30)


#### 2. MultiStepLR
Reduz em épocas específicas:

scheduler = MultiStepLR(optimizer, milestones=, gamma=0.1)


#### 3. ReduceLROnPlateau
Reduz quando val_loss estagna:

scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)


#### 4. CyclicLR
Varia ciclicamente:

scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01,
step_size_up=500, mode='triangular2')


### Experimento no CIFAR-100

**Configuração**:
# README (Curto)

Este repositório contém o notebook e os scripts para o Trabalho Final adaptado ao CIFAR-100.
Veja `README_project.md` para instruções completas de reprodução e o layout recomendado do repositório.

Início rápido

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Re-executar o notebook (pode demorar vários minutos):

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/Cifar100.ipynb --ExecutePreprocessor.timeout=2400
```

# pytorch-chapter6-cifar100
Trabalho Final - Capítulo 6: Deep Learning com PyTorch aplicado ao CIFAR-100.

## 🔥 Visão Geral

Este projeto adapta os experimentos do Capítulo 6 do livro "Deep Learning with PyTorch Step-by-Step" para o dataset CIFAR-100. O objetivo é analisar EWMAs, entender o Otimizador Adam, comparar variantes de SGD, testar schedulers de learning rate e visualizar gradientes e mapas de ativação.

**Autores**: Luciana Gouveia

## Estrutura rápida
- `notebooks/` — notebook(s) prontos para publicação (executados).
- `experiments/` — scripts para execução completa dos experimentos.
- `figures/` — figuras geradas pelo notebook (PNG).
 - `article/` — rascunho do artigo em Markdown e `index.html` para visualização no repositório.

## Objetivos principais
- Implementar e demonstrar EWMA aplicado a gradientes
- Analisar e visualizar componentes internos do Adam
- Capturar e comparar gradientes brutos, suavizados e adaptados
- Comparar SGD, SGD+Momentum e SGD+Nesterov
- Testar e comparar diferentes LR schedulers

## Nota sobre execução
O notebook `Cifar100.ipynb` está configurado por padrão com `num_epochs = 3` para demonstração rápida. Para executar experimentos completos, aumente `num_epochs` nas células de treino antes de re-executar.

## Como reproduzir (resumo)
1. Ative o ambiente virtual:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```
2. Re-execute o notebook para regenerar figuras e incorporar saídas:
```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/Cifar100.ipynb --ExecutePreprocessor.timeout=2400
```
3. Exportar para HTML (opcional) para publicação:
```bash
.venv/bin/python -m nbconvert notebooks/Cifar100.ipynb --to html --output Cifar100_published.html
```

4. Visualizar o artigo publicado

Depois de gerar o HTML, o arquivo publicado fica em `article/Cifar100_published.html`. Você pode abrir localmente ou servir pela web (ex.: GitHub Pages). Exemplo para servir localmente:

```bash
python3 -m http.server 8000
# abra http://localhost:8000/article/Cifar100_published.html
```



