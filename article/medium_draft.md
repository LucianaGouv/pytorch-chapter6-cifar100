# Rascunho (polido) — EWMA, Otimizadores e Schedulers no CIFAR-100

**Resumo**

Este texto sumariza os experimentos aplicados do Capítulo 6 (Deep Learning with PyTorch) ao CIFAR‑100. Abordamos: (i) o papel das médias móveis exponenciais (EWMA) na análise de gradientes; (ii) o funcionamento interno do Adam; (iii) comparações entre Adam, SGD (com e sem momentum) e Nesterov; (iv) LR range test e estratégias de *scheduling* de learning rate; (v) visualizações de mapas de ativação e gradientes.

O notebook reprodutível e os scripts estão no repositório: `notebooks/Cifar100.ipynb`. As figuras são servidas publicamente via GitHub Pages e, para garantir que o Medium importe corretamente as imagens, as links abaixo apontam para a URL pública do projeto.

---

## Introdução

Este trabalho objetiva esclarecer como suavizações (EWMA) e algoritmos adaptativos influenciam a dinâmica de treinamento em um problema de classificação com 100 classes (CIFAR‑100). As demonstrações são curtas e interpretativas — suficientes para ilustrar comportamento e permitir reprodução rápida; para resultados finais, execute mais épocas conforme indicado.

## Metodologia (resumo)

- Arquitetura: LeNet‑like adaptado para CIFAR‑100 (implementação em `notebooks/Cifar100.ipynb`).
- Ferramentas: PyTorch, torchvision, matplotlib, nbconvert.
- Técnicas destacadas: EWMA (com e sem correção de viés), captura de gradientes via hooks, LR range test, StepLR / ReduceLROnPlateau / CyclicLR, visualização de mapas de ativação.

## Configuração dos demos

Parâmetros usados nas execuções curtas mostradas aqui:

- `batch_size`: 128
- `num_epochs` (demos): 3
- `Adam lr`: 1e-3
- `SGD lr`: 1e-2
- `momentum`: 0.9
- `β₁` / `β₂` (Adam): 0.9 / 0.999
- LR range test: 1e-6 → 1.0

> Observação: os valores acima são para demonstração rápida; aumente `num_epochs` e batches para figuras finais.

## Experimentos e figuras (para publicação)

As imagens abaixo apontam para as versões públicas hospedadas via GitHub Pages (ex.: `https://LucianaGouv.github.io/pytorch-chapter6-cifar100/figures/<nome>.png`). Se for importar para o Medium, usar a URL pública na opção “Import a story” facilitará trazer as imagens automaticamente.

#### Figura 1 — LR range test
![LR range test](https://LucianaGouv.github.io/pytorch-chapter6-cifar100/figures/lr_range_test.png)
**Legenda:** Varredura de learning rate para identificar uma ordem de grandeza segura para iniciar o treino; o ponto onde a perda começa a decair indica bons candidatos para o LR.

#### Figura 2 — Adam vs SGD (demo curto)
![Adam vs SGD](https://LucianaGouv.github.io/pytorch-chapter6-cifar100/figures/adam_vs_sgd_demo.png)
**Legenda:** Comparação das curvas de perda (loss) em uma execução curta. Adam acelera a redução da perda no início; a comparação a longo prazo exige mais épocas.

#### Figura 3 — Gradientes: brutos / EWMA / Adam-adapted
![Gradientes capture](https://LucianaGouv.github.io/pytorch-chapter6-cifar100/figures/gradients_capture_demo.png)
**Legenda:** Gradientes brutos, suavisados por EWMA (β=0.9) e vetores adaptados utilizados pelo Adam. A EWMA mostra a tendência subjacente enquanto os adaptadores reescalam o passo por parâmetro.

#### Figura 4 — Feature maps (todas as camadas)
![Feature maps](https://LucianaGouv.github.io/pytorch-chapter6-cifar100/figures/feature_maps_all_layers.png)
**Legenda:** Mapas de ativação extraídos durante uma inferência — úteis para análise qualitativa de respostas de filtros.

#### Figura 5 — Scheduler examples
![Schedulers](https://LucianaGouv.github.io/pytorch-chapter6-cifar100/figures/scheduler_examples.png)
**Legenda:** Exemplos comparativos de políticas de redução de LR: `StepLR` vs `ReduceLROnPlateau`.

#### Figura 6 — SGD Momentum vs Nesterov
![SGD vs Nesterov](https://LucianaGouv.github.io/pytorch-chapter6-cifar100/figures/sgd_nesterov_demo.png)
**Legenda:** Comparação entre SGD+momentum e Nesterov (look-ahead). Em muitos casos Nesterov fornece uma resposta mais estável.

## Interpretações resumidas

- EWMA é eficaz para reduzir ruído dos gradientes e, com correção de viés, produz estatísticas comparáveis às usadas por Adam.
- Adam tende a convergir mais rápido nas fases iniciais; SGD + momentum / Nesterov pode alcançar melhores soluções em alguns cenários com ajuste de LR.
- O LR range test é uma ferramenta prática para escolher a ordem de grandeza do LR inicial.

## Reprodutibilidade

1. Clone o repositório e ative o virtualenv:

```bash
git clone https://github.com/LucianaGouv/pytorch-chapter6-cifar100.git
cd pytorch-chapter6-cifar100
source .venv/bin/activate
pip install -r requirements.txt
```

2. Re-execute o notebook para regenerar as figuras (pode levar vários minutos):

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/Cifar100.ipynb --ExecutePreprocessor.timeout=2400
```

3. O notebook e scripts principais:

- `notebooks/Cifar100.ipynb` — notebook executável (artefato primário)
- `experiments/Chapter6_CIFAR100_Complete.py` — script para execução completa
- `scripts/` — demos e utilitários (Nesterov demo, re-execução em alta-dpi, etc.)

## Conclusão e próximos passos

Este rascunho está pronto para publicação: recomendo importar via URL pública do GitHub Pages (ex.: `https://LucianaGouv.github.io/pytorch-chapter6-cifar100/`) para preservar imagens e formatação. Os próximos passos ideais antes da submissão final:

- Rodar experimentos completos (mais épocas) e substituir figuras por versões finais em `figures/`.
- Revisar redação final e verificar referências/DOIs.
- Publicar no Medium via "Import a story" usando a URL pública e revisar a versão importada.

## Referências

- Kingma, D. P. & Ba, J., "Adam: A Method for Stochastic Optimization", ICLR 2015.
- Goodfellow, Bengio, Courville — Deep Learning (capítulos relevantes).

---

Se quiser, eu posso (A) ajustar o texto para o tom do Medium (mais narrativo), (B) gerar uma versão com imagens embutidas inline no HTML para garantir import perfeito, ou (C) fazer a importação guiada no Medium passo a passo.