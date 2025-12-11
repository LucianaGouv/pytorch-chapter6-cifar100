 # Nota Técnica — EWMA, Otimizadores e Schedulers aplicados ao CIFAR‑100

 **Resumo (executivo)**

 Este documento descreve, de forma técnica e reprodutível, os experimentos que adaptam o Capítulo 6 (Deep Learning with PyTorch) ao CIFAR‑100. Cobre teoria e implementação de EWMA, o funcionamento interno do Adam (momentos, correção de viés), comparações práticas entre Adam e variantes de SGD (momentum, Nesterov), LR range test e exemplos de schedulers. Todos os códigos e figuras estão disponíveis no repositório: `notebooks/Cifar100.ipynb` e `figures/`.

 ---

 ## 1. Motivação e objetivo

 Treinar modelos de visão computacional requer escolhas cuidadosas de otimizador e política de learning rate. Neste trabalho buscamos (i) explicar matematicamente as operações internas do Adam; (ii) conectar essas operações com EWMA; (iii) demonstrar empiricamente o efeito sobre gradientes e curvas de perda no CIFAR‑100; e (iv) indicar práticas reprodutíveis para escolher LR e scheduler.

 ## 2. EWMA: definição e relação com Adam

 Uma EWMA (Exponentially Weighted Moving Average) de uma sequência {g_t} é definida por

 $$v_t = \\beta v_{t-1} + (1-\\beta) g_t,\\qquad v_0 = 0$$

 onde $0\\le\\beta<1$. Reescrevendo em termos de fator de atualização $\\alpha = 1-\\beta$:

 $$v_t = (1-\\alpha) v_{t-1} + \\alpha g_t.$$ 

 Esta operação coloca peso exponencialmente decrescente no histórico e tem janela efetiva aproximada $2/(1-\\beta)$.

 No Adam, são mantidas duas EWMA paralelas:

 - First moment: $m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$ (momentum)
 - Second moment: $v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$ (escala)

 Como $m_0=v_0=0$, Adam aplica correção de viés para estimativas não viciadas:

 $$\\hat m_t = m_t/(1-\\beta_1^t),\\qquad \\hat v_t = v_t/(1-\\beta_2^t).$$

 O update é então:

 $$\\theta_{t+1} = \\theta_t - \\eta\\; \\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}. $$ 

 Comentário técnico: a presença de $\\sqrt{\\hat v_t}$ reescala cada dimensão do gradiente pela sua variação histórica, enquanto $\\hat m_t$ contém a direção suavizada. Isso explica o caráter adaptativo do Adam.

 ## 3. Implementação: utilitários e captura de gradientes

 No notebook incluímos utilitários reprodutíveis:

 - `calc_ewma(x, alpha)` — EWMA simples sobre um vetor.
 - `calc_corrected_ewma(x, alpha)` — EWMA com correção de viés (m_hat).
 - Hooks para capturar gradientes e ativações por camada; função `capture_gradients` que retorna: gradientes brutos, EWMA ao longo do vetor e vetores adaptados (m/\\sqrt{v}).

 Trecho de código (Python, PyTorch) — exemplo de captura de gradientes:

 ```python
 def capture_gradients(model, optimizer, criterion, inputs, labels, device):
     model.train(); optimizer.zero_grad()
     outputs = model(inputs.to(device))
     loss = criterion(outputs, labels.to(device))
     loss.backward()
     # ler p.grad para cada parâmetro
     grads = {name: p.grad.detach().cpu().numpy().ravel() for name,p in model.named_parameters() if p.grad is not None}
     return grads
 ```

 ## 4. Experimentos (configuração)

 - Dataset: CIFAR‑100 (50k treino / 10k teste). Normalização usada: mean=(0.5071,0.4867,0.4408), std=(0.2675,0.2565,0.2761).
 - Arquitetura: LeNet‑like adaptada (descrita no notebook).
 - Demos curtos: `num_epochs=3`, batch_size=128 (configuração para notebooks públicos); rodar com `num_epochs` maior para resultados finais.
 - Hiperparâmetros importantes: Adam lr=1e‑3 (betas=(0.9,0.999)), SGD lr=1e‑2 (momentum=0.9).

 ## 5. Resultados (resumo e interpretação)

 - LR range test: gráfico `lr_range_test.png` usado para escolher LR base. Interprete a região onde a loss começa a decair como ponto inicial para buscas finas.
 - Adam vs SGD (demo): `adam_vs_sgd_demo.png` mostra convergência inicial mais rápida do Adam. Em execuções longas com regularização e ajuste fino, SGD+momentum pode competir ou superar dependendo do regime.
 - Gradientes: `gradients_capture_demo.png` — compare vetores brutos, EWMA e m/\\sqrt{v}. Observa-se que a EWMA suprime picos e evidencia tendência; o adaptador do Adam reescala por dimensão.

 Interpretação prática: usar EWMA (β1≈0.9) para detectar direções consistentes nos gradientes e LR range test para selecionar a ordem de grandeza do passo inicial; combine com scheduler (ex.: StepLR ou ReduceLROnPlateau) conforme validação.

![LR range test](figures/lr_range_test.png)
*Figura: LR range test — use para escolher a ordem de grandeza do learning rate.*

![Adam vs SGD demo](figures/adam_vs_sgd_demo.png)
*Figura: Comparação curta Adam vs SGD+momentum (3 épocas).* 

![Gradientes e EWMA](figures/gradients_capture_demo.png)
*Figura: Gradientes crus vs EWMA vs vetor adaptado (m/sqrt(v)).*

 ## 6. LR schedulers — recomendações e exemplos rápidos

 - `StepLR(optimizer, step_size=k, gamma=0.1)` — fácil e robusto para cortes em marcos.
 - `ReduceLROnPlateau(optimizer, patience=p)` — usar quando val_loss estagnou.
 - `CyclicLR` — útil em fases de descoberta de LR, não recomendado como política final sem cuidado.

 Exemplo curto (PyTorch):

 ```python
 scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
 for epoch in range(E):
     train_one_epoch(...)
     scheduler.step()
 ```

 ## 7. Reprodutibilidade e entrega

 - Notebooks: `notebooks/Cifar100.ipynb` (versão executada). Para reproduzir: clonar o repositório, ativar `.venv` e rodar `nbconvert --execute` (com timeout suficiente).
 - Figuras finais: `figures/` — versões em alta resolução (DPI=300).

 ## 8. Conclusão curta

 EWMA e os componentes internos do Adam oferecem duas alavancas interpretáveis para entender e controlar treinamento: suavização e reescalamento. A combinação de LR range test + schedulers e análise de gradientes dá um fluxo prático para selecionar hiperparâmetros e diagnosticar problemas de otimização.

 ## 9. Referências

 - Kingma, D. P. & Ba, J., "Adam: A Method for Stochastic Optimization", ICLR 2015.
 - Goodfellow, Bengio, Courville — Deep Learning.

 ---

 Este arquivo será transformado em HTML 100% embutido para import no Medium; se quiser, eu adiciono exemplos numéricos mais detalhados (tabelas de treino/val por época) ou expando a seção de derivação matemática.

## Apêndice A — Derivação da correção de viés no Adam

Considere o momento de primeira ordem definido recursivamente por

$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t,\\qquad m_0 = 0.$$ 

Expandindo recursivamente:

$$m_1=(1-\\beta_1)g_1\\\\
m_2=\\beta_1(1-\\beta_1)g_1 + (1-\\beta_1)g_2\\\\
m_t=(1-\\beta_1)\\sum_{i=1}^{t}\\beta_1^{t-i} g_i$$

Tomando expectativa (assumindo ruído com esperança zero ou estatísticas estacionárias), temos

$$E[m_t] = (1-\\beta_1)\\sum_{i=1}^{t}\\beta_1^{t-i} E[g_i].$$

Se $E[g_i]=\\mu$ constante, então

$$E[m_t] = (1-\\beta_1)\\mu \\sum_{k=0}^{t-1}\\beta_1^{k} = \\mu(1-\\beta_1)\\frac{1-\\beta_1^{t}}{1-\\beta_1} = \\mu(1-\\beta_1^{t}).$$

Portanto, o estimador $m_t$ está viciado para valores pequenos de $t$ (pois $E[m_t] \\approx \\mu(1-\\beta_1^{t}) < \\mu$). Dividindo por $(1-\\beta_1^{t})$ obtemos uma estimativa não-viciada:

$$\\hat m_t = m_t/(1-\\beta_1^{t}).$$

Análogo para $v_t$ (segunda momento) obtemos:

$$v_t = (1-\\beta_2)\\sum_{i=1}^{t}\\beta_2^{t-i} g_i^2,\\qquad \\hat v_t = v_t/(1-\\beta_2^{t}).$$

Este é o motivo matemático direto para a correção de viés do Adam.

## Apêndice B — Janela efetiva da EWMA

Para uma EWMA com parâmetro $\\beta$ (ou fator de atualização $\\alpha=1-\\beta$), a contribuição relativa de um gradiente antigo decai exponencialmente em $\\beta^k$ após $k$ passos. Uma heurística para a "janela efetiva" é:

$$ W_{eff} \\approx \\frac{2}{1-\\beta}. $$

Exemplo: com $\\beta=0.9$ temos $W_{eff}\approx 20$ passos; com $\\beta=0.99$, $W_{eff}\approx 200$ passos. Isso ajuda a escolher valores de $\\beta$ conforme a escala temporal desejada para suavização.

## Apêndice C — Tabelas por época e como gerar automaticamente (template)

Recomendo inserir uma tabela por época com pelo menos: `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr`. No notebook há pontos onde armazenamos esses valores durante o loop de treino. Exemplo de template em Markdown:

| epoch | train_loss | train_acc | val_loss | val_acc | lr |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.10 | 24.3% | 1.98 | 27.1% | 0.0100 |
| 2 | 1.85 | 31.0% | 1.90 | 29.4% | 0.0100 |
| 3 | 1.67 | 36.5% | 1.78 | 33.2% | 0.0100 |

Observação: os números acima são exemplos. Para gerar a tabela a partir do seu loop de treino real, use o snippet abaixo no final do treino.

```python
# Exemplo: cole este snippet no final do seu loop de treino
import pandas as pd

history = {
    'epoch': epochs_list,            # lista de ints
    'train_loss': train_loss_list,   # lista de floats
    'train_acc': train_acc_list,     # lista de floats (0-100)
    'val_loss': val_loss_list,
    'val_acc': val_acc_list,
    'lr': lr_list,
}

df = pd.DataFrame(history)
df['train_acc'] = df['train_acc'].map(lambda x: f"{x:.1f}%")
df['val_acc']   = df['val_acc'].map(lambda x: f"{x:.1f}%")
# exporta markdown pronto para colar no artigo
print(df.to_markdown(index=False))
df.to_csv('results/epoch_history.csv', index=False)
```

Cole a saída do `print(df.to_markdown(...))` no `article/medium_final.md` ou gere uma versão HTML para embutir no artigo.

## Apêndice D — Extra: comparação numérica (guia para tabelas de comparação)

Se desejar uma tabela comparativa entre otimizadores, um formato útil:

| optimizer | final_train_acc | final_val_acc | best_val_acc_epoch | total_time_s |
|---|---:|---:|---:|---:|
| SGD+momentum | 78.2% | 64.3% | 45 | 3600 |
| Adam         | 76.9% | 63.1% | 38 | 3400 |

Novamente: gere esses valores a partir dos logs de treino (salve `best_val_acc`, `epoch` e o tempo total usando `time.time()` nos pontos apropriados do seu script de treino).

---

Se quiser, eu injeto automaticamente a tabela real extraída dos logs que já foram gerados neste repositório (procuro por arquivos `results/` ou por variáveis em `notebooks/Cifar100.ipynb`) e atualizo o `.md` com os valores reais — quer que eu tente isso agora?  
