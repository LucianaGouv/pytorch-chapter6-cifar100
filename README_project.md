# PyTorch Capítulo 6 — CIFAR-100

Instruções e estrutura do projeto para o Trabalho Final.

Estrutura do repositório

- `notebooks/` — notebooks prontos para publicação (executados), incluindo `Cifar100.ipynb`.
- `experiments/` — scripts e utilitários para executar os experimentos completos.
- `figures/` — figuras geradas pelo notebook (PNG) usadas no artigo.
- `data/` — local para downloads do dataset (CIFAR-100). Evite commitar dados grandes.
- `Chapter6_CIFAR100_Complete.py` — script monolítico com os experimentos (cópia em `experiments/`).

Fluxo recomendado para reproduzir as saídas do notebook

1. Ative o ambiente virtual e instale dependências (o `.venv` já pode existir neste workspace):
```bash
source .venv/bin/activate
pip install -r requirements.txt
```
2. Re-execute o notebook em-place (isso incorpora as saídas e recria figuras):
```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/Cifar100.ipynb --ExecutePreprocessor.timeout=2400
```
3. Exporte para HTML para publicação (Medium/Substack):
```bash
.venv/bin/python -m nbconvert notebooks/Cifar100.ipynb --to html --output Cifar100_published.html
```

Onde as figuras ficam

- As figuras geradas pelo notebook são salvas em `figures/`. Ao reexecutar o notebook, novas versões serão escritas nessa pasta.

Se quiser, posso consolidar os arquivos nas pastas `notebooks/`, `experiments/` e `figures/` e ajustar referências nos READMEs.
