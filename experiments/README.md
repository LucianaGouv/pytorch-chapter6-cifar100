# Experimentos

Esta pasta contém scripts executáveis e utilitários usados para (re)executar os experimentos do Capítulo 6 no CIFAR-100.

Arquivos principais
- `Chapter6_CIFAR100_Complete.py`: script completo com EWMA, LR range test, captura de gradientes e loops de treino.

Como usar
- Execute os scripts no ambiente virtual do repositório. Para demonstrações rápidas, reduza `epochs`/batches; para experimentos completos, aumente-os.

Exemplo:
```bash
source .venv/bin/activate
python Chapter6_CIFAR100_Complete.py --epochs 24
```

Observações
- O notebook em `notebooks/` é o artefato principal para publicação; use `nbconvert` para reexecutar ou exportar para HTML.
